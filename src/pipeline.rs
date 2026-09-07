//! End-to-end pipeline documentation and shared pure-data helpers.
//!
//! ```text
//! input image
//!     v
//! face detection (SCRFD, ONNX Runtime Web, JS side)
//!     v
//! landmarks
//!     v
//! face alignment (estimate_norm -> warp, Rust/WASM here)
//!     v
//! source identity preparation (ArcFace embed + emap, cached once)
//!     v
//! face swap inference (inswapper_128, ONNX Runtime Web, JS side)
//!     v
//! inverse geometric transform (invert_affine -> warp back, Rust/WASM here)
//!     v
//! mask generation (crop-space erode+blur, Rust/WASM here)
//!     v
//! color/illumination adaptation (masked RGB moment match, Rust/WASM here)
//!     v
//! seamless blending (feathered composite, Rust/WASM here)
//!     v
//! final image (+ AI-manipulation disclosure, JS side)
//! ```
//!
//! The ONNX inference steps intentionally live in JS (`web/inference.js`):
//! ONNX Runtime Web is a JS API and cannot be driven from Rust without a
//! heavyweight bridge. Everything geometric / pixel-local lives here so it
//! is unit-tested in Rust and runs at near-native speed in WASM.

use crate::faces::Face;

/// Hard limits protecting the tab from OOM on gigapixel inputs.
pub struct PipelineLimits {
    /// Longest side (px) allowed for the *detection* pass input.
    pub max_detection_long_side: u32,
    /// Longest side (px) allowed for the full-resolution composition.
    pub max_output_long_side: u32,
    /// Maximum faces processed per image (DoS guard; UI warns).
    pub max_faces: usize,
}

impl Default for PipelineLimits {
    fn default() -> Self {
        Self {
            max_detection_long_side: 640,
            max_output_long_side: 4096,
            max_faces: 50,
        }
    }
}

/// Downscale factor (<=1) so the longest side fits `max_side`.
/// Detection runs on the downscaled image; coordinates map back with
/// `1/factor` (same idea as `SCRFD.detect`'s `det_scale`).
pub fn downscale_for_detection(w: u32, h: u32, max_side: u32) -> f32 {
    let long = w.max(h) as f32;
    if long <= max_side as f32 {
        1.0
    } else {
        max_side as f32 / long
    }
}

/// Downscale factor (<=1) so the output composition fits memory limits.
pub fn downscale_for_output(w: u32, h: u32, max_side: u32) -> f32 {
    downscale_for_detection(w, h, max_side)
}

/// Ordered, user-facing pipeline stages (mirrored by the JS progress UI).
pub fn stage_list(n_faces: usize) -> Vec<String> {
    let mut stages = vec![
        "Loading models...".to_string(),
        "Detecting faces...".to_string(),
        "Preparing source identity...".to_string(),
    ];
    for i in 0..n_faces {
        stages.push(format!("Swapping {} / {} faces...", i + 1, n_faces));
    }
    stages.push("Blending result...".to_string());
    stages.push("Done.".to_string());
    stages
}

/// Validate the job-level swap job before any heavy work: image dims sane,
/// at least one face, face count within limits.
pub fn validate_job(img_w: u32, img_h: u32, faces: &[Face], limits: &PipelineLimits) -> Result<(), String> {
    if img_w == 0 || img_h == 0 {
        return Err("Unsupported image: zero-size image".to_string());
    }
    if faces.is_empty() {
        return Err("No face detected".to_string());
    }
    if faces.len() > limits.max_faces {
        return Err(format!(
            "Too many faces: found {}, limit is {}",
            faces.len(),
            limits.max_faces
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faces::Face;

    fn f() -> Face {
        Face { bbox: [0.0, 0.0, 10.0, 10.0], landmarks: [[0.0; 2]; 5], score: 0.9 }
    }

    #[test]
    fn detection_downscale_only_shrinks() {
        assert_eq!(downscale_for_detection(4000, 3000, 640), 640.0 / 4000.0);
        assert_eq!(downscale_for_detection(512, 512, 640), 1.0);
    }

    #[test]
    fn stage_list_counts_faces() {
        let s = stage_list(3);
        assert_eq!(s.first().unwrap(), "Loading models...");
        assert_eq!(s.last().unwrap(), "Done.");
        assert!(s.iter().any(|x| x == "Swapping 3 / 3 faces..."));
        assert_eq!(s.len(), 3 + 3 + 2);
    }

    #[test]
    fn job_validation() {
        let lim = PipelineLimits::default();
        assert!(validate_job(100, 100, &[f()], &lim).is_ok());
        assert!(validate_job(100, 100, &[], &lim).is_err());
        assert!(validate_job(0, 100, &[f()], &lim).is_err());
        let many = vec![f(); 51];
        assert!(validate_job(100, 100, &many, &lim).is_err());
    }
}
