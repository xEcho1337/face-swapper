//! Face record types, confidence filtering and multi-face selection.
//!
//! Mirrors the post-NMS stage of the pipeline:
//! `confidence < threshold -> ignore`, otherwise process every face
//! independently with the same fixed source identity.

/// A detected face in original-image pixel coordinates.
#[derive(Debug, Clone)]
pub struct Face {
    /// xyxy bounding box.
    pub bbox: [f32; 4],
    /// 5 facial landmarks (left eye, right eye, nose, left mouth, right
    /// mouth) in original-image coordinates.
    pub landmarks: [[f32; 2]; 5],
    /// Detector confidence in [0,1].
    pub score: f32,
}

impl Face {
    pub fn area(&self) -> f32 {
        ((self.bbox[2] - self.bbox[0]).max(0.0)) * ((self.bbox[3] - self.bbox[1]).max(0.0))
    }

    pub fn center(&self) -> (f32, f32) {
        (
            (self.bbox[0] + self.bbox[2]) * 0.5,
            (self.bbox[1] + self.bbox[3]) * 0.5,
        )
    }

    /// Clamp the box (and nothing else) into `[0,w]x[0,h]`.
    pub fn clamp_bbox(&mut self, w: f32, h: f32) {
        self.bbox[0] = self.bbox[0].clamp(0.0, w);
        self.bbox[1] = self.bbox[1].clamp(0.0, h);
        self.bbox[2] = self.bbox[2].clamp(0.0, w);
        self.bbox[3] = self.bbox[3].clamp(0.0, h);
    }
}

/// Drop every face with `score < threshold`. Never invent faces: an empty
/// input yields an empty output (surfaced as `No face detected`).
pub fn filter_by_confidence(mut faces: Vec<Face>, threshold: f32) -> Vec<Face> {
    faces.retain(|f| f.score >= threshold);
    faces
}

/// Scale a bbox + landmarks from detection resolution to the original image
/// (`/ det_scale`, exactly like `SCRFD.detect`).
pub fn scale_face_to_original(face: &Face, det_scale: f32) -> Face {
    let landmarks = std::array::from_fn(|i| {
        [
            crate::image::map_to_original(face.landmarks[i][0], det_scale),
            crate::image::map_to_original(face.landmarks[i][1], det_scale),
        ]
    });
    Face {
        bbox: [
            crate::image::map_to_original(face.bbox[0], det_scale),
            crate::image::map_to_original(face.bbox[1], det_scale),
            crate::image::map_to_original(face.bbox[2], det_scale),
            crate::image::map_to_original(face.bbox[3], det_scale),
        ],
        landmarks,
        score: face.score,
    }
}

/// Cap the number of processed faces, preferring large, centered faces —
/// the same `area - 2*offset_dist^2` metric `SCRFD.detect(max_num=...)` uses.
pub fn select_top_faces(faces: Vec<Face>, max_num: usize, img_w: f32, img_h: f32) -> Vec<Face> {
    if max_num == 0 || faces.len() <= max_num {
        return faces;
    }
    let cx = img_w * 0.5;
    let cy = img_h * 0.5;
    let mut scored: Vec<(f32, Face)> = faces
        .into_iter()
        .map(|f| {
            let (fx, fy) = f.center();
            let off2 = (fx - cx).powi(2) + (fy - cy).powi(2);
            (f.area() - 2.0 * off2, f)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(max_num).map(|(_, f)| f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Face {
        Face {
            bbox: [x1, y1, x2, y2],
            landmarks: [[0.0; 2]; 5],
            score,
        }
    }

    #[test]
    fn threshold_boundary_is_inclusive() {
        let faces = vec![face(0.49, 0.0, 0.0, 10.0, 10.0), face(0.5, 0.0, 0.0, 10.0, 10.0)];
        let kept = filter_by_confidence(faces, 0.5);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_in_empty_out() {
        assert!(filter_by_confidence(vec![], 0.5).is_empty());
    }

    #[test]
    fn scaling_divides_by_det_scale() {
        let mut lm = [[0.0; 2]; 5];
        lm[0] = [64.0, 32.0];
        let f = Face { bbox: [0.0, 0.0, 64.0, 64.0], landmarks: lm, score: 0.9 };
        let o = scale_face_to_original(&f, 0.5);
        assert_eq!(o.bbox, [0.0, 0.0, 128.0, 128.0]);
        assert_eq!(o.landmarks[0], [128.0, 64.0]);
    }

    #[test]
    fn top_faces_prefers_large_centered() {
        // Small centered vs huge corner: huge corner wins on area... use two
        // faces where centering decides between equals.
        let a = face(0.9, 0.0, 0.0, 100.0, 100.0); // corner-ish
        let b = face(0.9, 450.0, 450.0, 550.0, 550.0); // centered (img 1000²)
        let picked = select_top_faces(vec![a, b], 1, 1000.0, 1000.0);
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].bbox, [450.0, 450.0, 550.0, 550.0]);
    }
}
