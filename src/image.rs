//! Image primitives for the WASM pipeline: warp, resize, letterbox, tensors.
//!
//! Conventions (must match the JS side and InsightFace Python reference):
//! - Pixels are RGBA8, row-major.
//! - `warp_affine_rgba` is an exact `cv2.warpAffine` equivalent with default
//!   flags and `borderValue=0`: OpenCV **inverts `M` first** (unless
//!   `WARP_INVERSE` is set), i.e. `dst(x,y) = src(M^-1 . (x,y))` with
//!   bilinear sampling. Callers therefore pass the SAME matrices as the
//!   Python reference: `M = estimate_norm(...)` for the aligned crop,
//!   `IM = invert(M)` for paste-back.
//! - SCRFD letterbox: aspect-preserving resize into a fixed canvas with the
//!   resized image at the top-left and zero padding (see
//!   `SCRFD.detect` in `detection/scrfd/tools/scrfd.py`).
//! - Tensor layout for ONNX Runtime Web is NCHW float32.

/// Bilinear sample of an RGBA8 image with clamp-to-edge (replicate) border.
fn sample_bilinear_replicate(src: &[u8], w: u32, h: u32, x: f32, y: f32) -> [f32; 4] {
    let w = w as i32;
    let h = h as i32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0.0f32; 4];
    for dy in 0..2 {
        for dx in 0..2 {
            let px = (x0 + dx).clamp(0, w - 1);
            let py = (y0 + dy).clamp(0, h - 1);
            let wx = if dx == 0 { 1.0 - fx } else { fx };
            let wy = if dy == 0 { 1.0 - fy } else { fy };
            let weight = wx * wy;
            if weight == 0.0 {
                continue;
            }
            let o = ((py as u32 * w as u32 + px as u32) * 4) as usize;
            for c in 0..4 {
                out[c] += src[o + c] as f32 * weight;
            }
        }
    }
    out
}

/// Warp like [`warp_affine_rgba`] (same matrix convention) but with bilinear
/// sampling and replicate borders — the combination Deep-Live-Cam uses for
/// its paste-back (`cv2.warpAffine(..., INTER_LINEAR, BORDER_REPLICATE)`).
/// Replicate (instead of zero) keeps the feathered rim from darkening when
/// the 128/256px swap output is magnified to full resolution. Masks keep
/// using the zero-border warp.
pub fn warp_affine_rgba_replicate(
    src: &[u8],
    sw: u32,
    sh: u32,
    m: &crate::geometry::Affine2x3,
    dw: u32,
    dh: u32,
) -> Vec<u8> {
    assert_eq!(src.len(), (sw * sh * 4) as usize);
    let inv = crate::geometry::invert_affine(m);
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    let inv = match inv {
        Some(v) => v,
        None => return out,
    };
    for y in 0..dh {
        for x in 0..dw {
            let sx = inv[0] * x as f32 + inv[1] * y as f32 + inv[2];
            let sy = inv[3] * x as f32 + inv[4] * y as f32 + inv[5];
            let s = sample_bilinear_replicate(src, sw, sh, sx, sy);
            let o = ((y * dw + x) * 4) as usize;
            for c in 0..4 {
                out[o + c] = s[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Bilinear sample of an RGBA8 image with zero border. Returns [r,g,b,a].
fn sample_bilinear(src: &[u8], w: u32, h: u32, x: f32, y: f32) -> [f32; 4] {
    let w = w as i32;
    let h = h as i32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0.0f32; 4];
    for dy in 0..2 {
        for dx in 0..2 {
            let px = x0 + dx;
            let py = y0 + dy;
            let wx = if dx == 0 { 1.0 - fx } else { fx };
            let wy = if dy == 0 { 1.0 - fy } else { fy };
            let weight = wx * wy;
            if weight == 0.0 {
                continue;
            }
            if px >= 0 && py >= 0 && px < w && py < h {
                let o = ((py as u32 * w as u32 + px as u32) * 4) as usize;
                for c in 0..4 {
                    out[c] += src[o + c] as f32 * weight;
                }
            }
        }
    }
    out
}

/// Warp `src` (RGBA8 `sw`x`sh`) with 2x3 matrix `m` into a `dw`x`dh` RGBA8
/// buffer — equivalent to
/// `cv2.warpAffine(src, M, (dw, dh), borderValue=0.0)` with bilinear
/// interpolation and default flags. Like OpenCV, `m` is inverted internally:
/// `dst(x,y) = src(M^-1 . (x,y))`. A singular `m` yields a black image
/// (OpenCV would raise; we stay total and let callers validate matrices).
pub fn warp_affine_rgba(
    src: &[u8],
    sw: u32,
    sh: u32,
    m: &crate::geometry::Affine2x3,
    dw: u32,
    dh: u32,
) -> Vec<u8> {
    assert_eq!(src.len(), (sw * sh * 4) as usize);
    let inv = crate::geometry::invert_affine(m);
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    let inv = match inv {
        Some(v) => v,
        None => return out,
    };
    for y in 0..dh {
        for x in 0..dw {
            let sx = inv[0] * x as f32 + inv[1] * y as f32 + inv[2];
            let sy = inv[3] * x as f32 + inv[4] * y as f32 + inv[5];
            let s = sample_bilinear(src, sw, sh, sx, sy);
            let o = ((y * dw + x) * 4) as usize;
            for c in 0..4 {
                out[o + c] = s[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Catmull-Rom cubic kernel (a = -0.5).
fn cubic_kernel(x: f32) -> f32 {
    let ax = x.abs();
    if ax <= 1.0 {
        1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0
    } else if ax < 2.0 {
        -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0
    } else {
        0.0
    }
}

/// Bicubic sample of an RGBA8 image with clamp-to-edge (replicate) border.
/// Clamping (rather than the zero border of [`warp_affine_rgba`]) mirrors
/// what Deep-Live-Cam does for its paste-back warp
/// (`cv2.warpAffine(..., borderMode=cv2.BORDER_REPLICATE)`): magnifying the
/// 128px swap output with a zero border would darken the feathered rim,
/// while replicate keeps edge pixels stable under the mask.
fn sample_bicubic_replicate(src: &[u8], w: u32, h: u32, x: f32, y: f32) -> [f32; 4] {
    let (w, h) = (w as i32, h as i32);
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let mut out = [0.0f32; 4];
    for dy in -1..=2 {
        let py = (y0 + dy).clamp(0, h - 1);
        let ky = cubic_kernel(y - (y0 + dy) as f32);
        if ky == 0.0 {
            continue;
        }
        for dx in -1..=2 {
            let px = (x0 + dx).clamp(0, w - 1);
            let kx = cubic_kernel(x - (x0 + dx) as f32);
            let weight = kx * ky;
            if weight == 0.0 {
                continue;
            }
            let o = ((py as u32 * w as u32 + px as u32) * 4) as usize;
            for c in 0..4 {
                out[c] += src[o + c] as f32 * weight;
            }
        }
    }
    out
}

/// Warp like [`warp_affine_rgba`] (same matrix convention: `m` is inverted
/// internally) but with bicubic sampling and replicate borders. Intended for
/// the paste-back of the 128px swap output to full resolution, where
/// bilinear magnification looks blocky. Masks must keep going through the
/// bilinear warp (weights must never overshoot [0,1]).
pub fn warp_affine_rgba_bicubic(
    src: &[u8],
    sw: u32,
    sh: u32,
    m: &crate::geometry::Affine2x3,
    dw: u32,
    dh: u32,
) -> Vec<u8> {
    assert_eq!(src.len(), (sw * sh * 4) as usize);
    let inv = crate::geometry::invert_affine(m);
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    let inv = match inv {
        Some(v) => v,
        None => return out,
    };
    for y in 0..dh {
        for x in 0..dw {
            let sx = inv[0] * x as f32 + inv[1] * y as f32 + inv[2];
            let sy = inv[3] * x as f32 + inv[4] * y as f32 + inv[5];
            let s = sample_bicubic_replicate(src, sw, sh, sx, sy);
            let o = ((y * dw + x) * 4) as usize;
            for c in 0..4 {
                out[o + c] = s[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Bilinear resize of an RGBA8 image (clamp-to-edge borders, like canvas
/// `drawImage` for interior/edge pixels — unlike `warp_affine_rgba`, which
/// uses zero borders to match `cv2.warpAffine(borderValue=0)`).
pub fn resize_rgba_bilinear(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    assert_eq!(src.len(), (sw * sh * 4) as usize);
    assert!(dw > 0 && dh > 0);
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, sw as f32 - 1.0);
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, sh as f32 - 1.0);
            let x0 = fx.floor() as u32;
            let y0 = fy.floor() as u32;
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let o = ((y * dw + x) * 4) as usize;
            for c in 0..4 {
                let p00 = src[((y0 * sw + x0) * 4) as usize + c] as f32;
                let p10 = src[((y0 * sw + x1) * 4) as usize + c] as f32;
                let p01 = src[((y1 * sw + x0) * 4) as usize + c] as f32;
                let p11 = src[((y1 * sw + x1) * 4) as usize + c] as f32;
                let v = p00 * (1.0 - tx) * (1.0 - ty)
                    + p10 * tx * (1.0 - ty)
                    + p01 * (1.0 - tx) * ty
                    + p11 * tx * ty;
                out[o + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Letterbox geometry used by `SCRFD.detect`: fit `src_w`x`src_h` into
/// `dst_w`x`dst_h` preserving aspect ratio, top-left aligned.
/// Returns `(new_w, new_h, det_scale)` where `det_scale = new_h / src_h`.
/// Model-space coords map back to image space by dividing by `det_scale`.
pub fn letterbox_geometry(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> (u32, u32, f32) {
    assert!(src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0);
    let im_ratio = src_h as f64 / src_w as f64;
    let model_ratio = dst_h as f64 / dst_w as f64;
    let (new_w, new_h) = if im_ratio > model_ratio {
        let nh = dst_h;
        let nw = ((nh as f64 / im_ratio).round() as u32).max(1).min(dst_w);
        (nw, nh)
    } else {
        let nw = dst_w;
        let nh = ((nw as f64 * im_ratio).round() as u32).max(1).min(dst_h);
        (nw, nh)
    };
    (new_w, new_h, new_h as f32 / src_h as f32)
}

/// Map a detection-space coordinate back to original-image space.
pub fn map_to_original(v: f32, det_scale: f32) -> f32 {
    v / det_scale
}

/// Validate an ONNX tensor shape against an expectation, allowing symbolic /
/// dynamic dims. `actual` and `expected` are dim lists; use `None` in
/// `expected` for "any size" (dynamic axis) and `Some(-1)` is rejected — pass
/// `None` instead. Returns `Ok(())` or a human-readable error (surfaced in
/// the UI as `Invalid model`).
pub fn validate_tensor_shape(
    tensor_name: &str,
    actual: &[i64],
    expected: &[Option<i64>],
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "Invalid model: tensor '{tensor_name}' rank {} != expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if let Some(want) = e {
            if a != want {
                return Err(format!(
                    "Invalid model: tensor '{tensor_name}' dim {i} is {a}, expected {want}"
                ));
            }
        }
    }
    Ok(())
}

/// Check an ONNX dtype string (as reported by ORT Web, e.g. `"float32"`).
pub fn validate_dtype(tensor_name: &str, actual: &str, expected: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "Invalid model: tensor '{tensor_name}' dtype is '{actual}', expected '{expected}'"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_landscape_matches_scrfd_detect() {
        // img 800x400 (w,h), model 640x640: im_ratio=0.5 < model_ratio=1
        // -> new_w=640, new_h=320, scale=320/400=0.8
        let (nw, nh, s) = letterbox_geometry(800, 400, 640, 640);
        assert_eq!((nw, nh), (640, 320));
        assert!((s - 0.8).abs() < 1e-6);
    }

    #[test]
    fn letterbox_portrait_matches_scrfd_detect() {
        // img 400x800: im_ratio=2 > 1 -> new_h=640, new_w=320, scale=0.8
        let (nw, nh, s) = letterbox_geometry(400, 800, 640, 640);
        assert_eq!((nw, nh), (320, 640));
        assert!((s - 0.8).abs() < 1e-6);
    }

    #[test]
    fn letterbox_square_full_canvas() {
        let (nw, nh, s) = letterbox_geometry(512, 512, 640, 640);
        assert_eq!((nw, nh), (640, 640));
        assert!((s - 1.25).abs() < 1e-6);
    }

    #[test]
    fn warp_identity_roundtrips_pixels() {
        let w = 8;
        let h = 6;
        let mut src = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            src[i * 4] = (i * 3 % 256) as u8;
            src[i * 4 + 1] = (i * 5 % 256) as u8;
            src[i * 4 + 2] = (i * 7 % 256) as u8;
            src[i * 4 + 3] = 255;
        }
        let ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let out = warp_affine_rgba(&src, w, h, &ident, w, h);
        assert_eq!(out, src);
    }

    #[test]
    fn warp_translate_shifts_content() {
        // Under cv2.warpAffine default semantics M is inverted internally:
        // out(x) = src(M^-1 . x). With M = [1,0,-1; 0,1,0], M^-1 shifts by
        // +1, so out[x] = src[x+1]: content moves LEFT by one pixel.
        let src = vec![10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 0, 0, 255];
        let m = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0];
        let out = warp_affine_rgba(&src, 4, 1, &m, 4, 1);
        assert_eq!(out[0], 20);
        assert_eq!(out[4], 30);
        assert_eq!(out[8], 40);
        assert_eq!(out[12], 0); // border
    }

    #[test]
    fn warp_aligns_landmark_to_template_like_norm_crop() {
        // Image landmark at (30, 40) must land on the template point after
        // warping with M = estimate_norm — the norm_crop2 contract.
        use crate::geometry::{apply_affine, estimate_norm};
        let mut lm = crate::geometry::ARCFACE_DST_112;
        // Pretend the detected face sits at an offset: shift template by
        // (+30, +40) to fake "image landmarks".
        for p in lm.iter_mut() {
            p[0] += 30.0;
            p[1] += 40.0;
        }
        let m = estimate_norm(&lm, 112).unwrap();
        // M maps image landmark -> template landmark.
        let (tx, ty) = apply_affine(&m, lm[0][0], lm[0][1]);
        assert!((tx - crate::geometry::ARCFACE_DST_112[0][0]).abs() < 1e-3);
        assert!((ty - crate::geometry::ARCFACE_DST_112[0][1]).abs() < 1e-3);
        // And the warp (which inverts M like cv2) puts a bright image pixel
        // at the template location in the crop.
        let (w, h) = (128u32, 128u32);
        let mut src = vec![0u8; (w * h * 4) as usize];
        let ix = lm[2][0] as u32;
        let iy = lm[2][1] as u32;
        let o = ((iy * w + ix) * 4) as usize;
        src[o] = 255;
        src[o + 3] = 255;
        let crop = warp_affine_rgba(&src, w, h, &m, 112, 112);
        // Brightest crop pixel must be near template nose point.
        let mut best = (0u32, 0u32, 0u8);
        for y in 0..112 {
            for x in 0..112 {
                let v = crop[((y * 112 + x) * 4) as usize];
                if v > best.2 {
                    best = (x, y, v);
                }
            }
        }
        assert!(best.2 > 0, "warp produced a black crop");
        let gx = crate::geometry::ARCFACE_DST_112[2][0];
        let gy = crate::geometry::ARCFACE_DST_112[2][1];
        assert!(
            (best.0 as f32 - gx).abs() <= 1.0 && (best.1 as f32 - gy).abs() <= 1.0,
            "bright pixel at {:?}, want ≈ ({gx},{gy})",
            (best.0, best.1)
        );
    }

    #[test]
    fn warp_singular_matrix_yields_black() {
        let src = vec![255u8; 4 * 4 * 4];
        let out = warp_affine_rgba(&src, 4, 4, &[1.0, 2.0, 3.0, 2.0, 4.0, 5.0], 4, 4);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn warp_replicate_identity_roundtrips() {
        let w = 8;
        let h = 6;
        let mut src = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            src[i * 4] = (i * 3 % 256) as u8;
            src[i * 4 + 1] = (i * 5 % 256) as u8;
            src[i * 4 + 2] = (i * 7 % 256) as u8;
            src[i * 4 + 3] = 255;
        }
        let ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let out = warp_affine_rgba_replicate(&src, w, h, &ident, w, h);
        assert_eq!(out, src);
    }

    #[test]
    fn warp_replicate_clamps_border_instead_of_zero() {
        // 4x1 row; shift content right by one (M = [1,0,+1; 0,1,0] samples
        // src[x-1]): the leftmost pixel has no source — zero-border gives 0,
        // replicate repeats the edge pixel.
        let src = vec![10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 0, 0, 255];
        let m = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let zero = warp_affine_rgba(&src, 4, 1, &m, 4, 1);
        assert_eq!(zero[0], 0);
        let rep = warp_affine_rgba_replicate(&src, 4, 1, &m, 4, 1);
        assert_eq!(rep[0], 10);
        assert_eq!(rep[4], 10);
        assert_eq!(rep[8], 20);
        assert_eq!(rep[12], 30);
    }

    #[test]
    fn warp_bicubic_identity_near_roundtrip() {
        let w = 8;
        let h = 6;
        let mut src = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            src[i * 4] = (i * 3 % 256) as u8;
            src[i * 4 + 1] = (i * 5 % 256) as u8;
            src[i * 4 + 2] = (i * 7 % 256) as u8;
            src[i * 4 + 3] = 255;
        }
        let ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let out = warp_affine_rgba_bicubic(&src, w, h, &ident, w, h);
        // Bicubic interpolates exactly at integer coords (up to fp error).
        for (a, b) in out.iter().zip(src.iter()) {
            assert!((*a as i16 - *b as i16).abs() <= 1, "{a} vs {b}");
        }
    }

    #[test]
    fn warp_bicubic_constant_stays_constant() {
        let src = vec![77u8, 88, 99, 255].repeat(16 * 16);
        // Translate by half a pixel: replicate border keeps it constant.
        let m = [1.0, 0.0, 0.5, 0.0, 1.0, 0.5];
        let out = warp_affine_rgba_bicubic(&src, 16, 16, &m, 32, 32);
        // All RGB channels constant; alpha constant too.
        for i in 0..(32 * 32) as usize {
            assert_eq!(out[i * 4], 77);
            assert_eq!(out[i * 4 + 1], 88);
            assert_eq!(out[i * 4 + 2], 99);
            assert_eq!(out[i * 4 + 3], 255);
        }
    }

    #[test]
    fn warp_bicubic_singular_matrix_yields_black() {
        let src = vec![255u8; 4 * 4 * 4];
        let out = warp_affine_rgba_bicubic(&src, 4, 4, &[1.0, 2.0, 3.0, 2.0, 4.0, 5.0], 4, 4);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn resize_output_dims_and_corners() {
        let src = vec![
            255, 0, 0, 255, 0, 255, 0, 255, //
            0, 0, 255, 255, 255, 255, 255, 255,
        ];
        let out = resize_rgba_bilinear(&src, 2, 2, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 4);
        // Corners stay close to source corners.
        assert!(out[0] > 200);
        assert!(out[(3 * 4) + 1] > 150); // top-right greenish
    }

    #[test]
    fn tensor_shape_validation() {
        assert!(validate_tensor_shape("input.1", &[1, 3, 640, 640], &[Some(1), Some(3), None, None]).is_ok());
        assert!(validate_tensor_shape("img", &[1, 3, 128, 128], &[Some(1), Some(3), Some(128), Some(128)]).is_ok());
        assert!(validate_tensor_shape("img", &[1, 3, 112, 112], &[Some(1), Some(3), Some(128), Some(128)]).is_err());
        assert!(validate_tensor_shape("x", &[1, 3], &[Some(1), Some(3), Some(3)]).is_err());
        assert!(validate_dtype("img", "float32", "float32").is_ok());
        assert!(validate_dtype("img", "float16", "float32").is_err());
    }

    #[test]
    fn map_to_original_inverts_scale() {
        assert!((map_to_original(320.0, 0.8) - 400.0).abs() < 1e-5);
    }
}
