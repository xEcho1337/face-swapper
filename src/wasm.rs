//! Minimal JS bridge: exposes the Rust pixel/geometry core to the browser.
//!
//! Design rule: **everything the browser cannot do conveniently from Rust**
//! (fetch, WebGPU/ORT inference, canvas, workers) stays in JS
//! (`web/inference.js`, `web/worker.js`, `web/app.js`). This module only
//! exposes hot pixel/geometry kernels operating on typed arrays, so large
//! buffers cross the boundary by copy-once `Uint8Array`/`Float32Array`
//! (WASM has no shared view of JS memory without extra plumbing; the worker
//! keeps transfers to one copy per stage and reuses buffers).

use wasm_bindgen::prelude::*;

use crate::{blending, faces::Face, geometry, image};

/// `estimate_norm` for JS: `landmarks` is 10 floats (5x2, x0,y0,...),
/// returns 6 floats (2x3 row-major) or throws on degenerate input.
#[wasm_bindgen(js_name = estimateNorm)]
pub fn estimate_norm_js(landmarks: &[f32], image_size: u32) -> Result<Vec<f32>, JsValue> {
    if landmarks.len() != 10 {
        return Err(JsValue::from_str("estimateNorm: need 10 floats (5x2 landmarks)"));
    }
    let mut lm = [[0.0f32; 2]; 5];
    for i in 0..5 {
        lm[i] = [landmarks[i * 2], landmarks[i * 2 + 1]];
    }
    geometry::estimate_norm(&lm, image_size)
        .map(|m| m.to_vec())
        .ok_or_else(|| JsValue::from_str("estimateNorm: degenerate landmarks"))
}

/// Invert a 2x3 matrix (6 floats) or throw if singular.
#[wasm_bindgen(js_name = invertAffine)]
pub fn invert_affine_js(m: &[f32]) -> Result<Vec<f32>, JsValue> {
    if m.len() != 6 {
        return Err(JsValue::from_str("invertAffine: need 6 floats"));
    }
    let a: geometry::Affine2x3 = [m[0], m[1], m[2], m[3], m[4], m[5]];
    geometry::invert_affine(&a)
        .map(|v| v.to_vec())
        .ok_or_else(|| JsValue::from_str("invertAffine: singular matrix"))
}

/// `cv2.warpAffine`-equivalent on RGBA bytes.
#[wasm_bindgen(js_name = warpRgba)]
pub fn warp_rgba_js(
    src: &[u8],
    sw: u32,
    sh: u32,
    m: &[f32],
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, JsValue> {
    if m.len() != 6 {
        return Err(JsValue::from_str("warpRgba: need 6 floats"));
    }
    if src.len() != (sw * sh * 4) as usize {
        return Err(JsValue::from_str("warpRgba: src length mismatch"));
    }
    Ok(image::warp_affine_rgba(src, sw, sh, &[m[0], m[1], m[2], m[3], m[4], m[5]], dw, dh))
}

/// Bilinear RGBA resize.
#[wasm_bindgen(js_name = resizeRgba)]
pub fn resize_rgba_js(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Result<Vec<u8>, JsValue> {
    if src.len() != (sw * sh * 4) as usize || dw == 0 || dh == 0 {
        return Err(JsValue::from_str("resizeRgba: bad dims"));
    }
    Ok(image::resize_rgba_bilinear(src, sw, sh, dw, dh))
}

/// Bicubic `warpAffine`-equivalent on RGBA bytes (replicate border).
/// For the paste-back magnification of the 128px swap output; masks stay
/// on the bilinear warp.
#[wasm_bindgen(js_name = warpRgbaBicubic)]
pub fn warp_rgba_bicubic_js(
    src: &[u8],
    sw: u32,
    sh: u32,
    m: &[f32],
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, JsValue> {
    if m.len() != 6 {
        return Err(JsValue::from_str("warpRgbaBicubic: need 6 floats"));
    }
    if src.len() != (sw * sh * 4) as usize {
        return Err(JsValue::from_str("warpRgbaBicubic: src length mismatch"));
    }
    Ok(image::warp_affine_rgba_bicubic(
        src,
        sw,
        sh,
        &[m[0], m[1], m[2], m[3], m[4], m[5]],
        dw,
        dh,
    ))
}

/// Letterbox geometry: returns `[new_w, new_h, det_scale]`.
#[wasm_bindgen(js_name = letterboxGeometry)]
pub fn letterbox_geometry_js(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
    let (nw, nh, s) = image::letterbox_geometry(src_w, src_h, dst_w, dst_h);
    vec![nw as f32, nh as f32, s]
}

/// Build the feathered paste-back mask in crop space. `white_bordered` is a
/// `size*size` f32 mask (255 interior, 0 on the 2px border). Returns
/// `size*size` weights in [0,1].
#[wasm_bindgen(js_name = buildCropMask)]
pub fn build_crop_mask_js(white_bordered: &[f32], size: u32) -> Result<Vec<f32>, JsValue> {
    if white_bordered.len() != (size * size) as usize {
        return Err(JsValue::from_str("buildCropMask: length mismatch"));
    }
    Ok(blending::build_crop_mask(white_bordered, size))
}

/// Masked RGB moment-match of the swapped crop onto the target crop.
/// Buffers are `npix*3` RGB bytes; `mask` is `npix` [0,1] weights.
#[wasm_bindgen(js_name = colorMatchCrop)]
pub fn color_match_crop_js(
    target_rgb: &[u8],
    swap_rgb: &[u8],
    mask: &[f32],
    npix: u32,
) -> Result<Vec<u8>, JsValue> {
    let n = npix as usize;
    if target_rgb.len() != n * 3 || swap_rgb.len() != n * 3 || mask.len() != n {
        return Err(JsValue::from_str("colorMatchCrop: length mismatch"));
    }
    Ok(blending::color_match_crop(target_rgb, swap_rgb, mask, n))
}

/// Feathered elliptical paste-back mask (Deep-Live-Cam style), `size*size`
/// weights in [0,1]. Replaces the square `buildCropMask` for the swap path.
#[wasm_bindgen(js_name = buildEllipticalMask)]
pub fn build_elliptical_mask_js(size: u32) -> Result<Vec<f32>, JsValue> {
    if size == 0 {
        return Err(JsValue::from_str("buildEllipticalMask: bad size"));
    }
    Ok(blending::build_elliptical_mask(size))
}

/// Light denoise of an `npix*3` RGB buffer (`amount` 0..1).
#[wasm_bindgen(js_name = softenRgb)]
pub fn soften_rgb_js(rgb: &[u8], w: u32, h: u32, amount: f32) -> Result<Vec<u8>, JsValue> {
    if rgb.len() != (w * h * 3) as usize {
        return Err(JsValue::from_str("softenRgb: length mismatch"));
    }
    Ok(blending::soften_rgb(rgb, w, h, amount))
}

/// Unsharp mask of an `npix*3` RGB buffer (`strength` 0..2).
#[wasm_bindgen(js_name = sharpenRgb)]
pub fn sharpen_rgb_js(rgb: &[u8], w: u32, h: u32, strength: f32) -> Result<Vec<u8>, JsValue> {
    if rgb.len() != (w * h * 3) as usize {
        return Err(JsValue::from_str("sharpenRgb: length mismatch"));
    }
    Ok(blending::sharpen_rgb(rgb, w, h, strength))
}

/// Full-resolution feathered composite. `target` RGBA, `swap` RGB,
/// `mask` [0,1] weights (already warped back). Returns RGBA.
#[wasm_bindgen(js_name = blendFullres)]
pub fn blend_fullres_js(
    target: &[u8],
    swap: &[u8],
    mask: &[f32],
    w: u32,
    h: u32,
) -> Result<Vec<u8>, JsValue> {
    let n = (w * h) as usize;
    if target.len() != n * 4 || swap.len() != n * 3 || mask.len() != n {
        return Err(JsValue::from_str("blendFullres: length mismatch"));
    }
    Ok(blending::blend_fullres(target, swap, mask, w, h))
}

/// Greedy NMS: `boxes` flat xyxy, `scores` flat. Returns kept indices.
#[wasm_bindgen(js_name = nmsBoxes)]
pub fn nms_boxes_js(boxes: &[f32], scores: &[f32], iou_thresh: f32) -> Result<Vec<u32>, JsValue> {
    if boxes.len() % 4 != 0 || boxes.len() / 4 != scores.len() {
        return Err(JsValue::from_str("nmsBoxes: length mismatch"));
    }
    let b: Vec<[f32; 4]> = boxes
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    Ok(geometry::nms(&b, scores, iou_thresh)
        .into_iter()
        .map(|i| i as u32)
        .collect())
}

/// Confidence filter + top-face selection helper for the debug overlay path.
/// Returns indices (into the input order) of faces to process.
#[wasm_bindgen(js_name = selectFaces)]
pub fn select_faces_js(
    boxes: &[f32],
    scores: &[f32],
    threshold: f32,
    max_faces: u32,
    img_w: f32,
    img_h: f32,
) -> Result<Vec<u32>, JsValue> {
    if boxes.len() % 4 != 0 || boxes.len() / 4 != scores.len() {
        return Err(JsValue::from_str("selectFaces: length mismatch"));
    }
    let faces: Vec<(usize, Face)> = boxes
        .chunks_exact(4)
        .zip(scores.iter())
        .enumerate()
        .filter(|(_, (_, s))| **s >= threshold)
        .map(|(i, (b, s))| {
            (
                i,
                Face {
                    bbox: [b[0], b[1], b[2], b[3]],
                    landmarks: [[0.0; 2]; 5],
                    score: *s,
                },
            )
        })
        .collect();
    let max = max_faces as usize;
    if max == 0 || faces.len() <= max {
        return Ok(faces.into_iter().map(|(i, _)| i as u32).collect());
    }
    // Same center-weighted metric as faces::select_top_faces, but keep the
    // original indices.
    let cx = img_w * 0.5;
    let cy = img_h * 0.5;
    let mut scored: Vec<(f32, usize)> = faces
        .iter()
        .map(|(i, f)| {
            let (fx, fy) = f.center();
            let off2 = (fx - cx).powi(2) + (fy - cy).powi(2);
            (f.area() - 2.0 * off2, *i)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(scored.into_iter().take(max).map(|(_, i)| i as u32).collect())
}
