//! 2D similarity-transform geometry for face alignment.
//!
//! This module is a dependency-free Rust port of the alignment math used by
//! InsightFace (`python-package/insightface/utils/face_align.py`):
//!
//! - Reference template `arcface_dst` (112x112):
//!   [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
//!    [41.5493, 92.3655], [70.7299, 92.2041]]
//! - `estimate_norm(lmk, image_size)`: scales the template
//!   (`ratio = size/112`, `diff_x = 0`; or for 128-based crops
//!   `ratio = size/128`, `diff_x = 8*ratio`) then estimates a
//!   `SimilarityTransform` mapping detected landmarks -> template.
//! - `norm_crop2` warps with `cv2.warpAffine(img, M, (size, size))` where
//!   `M` is the 2x3 matrix mapping *output* pixel coords to *input* coords
//!   (OpenCV convention: `dst(x,y) = src(M11*x+M12*y+M13, ...)`).
//!
//! Estimation itself is the 2D Umeyama/Kabsch closed form (no SVD needed in
//! 2D): the optimal rotation `R = [[c,-s],[s,c]]` maximises `tr(R^T H)` with
//! `H = sum(src_c * dst_c^T)/n`, giving `c = a/hypot(a,b)`,
//! `s = b/hypot(a,b)`, `a = H11+H22`, `b = H21-H12`, and
//! `scale = hypot(a,b)/var(src)`. This matches
//! `skimage.transform.SimilarityTransform.estimate` for non-degenerate
//! 5-point inputs (verified by unit tests against the OpenCV round-trip).

/// Row-major 2x3 affine matrix `[a, b, tx, c, d, ty]` in OpenCV convention:
/// `sample_x = a*x + b*y + tx`, `sample_y = c*x + d*y + ty`.
pub type Affine2x3 = [f32; 6];

/// InsightFace arcface reference template for 112x112 crops.
pub const ARCFACE_DST_112: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Scale the arcface template to `image_size`, exactly like
/// `face_align.estimate_norm`.
///
/// - `size % 112 == 0` -> `ratio = size/112`, `diff_x = 0`
/// - otherwise (128-based, e.g. inswapper) -> `ratio = size/128`,
///   `diff_x = 8*ratio`
pub fn arcface_dst_for_size(image_size: u32) -> [[f32; 2]; 5] {
    let size = image_size as f32;
    let (ratio, diff_x) = if image_size % 112 == 0 {
        (size / 112.0, 0.0)
    } else {
        let r = size / 128.0;
        (r, 8.0 * r)
    };
    let mut dst = [[0.0f32; 2]; 5];
    for i in 0..5 {
        dst[i][0] = ARCFACE_DST_112[i][0] * ratio + diff_x;
        dst[i][1] = ARCFACE_DST_112[i][1] * ratio;
    }
    dst
}

/// Estimate the similarity transform mapping `src` landmarks onto `dst`
/// landmarks (both 5x2). Returns `None` for degenerate inputs
/// (zero variance / coincident points).
pub fn estimate_similarity_transform(
    src: &[[f32; 2]; 5],
    dst: &[[f32; 2]; 5],
) -> Option<Affine2x3> {
    // Centroids (f64 for stability).
    let (mut csx, mut csy, mut cdx, mut cdy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..5 {
        csx += src[i][0] as f64;
        csy += src[i][1] as f64;
        cdx += dst[i][0] as f64;
        cdy += dst[i][1] as f64;
    }
    csx /= 5.0;
    csy /= 5.0;
    cdx /= 5.0;
    cdy /= 5.0;

    // Variance of src + cross-covariance H = sum(s* d^T)/n.
    let mut var = 0.0f64;
    let (mut h11, mut h12, mut h21, mut h22) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..5 {
        let sx = src[i][0] as f64 - csx;
        let sy = src[i][1] as f64 - csy;
        let dx = dst[i][0] as f64 - cdx;
        let dy = dst[i][1] as f64 - cdy;
        var += sx * sx + sy * sy;
        h11 += sx * dx;
        h12 += sx * dy;
        h21 += sy * dx;
        h22 += sy * dy;
    }
    var /= 5.0;
    if !(var > 1e-12) {
        return None;
    }
    h11 /= 5.0;
    h12 /= 5.0;
    h21 /= 5.0;
    h22 /= 5.0;

    // Optimal 2D rotation (Kabsch): minimise ||R*s - d||^2, i.e. maximise
    // tr(R*H). With R = [[ce,-se],[se,ce]] this is ce*a - se*b for
    // a = H11+H22, b = H21-H12, hence (ce,se) = (a,-b)/hypot(a,b).
    let a = h11 + h22;
    let b = h21 - h12;
    let n = a.hypot(b);
    if !(n > 1e-12) {
        return None;
    }
    let c = a / n;
    let s = -b / n;
    let scale = n / var;

    // t = cd - scale * R * cs
    let rcsx = scale * (c * csx - s * csy);
    let rcsy = scale * (s * csx + c * csy);
    let tx = cdx - rcsx;
    let ty = cdy - rcsy;

    Some([
        (scale * c) as f32,
        (-scale * s) as f32,
        tx as f32,
        (scale * s) as f32,
        (scale * c) as f32,
        ty as f32,
    ])
}

/// `M = estimate_norm(landmarks, image_size)`: matrix mapping image coords to
/// template coords (`M . p_image ≈ p_template`, column convention — same as
/// `SimilarityTransform.estimate(src, dst)`).
///
/// Pass it DIRECTLY to `cv2.warpAffine` / [`crate::image::warp_affine_rgba`]
/// to produce a `size x size` aligned crop: like OpenCV (default flags), the
/// warp inverts `M` internally, sampling `src(M^-1 . p)`. Its inverse
/// ([`invert_affine`]) is what the paste-back warp takes — exactly the
/// `M` / `IM = invertAffineTransform(M)` pair from `inswapper.py::get`.
pub fn estimate_norm(landmarks: &[[f32; 2]; 5], image_size: u32) -> Option<Affine2x3> {
    let dst = arcface_dst_for_size(image_size);
    estimate_similarity_transform(landmarks, &dst)
}

/// Invert a 2x3 affine matrix (OpenCV convention). `None` if singular.
pub fn invert_affine(m: &Affine2x3) -> Option<Affine2x3> {
    let (a, b, tx, c, d, ty) = (m[0] as f64, m[1] as f64, m[2] as f64, m[3] as f64, m[4] as f64, m[5] as f64);
    let det = a * d - b * c;
    if det.abs() < 1e-12 {
        return None;
    }
    Some([
        (d / det) as f32,
        (-b / det) as f32,
        ((b * ty - d * tx) / det) as f32,
        (-c / det) as f32,
        (a / det) as f32,
        ((c * tx - a * ty) / det) as f32,
    ])
}

/// Apply an affine matrix to a point.
pub fn apply_affine(m: &Affine2x3, x: f32, y: f32) -> (f32, f32) {
    (
        m[0] * x + m[1] * y + m[2],
        m[3] * x + m[4] * y + m[5],
    )
}

/// Transform a set of landmarks by `m`.
pub fn transform_landmarks(points: &[[f32; 2]], m: &Affine2x3) -> Vec<[f32; 2]> {
    points
        .iter()
        .map(|p| {
            let (x, y) = apply_affine(m, p[0], p[1]);
            [x, y]
        })
        .collect()
}

/// Intersection-over-union of two `xyxy` boxes. Matches the `+1` pixel
/// convention used by InsightFace's `SCRFD.nms`.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let w = (x2 - x1 + 1.0).max(0.0);
    let h = (y2 - y1 + 1.0).max(0.0);
    let inter = w * h;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0] + 1.0).max(0.0) * (a[3] - a[1] + 1.0).max(0.0);
    let area_b = (b[2] - b[0] + 1.0).max(0.0) * (b[3] - b[1] + 1.0).max(0.0);
    inter / (area_a + area_b - inter)
}

/// Greedy NMS over `xyxy` boxes with scores. Returns kept indices ordered by
/// descending score — same algorithm as `SCRFD.nms` in
/// `detection/scrfd/tools/scrfd.py`.
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], iou_thresh: f32) -> Vec<usize> {
    let n = boxes.len().min(scores.len());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        scores[j]
            .partial_cmp(&scores[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    let mut suppressed = vec![false; n];
    for (pos, &i) in order.iter().enumerate() {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &order[pos + 1..] {
            if !suppressed[j] && iou(&boxes[i], &boxes[j]) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn template_112_matches_insightface_constants() {
        let d = arcface_dst_for_size(112);
        for i in 0..5 {
            assert!(approx(d[i][0], ARCFACE_DST_112[i][0], 1e-4));
            assert!(approx(d[i][1], ARCFACE_DST_112[i][1], 1e-4));
        }
    }

    #[test]
    fn template_128_matches_estimate_norm_rule() {
        // ratio = 128/128 = 1, diff_x = 8
        let d = arcface_dst_for_size(128);
        assert!(approx(d[0][0], 38.2946 + 8.0, 1e-3));
        assert!(approx(d[0][1], 51.6963, 1e-3));
        assert!(approx(d[2][0], 56.0252 + 8.0, 1e-3));
    }

    #[test]
    fn similarity_recovery_known_transform() {
        // Known map: scale 2, rotate 30deg, translate (7, -4).
        let ang = 30.0f32.to_radians();
        let (c, s) = (ang.cos(), ang.sin());
        let sc = 2.0f32;
        let m = [sc * c, -sc * s, 7.0, sc * s, sc * c, -4.0];
        let src = ARCFACE_DST_112;
        let dst: [[f32; 2]; 5] = std::array::from_fn(|i| {
            let (x, y) = apply_affine(&m, src[i][0], src[i][1]);
            [x, y]
        });
        let est = estimate_similarity_transform(&src, &dst).unwrap();
        for k in 0..6 {
            assert!(approx(est[k], m[k], 1e-3), "k={k} got {} want {}", est[k], m[k]);
        }
    }

    #[test]
    fn similarity_identity_when_src_eq_dst() {
        let est = estimate_similarity_transform(&ARCFACE_DST_112, &ARCFACE_DST_112).unwrap();
        let ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for k in 0..6 {
            assert!(approx(est[k], ident[k], 1e-4));
        }
    }

    #[test]
    fn similarity_degenerate_returns_none() {
        let dup = [[10.0, 10.0]; 5];
        assert!(estimate_similarity_transform(&dup, &ARCFACE_DST_112).is_none());
    }

    #[test]
    fn inverse_roundtrip() {
        let m = [1.7, 0.4, -12.0, -0.3, 1.9, 33.0];
        let inv = invert_affine(&m).unwrap();
        // M^-1 * M == I on sample points.
        for (x, y) in [(0.0, 0.0), (128.0, 5.0), (40.0, 111.0)] {
            let (a, b) = apply_affine(&m, x, y);
            let (c, d) = apply_affine(&inv, a, b);
            assert!(approx(c, x, 1e-3) && approx(d, y, 1e-3));
        }
    }

    #[test]
    fn inverse_singular_returns_none() {
        assert!(invert_affine(&[1.0, 2.0, 3.0, 2.0, 4.0, 5.0]).is_none());
    }

    #[test]
    fn nms_keeps_highest_and_drops_overlap() {
        let boxes = [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]];
        let scores = [0.9, 0.8, 0.7];
        let keep = nms(&boxes, &scores, 0.4);
        assert_eq!(keep, vec![0, 2]);
    }

    #[test]
    fn iou_disjoint_is_zero() {
        assert_eq!(iou(&[0.0, 0.0, 5.0, 5.0], &[10.0, 10.0, 15.0, 15.0]), 0.0);
    }
}
