//! Mask generation, illumination adaptation and seamless blending.
//!
//! This mirrors the *effective* behaviour of InsightFace's
//! `INSwapper.get(..., paste_back=True)` (`model_zoo/inswapper.py`), with one
//! deliberate, documented deviation for performance (see below).
//!
//! Reference (`inswapper.py::get`):
//! 1. `aimg, M = norm_crop2(img, kps, 128)`; run swap -> `bgr_fake` (128).
//! 2. `fake_diff = mean(abs(bgr_fake - aimg))`, zero 2px borders,
//!    `IM = invertAffineTransform(M)`.
//! 3. Warp `bgr_fake`, a white `img_white` mask and `fake_diff` back with
//!    `IM` to full resolution.
//! 4. `img_mask`: white mask thresholded at >20, eroded with
//!    `k = max(mask_size//10, 10)` where
//!    `mask_size = sqrt(mask_h * mask_w)` of the mask extent, then
//!    Gaussian-blurred with `k2 = max(mask_size//20, 5)` (kernel
//!    `2*k2+1`, sigma auto = `0.3*((ksize-1)*0.5-1)+0.8`).
//! 5. `fake_diff` is thresholded at 10, dilated with a 2x2 kernel,
//!    blurred 11x11 — and then (in the current upstream code) *not* used for
//!    the final composite (`#img_mask = fake_diff` is commented out), so the
//!    final blend is `mask*bgr_fake + (1-mask)*target`.
//! 6. `fake_merged` uint8.
//!
//! DEVIATION (performance): upstream erodes/blurs at *full image resolution*,
//! which is O(full-res * k^2) per face — prohibitive for 12–24 MP photos in
//! WASM. We build the eroded+blurred mask in the 128x128 *crop* space and
//! warp the mask back with `IM` (bilinear), which is mathematically the same
//! family of feathered masks at a fraction of the cost. The kernel sizes are
//! computed with the same formulas, scaled to crop space.

/// OpenCV `getGaussianKernel` sigma used when `sigma=0` is passed to
/// `GaussianBlur`: `0.3*((ksize-1)*0.5 - 1) + 0.8`.
pub fn gaussian_sigma_for_kernel(ksize: u32) -> f32 {
    (0.3 * ((ksize as f32 - 1.0) * 0.5 - 1.0) + 0.8).max(0.0)
}

/// 1D Gaussian kernel, odd `ksize`.
fn gaussian_kernel_1d(ksize: u32) -> Vec<f32> {
    gaussian_kernel_1d_sigma(ksize, gaussian_sigma_for_kernel(ksize))
}

/// 1D Gaussian kernel with explicit sigma, odd `ksize`.
fn gaussian_kernel_1d_sigma(ksize: u32, sigma: f32) -> Vec<f32> {
    assert!(ksize % 2 == 1 && ksize >= 1);
    let r = (ksize / 2) as i32;
    // OpenCV with sigma<=0 falls back to the formula above which is > 0 for
    // ksize >= 3; ksize==1 is identity.
    if ksize == 1 || sigma <= 0.0 {
        let mut k = vec![0.0; ksize as usize];
        k[ksize as usize / 2] = 1.0;
        return k;
    }
    let mut k = Vec::with_capacity(ksize as usize);
    let mut sum = 0.0f32;
    for i in -r..=r {
        let v = (-0.5 * (i as f32 / sigma).powi(2)).exp();
        k.push(v);
        sum += v;
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

/// Separable Gaussian blur of a single-channel f32 mask, clamp-to-edge.
/// `ksize` must be odd (mirrors `cv2.GaussianBlur(mask, (k,k), 0)`).
pub fn gaussian_blur_f32(mask: &[f32], w: u32, h: u32, ksize: u32) -> Vec<f32> {
    gaussian_blur_f32_sigma(mask, w, h, ksize, gaussian_sigma_for_kernel(ksize))
}

/// Same as [`gaussian_blur_f32`] with an explicit sigma (mirrors
/// `cv2.GaussianBlur(mask, (k,k), sigma)`).
pub fn gaussian_blur_f32_sigma(mask: &[f32], w: u32, h: u32, ksize: u32, sigma: f32) -> Vec<f32> {
    assert_eq!(mask.len(), (w * h) as usize);
    if ksize <= 1 {
        return mask.to_vec();
    }
    let k = gaussian_kernel_1d_sigma(ksize, sigma);
    let r = (ksize / 2) as i32;
    let (w, h) = (w as i32, h as i32);
    let mut tmp = vec![0.0f32; (w * h) as usize];
    // Horizontal pass.
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x + i as i32 - r).clamp(0, w - 1);
                acc += mask[(y * w + xx) as usize] * kv;
            }
            tmp[(y * w + x) as usize] = acc;
        }
    }
    // Vertical pass.
    let mut out = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, kv) in k.iter().enumerate() {
                let yy = (y + i as i32 - r).clamp(0, h - 1);
                acc += tmp[(yy * w + x) as usize] * kv;
            }
            out[(y * w + x) as usize] = acc;
        }
    }
    out
}

fn sliding_extreme_1d(row: &[f32], k: usize, find_min: bool) -> Vec<f32> {
    let n = row.len();
    let r = k / 2;
    // Exact clamped-border sliding min/max, O(n*k). k is small in crop space
    // (<= ~13); full-res masks never pass through here (they are only warped,
    // see module docs).
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut best = row[i];
        for d in 0..k {
            let j = (i + d).saturating_sub(r).min(n - 1);
            let v = row[j];
            if find_min {
                if v < best {
                    best = v;
                }
            } else if v > best {
                best = v;
            }
        }
        out[i] = best;
    }
    out
}

/// Morphological erosion (min filter) with a square `k`x`k` kernel,
/// `iterations` times. Matches `cv2.erode(mask, ones((k,k)), iterations=1)`
/// closely enough for soft face masks (border handling: clamped, vs OpenCV's
/// constant-border — documented approximation; interior behaviour identical).
pub fn erode_square(mask: &[f32], w: u32, h: u32, k: u32, iterations: u32) -> Vec<f32> {
    assert_eq!(mask.len(), (w * h) as usize);
    if k <= 1 || iterations == 0 {
        return mask.to_vec();
    }
    let (w, h) = (w as usize, h as usize);
    let mut cur = mask.to_vec();
    let mut row = vec![0.0f32; w.max(h)];
    for _ in 0..iterations {
        // Horizontal min.
        let mut tmp = vec![0.0f32; w * h];
        for y in 0..h {
            let base = y * w;
            for x in 0..w {
                row[x] = cur[base + x];
            }
            let f = sliding_extreme_1d(&row[..w], k as usize, true);
            for x in 0..w {
                tmp[base + x] = f[x];
            }
        }
        // Vertical min.
        let mut nxt = vec![0.0f32; w * h];
        let mut col = vec![0.0f32; h];
        let mut col_out = vec![0.0f32; h];
        for x in 0..w {
            for y in 0..h {
                col[y] = tmp[y * w + x];
            }
            col_out.copy_from_slice(&sliding_extreme_1d(&col, k as usize, true));
            for y in 0..h {
                nxt[y * w + x] = col_out[y];
            }
        }
        cur = nxt;
    }
    cur
}

/// Morphological dilation (max filter), same conventions as [`erode_square`].
/// Used for the `fake_diff` gate (`cv2.dilate(fake_diff, ones((2,2)))`).
pub fn dilate_square(mask: &[f32], w: u32, h: u32, k: u32, iterations: u32) -> Vec<f32> {
    assert_eq!(mask.len(), (w * h) as usize);
    if k <= 1 || iterations == 0 {
        return mask.to_vec();
    }
    let (w, h) = (w as usize, h as usize);
    let mut cur = mask.to_vec();
    for _ in 0..iterations {
        let mut tmp = vec![0.0f32; w * h];
        let mut buf = vec![0.0f32; w.max(h)];
        for y in 0..h {
            for x in 0..w {
                buf[x] = cur[y * w + x];
            }
            // Reuse min routine by negation for exactness.
            for v in buf.iter_mut().take(w) {
                *v = -*v;
            }
            let f = sliding_extreme_1d(&buf[..w], k as usize, true);
            for x in 0..w {
                tmp[y * w + x] = -f[x];
            }
        }
        let mut nxt = vec![0.0f32; w * h];
        let mut col = vec![0.0f32; h];
        for x in 0..w {
            for y in 0..h {
                col[y] = -tmp[y * w + x];
            }
            let f = sliding_extreme_1d(&col, k as usize, true);
            for y in 0..h {
                nxt[y * w + x] = -f[y];
            }
        }
        cur = nxt;
    }
    cur
}

/// Mask extent `mask_size = sqrt(mask_h * mask_w)` from `inswapper.py`, where
/// `mask_h/w` span the rows/cols where the (thresholded) mask is 255.
pub fn mask_extent_size(mask_binary_255: &[f32], w: u32, h: u32) -> u32 {
    let (w, h) = (w as usize, h as usize);
    let (mut rmin, mut rmax) = (h, 0usize);
    let (mut cmin, mut cmax) = (w, 0usize);
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if mask_binary_255[y * w + x] >= 128.0 {
                any = true;
                rmin = rmin.min(y);
                rmax = rmax.max(y);
                cmin = cmin.min(x);
                cmax = cmax.max(x);
            }
        }
    }
    if !any {
        return 0;
    }
    let mh = (rmax - rmin + 1) as f32;
    let mw = (cmax - cmin + 1) as f32;
    (mh * mw).sqrt() as u32
}

/// Build the paste-back blending mask in *crop* space (`size` x `size`,
/// typically 128), following `inswapper.py` kernel sizing:
/// `erode_k = max(mask_size//10, 10)`, `blur_k = 2*max(mask_size//20, 5)+1`.
/// Input is the all-white mask with the 2px zero border already applied by
/// the caller (mirrors the `fake_diff` border zeroing convention).
/// Returns f32 weights in [0,1].
pub fn build_crop_mask(white_bordered: &[f32], size: u32) -> Vec<f32> {
    assert_eq!(white_bordered.len(), (size * size) as usize);
    let extent = mask_extent_size(white_bordered, size, size);
    let erode_k = (extent / 10).max(10).max(1);
    let blur_k2 = (extent / 20).max(5);
    let blur_k = blur_k2 * 2 + 1;
    let eroded = erode_square(white_bordered, size, size, erode_k, 1);
    let blurred = gaussian_blur_f32(&eroded, size, size, blur_k);
    blurred.iter().map(|v| (v / 255.0).clamp(0.0, 1.0)).collect()
}

/// Feathered elliptical paste-back mask in *crop* space, Deep-Live-Cam style.
///
/// A filled ellipse (semi-axes `0.44 * size`, matching DLC's
/// `_create_elliptical_mask`) heavily blurred (`31x31`, sigma `12`) with
/// values in [0,1]. Unlike the square [`build_crop_mask`], the corners are
/// zero, so the swapped square's straight edges can never show as a visible
/// box on the face — at the cost of covering slightly less forehead/chin,
/// which the blur feather compensates.
pub fn build_elliptical_mask(size: u32) -> Vec<f32> {
    assert!(size > 0);
    let s = size as usize;
    let c = (size as f32 - 1.0) / 2.0;
    let a = size as f32 * 0.44;
    let mut mask = vec![0.0f32; s * s];
    for y in 0..s {
        for x in 0..s {
            let dx = (x as f32 - c) / a;
            let dy = (y as f32 - c) / a;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * s + x] = 255.0;
            }
        }
    }
    let k = 31.min(size | 1); // odd, capped at the crop size
    let k = if k % 2 == 1 { k } else { k - 1 }.max(1);
    gaussian_blur_f32_sigma(&mask, size, size, k, 12.0)
        .iter()
        .map(|v| (v / 255.0).clamp(0.0, 1.0))
        .collect()
}

/// Separable small-kernel blur of an interleaved RGB u8 buffer,
/// clamp-to-edge. Internal helper for [`soften_rgb`]/[`sharpen_rgb`].
fn blur_rgb(rgb: &[u8], w: usize, h: usize, kernel: &[f32]) -> Vec<f32> {
    let r = kernel.len() / 2;
    let mut tmp = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (i, kv) in kernel.iter().enumerate() {
                    let xx = x.saturating_add(i).saturating_sub(r).min(w - 1);
                    acc += rgb[(y * w + xx) * 3 + c] as f32 * kv;
                }
                tmp[(y * w + x) * 3 + c] = acc;
            }
        }
    }
    let mut out = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (i, kv) in kernel.iter().enumerate() {
                    let yy = y.saturating_add(i).saturating_sub(r).min(h - 1);
                    acc += tmp[(yy * w + x) * 3 + c] * kv;
                }
                out[(y * w + x) * 3 + c] = acc;
            }
        }
    }
    out
}

/// Light denoise for swap-model outputs: 3x3 Gaussian blended back with the
/// original (`amount` in [0,1]). Kills single-pixel model grain before the
/// illumination match (whose gain would otherwise amplify it) without
/// flattening real edges — the follow-up unsharp pass restores crispness.
pub fn soften_rgb(rgb: &[u8], w: u32, h: u32, amount: f32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    assert_eq!(rgb.len(), w * h * 3);
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.0 {
        return rgb.to_vec();
    }
    let kernel = [0.25f32, 0.5, 0.25];
    let blurred = blur_rgb(rgb, w, h, &kernel);
    rgb.iter()
        .enumerate()
        .map(|(i, &v)| {
            let m = v as f32 * (1.0 - amount) + blurred[i] * amount;
            m.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Unsharp mask (Deep-Live-Cam's post-swap `sharpen` step):
/// `out = src * (1 + strength) - blur5(src) * strength`, per channel.
/// Applied lightly on the 128px swapped crop it restores edge crispness lost
/// to the model output + feathering; keep `strength` modest (0.3–0.5) or
/// grain comes back.
pub fn sharpen_rgb(rgb: &[u8], w: u32, h: u32, strength: f32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    assert_eq!(rgb.len(), w * h * 3);
    let strength = strength.clamp(0.0, 2.0);
    if strength <= 0.0 {
        return rgb.to_vec();
    }
    let kernel = gaussian_kernel_1d(5);
    let blurred = blur_rgb(rgb, w, h, &kernel);
    rgb.iter()
        .enumerate()
        .map(|(i, &v)| {
            let m = v as f32 * (1.0 + strength) - blurred[i] * strength;
            m.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// `fake_diff` gate from `inswapper.py`, in crop space:
/// `mean(abs(fake-aimg))`, zero 2px borders, threshold 10 -> {0,255},
/// dilate 2x2 x1, blur 11x11. Returns [0,1] weights. (Upstream leaves this
/// unused in the final composite; we expose it so the JS pipeline and tests
/// can verify parity, and optionally multiply it into the mask — off by
/// default.)
pub fn fake_diff_gate(aimg_rgb: &[u8], fake_rgb: &[u8], size: u32) -> Vec<f32> {
    assert_eq!(aimg_rgb.len(), (size * size * 3) as usize);
    assert_eq!(fake_rgb.len(), (size * size * 3) as usize);
    let s = size as usize;
    let mut diff = vec![0.0f32; s * s];
    for i in 0..s * s {
        let m = ((aimg_rgb[i * 3] as f32 - fake_rgb[i * 3] as f32).abs()
            + (aimg_rgb[i * 3 + 1] as f32 - fake_rgb[i * 3 + 1] as f32).abs()
            + (aimg_rgb[i * 3 + 2] as f32 - fake_rgb[i * 3 + 2] as f32).abs())
            / 3.0;
        diff[i] = m;
    }
    // Zero 2px borders.
    for y in 0..s {
        for x in 0..s {
            if x < 2 || y < 2 || x >= s - 2 || y >= s - 2 {
                diff[y * s + x] = 0.0;
            }
        }
    }
    for v in diff.iter_mut() {
        *v = if *v < 10.0 { 0.0 } else { 255.0 };
    }
    let dilated = dilate_square(&diff, size, size, 2, 1);
    let blurred = gaussian_blur_f32(&dilated, size, size, 11);
    blurred.iter().map(|v| (v / 255.0).clamp(0.0, 1.0)).collect()
}

/// Reinhard-style per-channel illumination adaptation in RGB: match the
/// swapped crop's masked mean/std to the target crop's. `gain` is clamped to
/// [0.5, 2.0] per channel to avoid hue blowups. Operates on RGB triplets.
pub fn color_match_crop(
    target_rgb: &[u8],
    swap_rgb: &[u8],
    mask_01: &[f32],
    npix: usize,
) -> Vec<u8> {
    assert_eq!(target_rgb.len(), npix * 3);
    assert_eq!(swap_rgb.len(), npix * 3);
    assert_eq!(mask_01.len(), npix);
    let mut wsum = 0.0f64;
    let (mut mt, mut ms) = ([0.0f64; 3], [0.0f64; 3]);
    for i in 0..npix {
        let w = mask_01[i] as f64;
        wsum += w;
        for c in 0..3 {
            mt[c] += target_rgb[i * 3 + c] as f64 * w;
            ms[c] += swap_rgb[i * 3 + c] as f64 * w;
        }
    }
    if wsum < 1e-6 {
        return swap_rgb.to_vec();
    }
    for c in 0..3 {
        mt[c] /= wsum;
        ms[c] /= wsum;
    }
    let (mut vt, mut vs) = ([0.0f64; 3], [0.0f64; 3]);
    for i in 0..npix {
        let w = mask_01[i] as f64;
        for c in 0..3 {
            let dt = target_rgb[i * 3 + c] as f64 - mt[c];
            let ds = swap_rgb[i * 3 + c] as f64 - ms[c];
            vt[c] += dt * dt * w;
            vs[c] += ds * ds * w;
        }
    }
    let mut out = vec![0u8; npix * 3];
    for c in 0..3 {
        let st = (vt[c] / wsum).sqrt();
        let ss = (vs[c] / wsum).sqrt();
        let g = if ss < 1e-6 { 1.0 } else { (st / ss).clamp(0.5, 2.0) };
        for i in 0..npix {
            let v = (swap_rgb[i * 3 + c] as f64 - ms[c]) * g + mt[c];
            out[i * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Composite: `out = mask*swap_rgb + (1-mask)*target_rgba` (alpha forced 255).
/// All buffers full-resolution `w`x`h`; `mask` is [0,1] f32 warped back.
pub fn blend_fullres(
    target_rgba: &[u8],
    swap_rgb: &[u8],
    mask_01: &[f32],
    w: u32,
    h: u32,
) -> Vec<u8> {
    let n = (w * h) as usize;
    assert_eq!(target_rgba.len(), n * 4);
    assert_eq!(swap_rgb.len(), n * 3);
    assert_eq!(mask_01.len(), n);
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        let m = mask_01[i].clamp(0.0, 1.0);
        for c in 0..3 {
            let v = m * swap_rgb[i * 3 + c] as f32 + (1.0 - m) * target_rgba[i * 4 + c] as f32;
            out[i * 4 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
        out[i * 4 + 3] = 255;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_formula_matches_opencv() {
        // ksize=11 -> 0.3*((10)*0.5-1)+0.8 = 0.3*4+0.8 = 2.0
        assert!((gaussian_sigma_for_kernel(11) - 2.0).abs() < 1e-6);
        // ksize=3 -> 0.3*(1*0.5... ((3-1)*0.5-1)=0 -> 0.8
        assert!((gaussian_sigma_for_kernel(3) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn blur_preserves_mean_and_range() {
        let w = 16;
        let h = 16;
        let mut m = vec![0.0f32; (w * h) as usize];
        m[(8 * w + 8) as usize] = 255.0;
        let b = gaussian_blur_f32(&m, w, h, 5);
        let sum: f32 = b.iter().sum();
        // Clamp-to-edge loses a little at borders only; impulse is central.
        assert!((sum - 255.0).abs() < 2.0, "sum={sum}");
        assert!(b.iter().all(|&v| v >= 0.0 && v <= 255.0));
        assert!(b[(8 * w + 8) as usize] < 255.0); // spread out
    }

    #[test]
    fn erode_shrinks_white_block() {
        let (w, h) = (16u32, 16u32);
        let mut m = vec![0.0f32; 256];
        for y in 4..12 {
            for x in 4..12 {
                m[y * 16 + x] = 255.0;
            }
        }
        let e = erode_square(&m, w, h, 3, 1);
        // 8x8 block eroded by 3x3 -> 6x6.
        let count = e.iter().filter(|&&v| v > 128.0).count();
        assert_eq!(count, 36);
    }

    #[test]
    fn mask_extent_and_kernel_sizes_follow_inswapper() {
        // Full-white 128 crop with 2px zero border: extent 124 -> sqrt ~124.
        let s = 128usize;
        let mut m = vec![255.0f32; s * s];
        for y in 0..s {
            for x in 0..s {
                if x < 2 || y < 2 || x >= s - 2 || y >= s - 2 {
                    m[y * s + x] = 0.0;
                }
            }
        }
        let extent = mask_extent_size(&m, 128, 128);
        assert!((124 - 4..=124 + 4).contains(&extent), "extent={extent}");
        let mask = build_crop_mask(&m, 128);
        assert_eq!(mask.len(), s * s);
        assert!(mask.iter().all(|&v| (0.0..=1.0).contains(&v)));
        // Center opaque, borders feathered to ~0.
        assert!(mask[64 * s + 64] > 0.99);
        assert!(mask[0] < 0.01);
    }

    #[test]
    fn color_match_moves_means_together() {
        // Target bright, swap dark.
        let n = 64;
        let target = vec![200u8; n * 3];
        let swap = vec![50u8; n * 3];
        let mask = vec![1.0f32; n];
        let out = color_match_crop(&target, &swap, &mask, n);
        let mean: f32 = out.iter().map(|&v| v as f32).sum::<f32>() / (n * 3) as f32;
        assert!((mean - 200.0).abs() < 2.0, "mean={mean}");
    }

    #[test]
    fn color_match_gain_is_clamped() {
        // Target variance huge, swap variance tiny: raw gain would be ~60x.
        let n = 64;
        let mut target = vec![0u8; n * 3];
        let mut swap = vec![0u8; n * 3];
        for i in 0..n {
            let t = if i % 2 == 0 { 0u8 } else { 255u8 };
            let s = if i % 2 == 0 { 126u8 } else { 130u8 };
            for c in 0..3 {
                target[i * 3 + c] = t;
                swap[i * 3 + c] = s;
            }
        }
        let mask = vec![1.0f32; n];
        let out = color_match_crop(&target, &swap, &mask, n);
        // Mean still matches the target...
        let mean: f32 = out.iter().map(|&v| v as f32).sum::<f32>() / (n * 3) as f32;
        assert!((mean - 127.5).abs() < 2.0, "mean={mean}");
        // ...but the range is bounded by the 2.0x clamp, not the ~60x raw gain.
        let (lo, hi) = out.iter().fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!((hi - lo) as f32 <= 4.0 * 2.0 + 2.0, "range={}..{}", lo, hi);
    }

    #[test]
    fn elliptical_mask_covers_center_not_corners() {
        let m = build_elliptical_mask(128);
        assert_eq!(m.len(), 128 * 128);
        assert!(m.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!(m[64 * 128 + 64] > 0.99, "center opaque");
        assert!(m[0] < 0.01 && m[127] < 0.01, "corners transparent");
        // Mostly face, not a full square: opaque area well below the square's.
        let opaque = m.iter().filter(|&&v| v > 0.5).count() as f32 / m.len() as f32;
        assert!(opaque > 0.3 && opaque < 0.75, "opaque fraction={opaque}");
    }

    #[test]
    fn soften_uniform_is_identity_and_kills_impulse() {
        let px = vec![100u8; 32 * 32 * 3];
        assert_eq!(soften_rgb(&px, 32, 32, 0.5), px);
        let mut noisy = px.clone();
        noisy[(16 * 32 + 16) * 3] = 255;
        let out = soften_rgb(&noisy, 32, 32, 0.5);
        assert!(out[(16 * 32 + 16) * 3] < 255, "impulse reduced");
        assert!(out[(16 * 32 + 16) * 3] > 100, "detail preserved");
    }

    #[test]
    fn sharpen_uniform_is_identity_and_boosts_impulse() {
        let px = vec![100u8; 32 * 32 * 3];
        assert_eq!(sharpen_rgb(&px, 32, 32, 0.4), px);
        let mut img = px.clone();
        img[(16 * 32 + 16) * 3] = 200;
        let out = sharpen_rgb(&img, 32, 32, 0.4);
        assert!(out[(16 * 32 + 16) * 3] > 200, "peak enhanced");
        assert!(out.iter().all(|&v| v <= 255));
    }

    #[test]
    fn blend_endpoints() {
        let t = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
        let s = vec![200u8, 210, 220, 100, 110, 120];
        let m = vec![1.0f32, 0.0];
        let o = blend_fullres(&t, &s, &m, 2, 1);
        assert_eq!(&o[0..4], &[200, 210, 220, 255]);
        assert_eq!(&o[4..8], &[40, 50, 60, 255]);
    }

    #[test]
    fn fake_diff_gate_all_same_is_zero() {
        let px = vec![100u8; 128 * 128 * 3];
        let g = fake_diff_gate(&px, &px, 128);
        assert!(g.iter().all(|&v| v < 0.01));
    }
}
