//! FaceSwapper core: dependency-free geometry, masking and blending for the
//! browser face-swap pipeline.
//!
//! Module map (mirrors the required architecture split):
//! - [`geometry`]: similarity transforms, affine inverse, IoU/NMS.
//! - [`image`]: RGBA warp/resize, SCRFD letterbox, tensor validation.
//! - [`faces`]: face records, confidence filter, top-face selection.
//! - [`pipeline`]: stage list, resolution guards, job validation.
//! - [`blending`]: feathered masks, illumination match, composite.
//! - [`wasm`]: minimal `wasm-bindgen` bridge for the browser.

pub mod blending;
pub mod faces;
pub mod geometry;
pub mod image;
pub mod pipeline;
pub mod wasm;
