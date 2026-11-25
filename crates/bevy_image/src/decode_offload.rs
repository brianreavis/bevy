//! Optional off-thread decoding for [`ImageLoader`](crate::ImageLoader).
//!
//! A platform whose loaders share a thread with rendering (the
//! single-threaded web build) installs an [`ImageDecodeOffload`] that
//! decodes on another worker. The loader asks it first and decodes inline
//! when it declines, so behavior without one is unchanged.

use std::sync::OnceLock;

use bevy_tasks::BoxedFuture;

use crate::ImageFormat;

/// An image decoded to 8-bit RGBA, the layout [`Image::from_dynamic`]
/// produces for every 8-bit color type.
///
/// [`Image::from_dynamic`]: crate::Image::from_dynamic
pub struct DecodedRgba8 {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes.
    pub data: Vec<u8>,
}

/// Decodes encoded image bytes somewhere other than the calling thread.
pub trait ImageDecodeOffload: Send + Sync + 'static {
    /// Decodes `bytes` as `format`, converted to RGBA8 the way
    /// `DynamicImage::into_rgba8` converts. `None` declines (unsupported
    /// format or color type, decoder unavailable, failure): the loader then
    /// decodes inline.
    fn decode(&self, format: ImageFormat, bytes: Vec<u8>) -> BoxedFuture<'static, Option<DecodedRgba8>>;
}

static OFFLOAD: OnceLock<Box<dyn ImageDecodeOffload>> = OnceLock::new();

/// Installs the platform's image decode offload. The first call wins.
pub fn set_image_decode_offload(offload: Box<dyn ImageDecodeOffload>) {
    let _ = OFFLOAD.set(offload);
}

pub(crate) fn image_decode_offload() -> Option<&'static dyn ImageDecodeOffload> {
    OFFLOAD.get().map(|offload| offload.as_ref())
}
