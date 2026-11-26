use core::hash::{BuildHasher, Hash, Hasher};

use bevy_platform::hash::FixedHasher;

use super::{BindGroup, BindGroupEntry, BindGroupLayout, BindingResource};
use crate::renderer::RenderDevice;

/// A bind group that is rebuilt only when what it points at changes.
///
/// A bind group references buffers, texture views and samplers, not their
/// contents, so rewriting a buffer in place leaves it valid. Systems that
/// rebuild one every frame anyway pay a `createBindGroup` per frame, which on
/// the web is a round trip into JavaScript and the browser's GPU process.
///
/// Keyed by a hash of the layout and of each entry's binding, resource
/// identity, offset and size. A wgpu handle's identity cannot be reused while
/// the handle is alive, and the cached bind group keeps every resource it was
/// built from alive, so a new resource never matches a stale key.
///
/// Entries the key cannot describe (binding arrays and the like) are not
/// cached: the bind group is rebuilt each call, as before.
#[derive(Default)]
pub struct CachedBindGroup {
    key: Option<u64>,
    bind_group: Option<BindGroup>,
}

impl CachedBindGroup {
    /// The bind group for `layout` and `entries`, created only when they
    /// differ from the last call's.
    pub fn get_or_create<'a>(
        &mut self,
        render_device: &RenderDevice,
        label: &'a str,
        layout: &'a BindGroupLayout,
        entries: &'a [BindGroupEntry<'a>],
    ) -> &BindGroup {
        let key = bind_group_key(layout, entries);
        let stale = key.is_none() || key != self.key || self.bind_group.is_none();
        if stale {
            self.bind_group = Some(render_device.create_bind_group(label, layout, entries));
            self.key = key;
        }
        self.bind_group.as_ref().unwrap()
    }

    /// The bind group last returned, if any.
    pub fn get(&self) -> Option<&BindGroup> {
        self.bind_group.as_ref()
    }
}

/// A hash identifying the resources `entries` bind under `layout`, or `None`
/// when an entry's resource kind has no identity to hash.
pub fn bind_group_key(layout: &BindGroupLayout, entries: &[BindGroupEntry]) -> Option<u64> {
    let mut hasher = FixedHasher.build_hasher();
    layout.id().hash(&mut hasher);
    for entry in entries {
        entry.binding.hash(&mut hasher);
        match &entry.resource {
            BindingResource::Buffer(binding) => {
                0u8.hash(&mut hasher);
                binding.buffer.hash(&mut hasher);
                binding.offset.hash(&mut hasher);
                binding.size.hash(&mut hasher);
            }
            BindingResource::TextureView(view) => {
                1u8.hash(&mut hasher);
                view.hash(&mut hasher);
            }
            BindingResource::Sampler(sampler) => {
                2u8.hash(&mut hasher);
                sampler.hash(&mut hasher);
            }
            _ => return None,
        }
    }
    Some(hasher.finish())
}
