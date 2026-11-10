//! offline-precompiled Metal shader support — per-stage keys
//! and the build-time harvest recorder.
//!
//! A pipeline stage's generated MSL depends on exactly: the stage's shader
//! module (asset identity + shader defs), its entry point, the pipeline
//! layout (bind group layouts + immediate size), workgroup-memory zero
//! init, and — for render stages — the vertex buffer layouts and whether
//! the topology is the point class. It does NOT depend on color targets,
//! blend, depth-stencil, or multisample state (see wgpu-hal's Metal
//! `load_shader`). The stage key hashes precisely those inputs, so every
//! pipeline whose stage would compile identical MSL shares a key, and the
//! build-time harvest (`BEVY_SHADER_HARVEST_DIR`) and the runtime
//! lookup compute it identically — same engine code, same descriptors,
//! deterministic by construction.
//!
//! Device-dependent inputs that also shape the MSL but are not per-stage —
//! the global shader defs (`AVAILABLE_STORAGE_BUFFER_BINDINGS`) and the
//! adapter's MSL language version — are recorded in the manifest header by
//! the harvest and validated once at manifest load instead of being folded
//! into every key.
//!
//! Harvest mode (`BEVY_SHADER_HARVEST_DIR=<dir>`): each queued pipeline
//! gets a unique `shader-harvest:<id>` label, and a
//! `stage-keys.jsonl` record `{ variant_tag, stage, stage_key }` per
//! stage. The vendored wgpu-hal Metal backend dumps the corresponding MSL
//! keyed by the same label (`WGPU_HAL_DUMP_MSL_DIR`), and the offline
//! tool joins the two on `(label, stage)` to build the manifest.
//!
//! Harvest mode also writes `pipeline-fixed.jsonl`: per pipeline, the
//! fixed-function state no stage key covers (targets, depth-stencil,
//! multisample, primitive), for audits that need to tell apart pipelines
//! sharing their shaders.
//!
//! Runtime: [`PrecompiledShaders`] can hold several bundles (the engine's,
//! plus any an app registers for its project's generated shaders), each
//! disabled on its own if one of its blobs is rejected.

use alloc::borrow::Cow;
use bevy_asset::Handle;
use bevy_shader::Shader;
use core::fmt::Write as _;
use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::{PrimitiveTopology, ShaderModule};

use crate::renderer::{RenderDevice, WgpuWrapper};

use super::{ComputePipelineDescriptor, RenderPipelineDescriptor};

/// FNV-1a 64 with a caller-supplied offset basis. Chosen over a hasher
/// dependency because the key must be STABLE across builds and platforms —
/// never `DefaultHasher`.
fn fnv1a64(offset_basis: u64, bytes: &[u8]) -> u64 {
    let mut hash = offset_basis;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// 128-bit hex key from two independent FNV-1a 64 passes over the
/// canonical string.
fn key_hash(canonical: &str) -> String {
    format!(
        "{:016x}{:016x}",
        fnv1a64(0xcbf29ce484222325, canonical.as_bytes()),
        fnv1a64(0x9e3779b97f4a7c15, canonical.as_bytes()),
    )
}

/// Stable identity for a stage's shader: the asset path when one exists,
/// otherwise the asset id — which for `load_internal_asset!` shaders is a
/// `uuid_handle!` constant, stable across builds. Runtime-generated
/// handles (neither path nor fixed UUID) would produce unstable keys, but
/// those never match a harvest entry anyway — they just fall back.
fn shader_identity(shader: &Handle<Shader>) -> String {
    match shader.path() {
        Some(path) => path.to_string(),
        None => format!("{:?}", shader.id()),
    }
}

/// Canonical-string tail shared by every stage: defs, layout, immediates,
/// workgroup zero-init. Field order is part of the key format — never
/// reorder without re-harvesting.
fn common_tail(
    s: &mut String,
    shader: &Handle<Shader>,
    shader_defs: &[bevy_shader::ShaderDefVal],
    entry_point: &Option<Cow<'static, str>>,
    layout: &[super::BindGroupLayoutDescriptor],
    immediate_size: u32,
    zero_initialize_workgroup_memory: bool,
) {
    let _ = write!(
        s,
        "shader={};entry={:?};defs={:?};layout={:?};immediate_size={};zero_init_wg={}",
        shader_identity(shader),
        entry_point,
        shader_defs,
        layout,
        immediate_size,
        zero_initialize_workgroup_memory,
    );
}

/// The canonical (pre-hash) key string — exposed separately from the hash
/// so both the harvest records and the runtime miss-debugging can emit it
/// verbatim for diffing when harvest and device disagree.
pub fn render_vertex_stage_canonical(descriptor: &RenderPipelineDescriptor) -> String {
    let mut s = String::from("stage=vertex;");
    common_tail(
        &mut s,
        &descriptor.vertex.shader,
        &descriptor.vertex.shader_defs,
        &descriptor.vertex.entry_point,
        &descriptor.layout,
        descriptor.immediate_size,
        descriptor.zero_initialize_workgroup_memory,
    );
    // Vertex MSL additionally embeds the forced vertex-pulling unpacking
    // (per vertex-buffer stride/step/attributes) and the point-size forcing.
    let _ = write!(
        s,
        ";buffers={:?};point_class={}",
        descriptor.vertex.buffers,
        matches!(descriptor.primitive.topology, PrimitiveTopology::PointList),
    );
    s
}

pub fn render_fragment_stage_canonical(descriptor: &RenderPipelineDescriptor) -> Option<String> {
    let fragment = descriptor.fragment.as_ref()?;
    let mut s = String::from("stage=fragment;");
    common_tail(
        &mut s,
        &fragment.shader,
        &fragment.shader_defs,
        &fragment.entry_point,
        &descriptor.layout,
        descriptor.immediate_size,
        descriptor.zero_initialize_workgroup_memory,
    );
    // `allow_and_force_point_size` is passed for every stage of a
    // point-class pipeline, so it is (conservatively) part of the fragment
    // key too.
    let _ = write!(
        s,
        ";point_class={}",
        matches!(descriptor.primitive.topology, PrimitiveTopology::PointList),
    );
    Some(s)
}

pub fn compute_stage_canonical(descriptor: &ComputePipelineDescriptor) -> String {
    let mut s = String::from("stage=compute;");
    common_tail(
        &mut s,
        &descriptor.shader,
        &descriptor.shader_defs,
        &descriptor.entry_point,
        &descriptor.layout,
        descriptor.immediate_size,
        descriptor.zero_initialize_workgroup_memory,
    );
    s
}

pub fn render_vertex_stage_key(descriptor: &RenderPipelineDescriptor) -> String {
    key_hash(&render_vertex_stage_canonical(descriptor))
}

pub fn render_fragment_stage_key(descriptor: &RenderPipelineDescriptor) -> Option<String> {
    Some(key_hash(&render_fragment_stage_canonical(descriptor)?))
}

pub fn compute_stage_key(descriptor: &ComputePipelineDescriptor) -> String {
    key_hash(&compute_stage_canonical(descriptor))
}


// ─── Harvest recorder ───────────────────────────────────────────────────

fn harvest_dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::var_os("BEVY_SHADER_HARVEST_DIR")?;
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    })
    .as_ref()
}

/// `BEVY_SHADER_HARVEST_TAG_PREFIX`: disambiguates pipeline tags when
/// several harvest passes append to the same record files. Pipeline ids
/// restart per process and creation order is not deterministic across
/// runs, so without a per-pass prefix the join would pair one pass's
/// stage keys with another pass's MSL.
fn harvest_tag(id: bevy_shader::CachedPipelineId) -> String {
    static PREFIX: OnceLock<String> = OnceLock::new();
    let prefix = PREFIX.get_or_init(|| {
        std::env::var("BEVY_SHADER_HARVEST_TAG_PREFIX").unwrap_or_default()
    });
    format!("shader-harvest:{prefix}{id}")
}

fn record(
    dir: &PathBuf,
    variant_tag: &str,
    stage: &str,
    stage_key: &str,
    label: &str,
    canonical: &str,
) {
    let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let escaped_label = escape(label);
    let escaped_canonical = escape(canonical);
    let line = format!(
        "{{\"variant_tag\":\"{variant_tag}\",\"stage\":\"{stage}\",\"stage_key\":\"{stage_key}\",\"label\":\"{escaped_label}\",\"canonical\":\"{escaped_canonical}\"}}\n"
    );
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("stage-keys.jsonl"))
    {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Harvest hook for render pipelines: records each stage's key (plus the
/// pipeline's original, human-readable label — the join tag then REPLACES
/// the label, so this is the only place it survives) and tags the pipeline
/// so the wgpu-hal MSL dump can be joined back. No-op unless
/// `BEVY_SHADER_HARVEST_DIR` is set.
pub(crate) fn maybe_harvest_render(
    id: bevy_shader::CachedPipelineId,
    descriptor: &mut RenderPipelineDescriptor,
) {
    let Some(dir) = harvest_dir() else { return };
    let tag = harvest_tag(id);
    let label = descriptor.label.as_deref().unwrap_or("<unlabeled>").to_owned();
    let vertex_canonical = render_vertex_stage_canonical(descriptor);
    // Fixed-function state, outside every stage key: lets a pipeline audit
    // tell apart pipelines that share their shaders.
    let fixed = format!(
        "targets={:?};depth_stencil={:?};multisample={:?};primitive={:?}",
        descriptor.fragment.as_ref().map(|fragment| &fragment.targets),
        descriptor.depth_stencil,
        descriptor.multisample,
        descriptor.primitive,
    );
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("pipeline-fixed.jsonl"))
    {
        let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let _ = file.write_all(
            format!(
                "{{\"variant_tag\":\"{tag}\",\"label\":\"{}\",\"fixed\":\"{}\"}}\n",
                escape(&label),
                escape(&fixed)
            )
            .as_bytes(),
        );
    }
    record(
        dir,
        &tag,
        "vertex",
        &key_hash(&vertex_canonical),
        &label,
        &vertex_canonical,
    );
    if let Some(fragment_canonical) = render_fragment_stage_canonical(descriptor) {
        record(
            dir,
            &tag,
            "fragment",
            &key_hash(&fragment_canonical),
            &label,
            &fragment_canonical,
        );
    }
    descriptor.label = Some(tag.into());
}

/// Harvest hook for compute pipelines — see [`maybe_harvest_render`].
pub(crate) fn maybe_harvest_compute(
    id: bevy_shader::CachedPipelineId,
    descriptor: &mut ComputePipelineDescriptor,
) {
    let Some(dir) = harvest_dir() else { return };
    let tag = harvest_tag(id);
    let label = descriptor.label.as_deref().unwrap_or("<unlabeled>").to_owned();
    let canonical = compute_stage_canonical(descriptor);
    record(dir, &tag, "compute", &key_hash(&canonical), &label, &canonical);
    descriptor.label = Some(tag.into());
}

// ─── Runtime table ──────────────────────────────────────────────────────

/// One manifest entry: which deduplicated MSL blob a stage key resolves
/// to, the naga-mangled entry name inside it, and the workgroup size
/// (compute only; render stages carry zeros).
#[derive(Clone)]
pub struct PrecompiledStageEntry {
    pub msl_hash: String,
    pub entry: String,
    pub wg_size: [u32; 3],
}

/// A stage served from a precompiled `.metallib`: the passthrough module
/// plus the mangled entry-point name the raw pipeline descriptor must use
/// (passthrough modules look functions up by their literal MSL name, not
/// the WGSL `vertex`/`fragment` name).
pub struct PrecompiledStageHit {
    pub module: Arc<WgpuWrapper<ShaderModule>>,
    pub entry_point: String,
}

/// One installed bundle: the directory its `<msl_hash>.metallib` blobs live
/// in, and its own state. Shared (behind an `Arc`) by every table that
/// includes it, so a disable or a hit count survives the table being
/// extended with another bundle.
struct PrecompiledBundle {
    dir: PathBuf,
    /// Set when one of this bundle's metallibs is rejected at load (e.g.
    /// built for an MSL language version this OS doesn't support): one bad
    /// blob means the bundle doesn't fit this device, so all of its stages
    /// fall back to runtime compilation. Other bundles keep serving.
    disabled: AtomicBool,
    hits: AtomicU32,
}

/// The precompiled-shader table, installed on the [`PipelineCache`] by the
/// embedder (the C API loads + validates each `shaders-manifest.json` and
/// calls `PipelineCache::set_precompiled_shaders`). It can hold several
/// bundles: the engine's own, plus any an app registers for its project's
/// generated shaders. Metallib blobs load lazily (`newLibraryWithData`) and
/// are cached per content hash across bundles, so stages sharing MSL share
/// one `MTLLibrary`.
///
/// Requires `wgpu::Features::PASSTHROUGH_SHADERS` on the device.
///
/// [`PipelineCache`]: super::PipelineCache
pub struct PrecompiledShaders {
    bundles: Vec<Arc<PrecompiledBundle>>,
    /// Stage key → (index into `bundles`, entry).
    stages: HashMap<String, (usize, PrecompiledStageEntry)>,
    modules: Arc<Mutex<HashMap<String, Arc<WgpuWrapper<ShaderModule>>>>>,
    misses: Arc<AtomicU32>,
}

impl PrecompiledShaders {
    /// A table holding one bundle. `dir` holds `<msl_hash>.metallib` files;
    /// `stages` maps stage keys to manifest entries (parsed and validated by
    /// the embedder).
    pub fn new(dir: PathBuf, stages: HashMap<String, PrecompiledStageEntry>) -> Self {
        Self {
            bundles: Vec::new(),
            stages: HashMap::new(),
            modules: Arc::new(Mutex::new(HashMap::new())),
            misses: Arc::new(AtomicU32::new(0)),
        }
        .with_bundle(dir, stages)
    }

    /// This table plus one more bundle. A stage key already in the table
    /// keeps its entry: within one shader fingerprint the same key is the
    /// same MSL (each harvest fails on a collision), so which bundle serves
    /// it doesn't matter, and the earlier one (the engine's) keeps serving.
    /// Swapping the result into the [`PipelineCache`] is safe at any time:
    /// each pipeline creation holds the table it was queued with.
    ///
    /// [`PipelineCache`]: super::PipelineCache
    pub fn with_bundle(&self, dir: PathBuf, stages: HashMap<String, PrecompiledStageEntry>) -> Self {
        let index = self.bundles.len();
        let mut bundles = self.bundles.clone();
        bundles.push(Arc::new(PrecompiledBundle {
            dir,
            disabled: AtomicBool::new(false),
            hits: AtomicU32::new(0),
        }));
        let mut merged = self.stages.clone();
        for (key, entry) in stages {
            merged.entry(key).or_insert((index, entry));
        }
        Self {
            bundles,
            stages: merged,
            modules: self.modules.clone(),
            misses: self.misses.clone(),
        }
    }

    /// Stage keys the table serves, across all bundles.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// `true` when the table serves no stage keys.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// `(hits, misses)` across all stage lookups so far — the "precompiled
    /// shaders: X/Y" diagnostic.
    pub fn stats(&self) -> (u32, u32) {
        (
            self.bundles
                .iter()
                .map(|bundle| bundle.hits.load(Ordering::Relaxed))
                .sum(),
            self.misses.load(Ordering::Relaxed),
        )
    }

    /// Per bundle, in install order: its directory, the stages it served, and
    /// whether it was disabled by a rejected blob.
    pub fn bundle_stats(&self) -> Vec<(PathBuf, u32, bool)> {
        self.bundles
            .iter()
            .map(|bundle| {
                (
                    bundle.dir.clone(),
                    bundle.hits.load(Ordering::Relaxed),
                    bundle.disabled.load(Ordering::Relaxed),
                )
            })
            .collect()
    }

    pub fn lookup_render_vertex(
        &self,
        device: &RenderDevice,
        descriptor: &RenderPipelineDescriptor,
    ) -> Option<PrecompiledStageHit> {
        let canonical = render_vertex_stage_canonical(descriptor);
        self.lookup_with_miss_debug(device, &canonical, descriptor.label.as_deref(), "vertex")
    }

    pub fn lookup_render_fragment(
        &self,
        device: &RenderDevice,
        descriptor: &RenderPipelineDescriptor,
    ) -> Option<PrecompiledStageHit> {
        let canonical = render_fragment_stage_canonical(descriptor)?;
        self.lookup_with_miss_debug(device, &canonical, descriptor.label.as_deref(), "fragment")
    }

    pub fn lookup_compute(
        &self,
        device: &RenderDevice,
        descriptor: &ComputePipelineDescriptor,
    ) -> Option<PrecompiledStageHit> {
        let canonical = compute_stage_canonical(descriptor);
        self.lookup_with_miss_debug(device, &canonical, descriptor.label.as_deref(), "compute")
    }

    /// Hash the canonical, look it up, and log key-miss details with the
    /// full canonical string, diffable against the harvest's
    /// `stage-keys.jsonl` (which records the same canonical per stage).
    fn lookup_with_miss_debug(
        &self,
        device: &RenderDevice,
        canonical: &str,
        label: Option<&str>,
        stage: &str,
    ) -> Option<PrecompiledStageHit> {
        let key = key_hash(canonical);
        let hit = self.lookup(device, &key);
        if hit.is_none() && !self.stages.contains_key(&key) {
            tracing::warn!(
                "precompiled miss [{stage}] {}: key={key} canonical={canonical}",
                label.unwrap_or("<unlabeled>"),
            );
        }
        hit
    }

    fn lookup(&self, device: &RenderDevice, stage_key: &str) -> Option<PrecompiledStageHit> {
        let Some((bundle_index, entry)) = self.stages.get(stage_key) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let bundle = &self.bundles[*bundle_index];
        if bundle.disabled.load(Ordering::Relaxed) {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let module = {
            let mut modules = self.modules.lock().unwrap();
            match modules.get(&entry.msl_hash) {
                Some(module) => module.clone(),
                None => {
                    let path = bundle.dir.join(format!("{}.metallib", entry.msl_hash));
                    let bytes = match std::fs::read(&path) {
                        Ok(bytes) => bytes,
                        Err(err) => {
                            tracing::warn!(
                                "precompiled shader blob missing ({}): {err} — falling back to runtime compile",
                                path.display()
                            );
                            self.misses.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };
                    // A rejected metallib (wrong MSL language version or
                    // deployment target for this OS, corrupt data, …)
                    // surfaces as a wgpu error, which the default handler
                    // treats as FATAL — catch it and degrade to runtime
                    // compilation instead. The metallib-loader failure is
                    // classified `Internal` (`CreateShaderModuleError::
                    // Generation`), while entry-point/backend problems are
                    // `Validation`, so BOTH scopes are needed.
                    let validation_scope = device
                        .wgpu_device()
                        .push_error_scope(wgpu::ErrorFilter::Validation);
                    let internal_scope = device
                        .wgpu_device()
                        .push_error_scope(wgpu::ErrorFilter::Internal);
                    // SAFETY: the metallib was compiled offline from the
                    // byte-exact MSL wgpu-hal would generate for this stage
                    // key (same naga, same binding maps — see the module
                    // docs), so the binding ABI matches by construction.
                    let module = unsafe {
                        device.wgpu_device().create_shader_module_passthrough(
                            wgpu::ShaderModuleDescriptorPassthrough {
                                label: Some("precompiled-metallib"),
                                entry_points: Cow::Owned(vec![
                                    wgpu::PassthroughShaderEntryPoint {
                                        name: Cow::Owned(entry.entry.clone()),
                                        workgroup_size: (
                                            entry.wg_size[0],
                                            entry.wg_size[1],
                                            entry.wg_size[2],
                                        ),
                                    },
                                ]),
                                metallib: Some(Cow::Owned(bytes)),
                                ..Default::default()
                            },
                        )
                    };
                    // Pop in reverse push order (scopes are a stack).
                    let internal_error =
                        bevy_tasks::futures::now_or_never(internal_scope.pop()).flatten();
                    let validation_error =
                        bevy_tasks::futures::now_or_never(validation_scope.pop()).flatten();
                    if let Some(error) = internal_error.or(validation_error) {
                        tracing::warn!(
                            "precompiled shader library {} rejected ({error}) — disabling the precompiled bundle {}, falling back to runtime compilation",
                            path.display(),
                            bundle.dir.display()
                        );
                        bundle.disabled.store(true, Ordering::Relaxed);
                        self.misses.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                    let module = Arc::new(WgpuWrapper::new(module));
                    modules.insert(entry.msl_hash.clone(), module.clone());
                    module
                }
            }
        };
        bundle.hits.fetch_add(1, Ordering::Relaxed);
        Some(PrecompiledStageHit {
            module,
            entry_point: entry.entry.clone(),
        })
    }
}

#[cfg(test)]
mod bundle_tests {
    use super::*;

    fn entry(msl_hash: &str) -> PrecompiledStageEntry {
        PrecompiledStageEntry {
            msl_hash: msl_hash.to_string(),
            entry: "main".to_string(),
            wg_size: [0, 0, 0],
        }
    }

    #[test]
    fn an_added_bundle_serves_only_keys_the_table_lacks() {
        let engine = PrecompiledShaders::new(
            PathBuf::from("engine"),
            HashMap::from([("a".to_string(), entry("engine-a"))]),
        );
        let merged = engine.with_bundle(
            PathBuf::from("app"),
            HashMap::from([
                ("a".to_string(), entry("app-a")),
                ("b".to_string(), entry("app-b")),
            ]),
        );
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.stages["a"].0, 0);
        assert_eq!(merged.stages["a"].1.msl_hash, "engine-a");
        assert_eq!(merged.stages["b"].0, 1);
        // The original table is untouched.
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn extending_a_table_shares_its_bundles_and_library_cache() {
        let engine = PrecompiledShaders::new(PathBuf::from("engine"), HashMap::new());
        let merged = engine.with_bundle(PathBuf::from("app"), HashMap::new());
        assert!(Arc::ptr_eq(&engine.bundles[0], &merged.bundles[0]));
        assert!(Arc::ptr_eq(&engine.modules, &merged.modules));
        engine.bundles[0].disabled.store(true, Ordering::Relaxed);
        let stats = merged.bundle_stats();
        assert!(stats[0].2, "a disable is seen through every table holding the bundle");
        assert!(!stats[1].2, "and only for that bundle");
    }
}
