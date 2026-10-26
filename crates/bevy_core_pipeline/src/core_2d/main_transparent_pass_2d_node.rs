use crate::core_2d::Transparent2d;
use std::sync::Arc;
use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_phase::{TrackedRenderPass, ViewSortedRenderPhases},
    render_resource::{CommandEncoder, CommandEncoderDescriptor, RenderPassDescriptor, StoreOp},
    renderer::RenderContext,
    view::{ExtractedView, ViewDepthTexture, ViewTarget},
};
use tracing::error;
#[cfg(feature = "trace")]
use tracing::info_span;

/// Splits a view's main transparent pass at these sort keys (ascending):
/// items sorting before a key render before the break, the rest after. At
/// each break the pass ends (the color target resolved and stored) and
/// [`Transparent2dPassBreakHook`] records into the same encoder, so it can
/// copy what was painted and render elsewhere before the pass resumes.
#[derive(Component, Default, Clone, Debug)]
pub struct Transparent2dPassBreaks(pub Vec<f32>);

/// Runs at each [`Transparent2dPassBreaks`] break: `(world, view entity,
/// break index, encoder)`.
pub type Transparent2dPassBreakFn =
    dyn Fn(&World, Entity, usize, &mut CommandEncoder) + Send + Sync + 'static;

#[derive(Resource, Clone)]
pub struct Transparent2dPassBreakHook(pub Arc<Transparent2dPassBreakFn>);

#[derive(Default)]
pub struct MainTransparentPass2dNode {}

impl ViewNode for MainTransparentPass2dNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ExtractedView,
        &'static ViewTarget,
        &'static ViewDepthTexture,
        Option<&'static Transparent2dPassBreaks>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, view, target, depth, breaks): bevy_ecs::query::QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(transparent_phases) =
            world.get_resource::<ViewSortedRenderPhases<Transparent2d>>()
        else {
            return Ok(());
        };

        let view_entity = graph.view_entity();
        let Some(transparent_phase) = transparent_phases.get(&view.retained_view_entity) else {
            return Ok(());
        };

        let diagnostics = render_context.diagnostic_recorder();

        let color_attachments = [Some(target.get_color_attachment())];
        // NOTE: For the transparent pass we load the depth buffer. There should be no
        // need to write to it, but store is set to `true` as a workaround for issue #3776,
        // https://github.com/bevyengine/bevy/issues/3776
        // so that wgpu does not clear the depth buffer.
        // As the opaque and alpha mask passes run first, opaque meshes can occlude
        // transparent ones.
        let depth_stencil_attachment = Some(depth.get_attachment(StoreOp::Store));

        // Item index of each break, and the attachments the resumed passes
        // use (taken after the first ones, so they load rather than clear).
        let hook = world.get_resource::<Transparent2dPassBreakHook>().cloned();
        let split_at: Vec<usize> = match (breaks, &hook) {
            (Some(breaks), Some(_)) => breaks
                .0
                .iter()
                .map(|key| {
                    transparent_phase
                        .items
                        .partition_point(|item| item.sort_key.0 < *key)
                })
                .collect(),
            _ => Vec::new(),
        };
        let resume_attachments = (!split_at.is_empty()).then(|| {
            (
                [Some(target.get_color_attachment())],
                Some(depth.get_attachment(StoreOp::Store)),
            )
        });

        render_context.add_command_buffer_generation_task(move |render_device| {
            // Command encoder setup
            let mut command_encoder =
                render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("main_transparent_pass_2d_command_encoder"),
                });

            // This needs to run at least once to clear the background color, even if there are no items to render
            {
                #[cfg(feature = "trace")]
                let _main_pass_2d = info_span!("main_transparent_pass_2d").entered();

                let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("main_transparent_pass_2d"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);

                let pass_span = diagnostics.pass_span(&mut render_pass, "main_transparent_pass_2d");

                if let Some(viewport) = camera.viewport.as_ref() {
                    render_pass.set_camera_viewport(viewport);
                }

                let first_end = split_at.first().copied().unwrap_or(transparent_phase.items.len());
                if first_end > 0 {
                    #[cfg(feature = "trace")]
                    let _transparent_main_pass_2d_span =
                        info_span!("transparent_main_pass_2d").entered();
                    if let Err(err) =
                        transparent_phase.render_range(&mut render_pass, world, view_entity, ..first_end)
                    {
                        error!(
                            "Error encountered while rendering the transparent 2D phase {err:?}"
                        );
                    }
                }

                pass_span.end(&mut render_pass);
            }

            // Pass breaks: end, let the hook record, resume with loads.
            if let (Some(hook), Some((color, depth_attachment))) = (&hook, &resume_attachments) {
                for (index, start) in split_at.iter().copied().enumerate() {
                    (hook.0)(world, view_entity, index, &mut command_encoder);
                    let end = split_at
                        .get(index + 1)
                        .copied()
                        .unwrap_or(transparent_phase.items.len())
                        .max(start);
                    let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some("main_transparent_pass_2d_resumed"),
                        color_attachments: color,
                        depth_stencil_attachment: depth_attachment.clone(),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);
                    if let Some(viewport) = camera.viewport.as_ref() {
                        render_pass.set_camera_viewport(viewport);
                    }
                    if end > start
                        && let Err(err) =
                            transparent_phase.render_range(&mut render_pass, world, view_entity, start..end)
                    {
                        error!(
                            "Error encountered while rendering the transparent 2D phase {err:?}"
                        );
                    }
                }
            }

            // WebGL2 quirk: if ending with a render pass with a custom viewport, the viewport isn't
            // reset for the next render pass so add an empty render pass without a custom viewport
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            if camera.viewport.is_some() {
                #[cfg(feature = "trace")]
                let _reset_viewport_pass_2d = info_span!("reset_viewport_pass_2d").entered();
                let pass_descriptor = RenderPassDescriptor {
                    label: Some("reset_viewport_pass_2d"),
                    color_attachments: &[Some(target.get_color_attachment())],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };

                command_encoder.begin_render_pass(&pass_descriptor);
            }

            command_encoder.finish()
        });

        Ok(())
    }
}
