//! Where pipeline creation spends its time, summarized per burst.
//!
//! Every pipeline the cache creates records three intervals: waiting for a
//! task-pool thread after it was queued, acquiring its shader modules
//! (precompiled-bundle lookup, or WGSL composition and parsing under the
//! shared shader-cache lock), and the backend's pipeline creation (on Metal,
//! the device compile from AIR to GPU code, which the OS caches after the
//! first run of an install). When the queue drains, one line reports the
//! burst: counts, wall time, thread count and the sum and maximum of each
//! interval.
//!
//! `BEVY_LOG_PIPELINE_TIMING=1` logs bursts of [`LOGGED_BURST`] or more
//! pipelines (warm-up rounds, a new view's first pipelines) at warn, so release
//! builds, which compile out everything below warn, keep them. Otherwise
//! every burst logs at debug.

use bevy_platform::time::Instant;
use core::time::Duration;
use std::sync::Mutex;

/// Bursts this large log at warn under `BEVY_LOG_PIPELINE_TIMING`.
const LOGGED_BURST: u32 = 8;

/// `BEVY_LOG_PIPELINE_TIMING=1`: log pipeline timing at warn.
pub fn pipeline_timing_logged() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("BEVY_LOG_PIPELINE_TIMING").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

#[derive(Default)]
struct Interval {
    sum: Duration,
    max: Duration,
    max_label: String,
}

impl Interval {
    fn add(&mut self, value: Duration, label: Option<&str>) {
        self.sum += value;
        if value > self.max {
            self.max = value;
            self.max_label.clear();
            self.max_label.push_str(label.unwrap_or("<unlabeled>"));
        }
    }

    fn describe(&self, name: &str) -> String {
        format!(
            "{name} sum={:.0}ms max={:.1}ms ({})",
            self.sum.as_secs_f64() * 1000.0,
            self.max.as_secs_f64() * 1000.0,
            self.max_label
        )
    }
}

#[derive(Default)]
struct Burst {
    count: u32,
    precompiled: u32,
    first_queued: Option<Instant>,
    last_done: Option<Instant>,
    queue_wait: Interval,
    modules: Interval,
    backend: Interval,
}

static BURST: Mutex<Option<Burst>> = Mutex::new(None);

/// One pipeline's intervals, measured inside its creation task.
pub(crate) struct PipelineTiming {
    queued: Instant,
    started: Option<Instant>,
    modules_done: Option<Instant>,
}

impl PipelineTiming {
    /// At queueing, on the render thread.
    pub(crate) fn queued() -> Self {
        Self {
            queued: Instant::now(),
            started: None,
            modules_done: None,
        }
    }

    /// First thing in the task.
    pub(crate) fn start(&mut self) {
        self.started = Some(Instant::now());
    }

    /// Once the shader modules are in hand.
    pub(crate) fn modules_done(&mut self) {
        self.modules_done = Some(Instant::now());
    }

    /// After the backend created the pipeline. `precompiled`: every stage
    /// came from the bundle.
    pub(crate) fn finish(self, label: Option<&str>, precompiled: bool) {
        let done = Instant::now();
        let started = self.started.unwrap_or(self.queued);
        let modules_done = self.modules_done.unwrap_or(started);
        let mut guard = BURST.lock().unwrap_or_else(|e| e.into_inner());
        let burst = guard.get_or_insert_with(Burst::default);
        burst.count += 1;
        burst.precompiled += precompiled as u32;
        burst.first_queued = Some(burst.first_queued.map_or(self.queued, |t| t.min(self.queued)));
        burst.last_done = Some(burst.last_done.map_or(done, |t| t.max(done)));
        burst.queue_wait.add(started - self.queued, label);
        burst.modules.add(modules_done - started, label);
        burst.backend.add(done - modules_done, label);
    }
}

/// Called when the pipeline queue has nothing left creating: logs the burst
/// that just finished, if any.
pub(crate) fn flush(threads: impl FnOnce() -> usize) {
    let Some(burst) = BURST.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let threads = threads();
    let wall = match (burst.first_queued, burst.last_done) {
        (Some(first), Some(last)) => last - first,
        _ => Duration::ZERO,
    };
    let line = format!(
        "pipeline compile burst: {} pipelines ({} precompiled, {} runtime) in {:.0}ms wall on {threads} threads; {}; {}; {}",
        burst.count,
        burst.precompiled,
        burst.count - burst.precompiled,
        wall.as_secs_f64() * 1000.0,
        burst.queue_wait.describe("queue wait"),
        burst.modules.describe("modules"),
        burst.backend.describe("backend create"),
    );
    if burst.count >= LOGGED_BURST && pipeline_timing_logged() {
        tracing::warn!("{line}");
    } else {
        tracing::debug!("{line}");
    }
}
