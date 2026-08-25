#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// Opt-in per-op GPU profiler. Enabled by setting QWEN3_TTS_PROFILE_OPS=1.
//
// Hooks ggml_backend_sched's eval callback: ggml synchronises the backend
// before invoking it, so the wall time between two callbacks is the cost of
// the node that just finished. Results are aggregated per (op, shape) under a
// caller-supplied label and dumped by op_profiler_report().
//
// The same hook carries a second, independent probe: QWEN3_TTS_PROBE_NUM=1
// records each node's largest |value| and any non-finite output, and reports
// every node that cannot survive a cast to F16. It exists because the weight
// type decides the activation type in ggml, so an activation above 65504 turns
// into inf under F16 weights and under nothing else - see known-issues.md #16.
// Both flags are independent; either one installs the callback.
//
// Reading the probe: rows naming a *weight* ("...weight (reshaped)") or an
// IM2COL output, capped at exactly 65504.0 with a small non-finite count, are
// an artefact and not a finding. The scheduler reuses compute buffers, so the
// regions a node does not write still hold the previous node's bytes, and the
// scan reads them. Checked on 2026-08-25: the vocoder GGUF's F16 weights hold
// zero inf, zero nan and nothing above 60000. Trust the rows for tensors a node
// actually produces; treat a weight row as noise.
namespace qwen3_tts {

bool op_profiler_enabled();

// Installs the callback on `sched` when profiling is enabled; no-op otherwise.
// `label` must outlive the scheduler (use a string literal).
void op_profiler_attach(ggml_backend_sched_t sched, const char * label);

// Prints the aggregated table for `label` and clears its accumulator.
void op_profiler_report(const char * label);

}  // namespace qwen3_tts
