#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// Opt-in per-op GPU profiler. Enabled by setting QWEN3_TTS_PROFILE_OPS=1.
//
// Hooks ggml_backend_sched's eval callback: ggml synchronises the backend
// before invoking it, so the wall time between two callbacks is the cost of
// the node that just finished. Results are aggregated per (op, shape) under a
// caller-supplied label and dumped by op_profiler_report().
namespace qwen3_tts {

bool op_profiler_enabled();

// Installs the callback on `sched` when profiling is enabled; no-op otherwise.
// `label` must outlive the scheduler (use a string literal).
void op_profiler_attach(ggml_backend_sched_t sched, const char * label);

// Prints the aggregated table for `label` and clears its accumulator.
void op_profiler_report(const char * label);

}  // namespace qwen3_tts
