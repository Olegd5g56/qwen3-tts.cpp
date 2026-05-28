#pragma once

// Tiny stderr logger shared by the server, CLI, and library helpers
// (e.g. voice_store). Output format: HH:MM:SS.mmm LEVEL [req_id] message.
// All emission goes through one mutex so concurrent calls don't interleave
// mid-line.

namespace qwen3_tts {

// Severity tags used by log_at(). Exposed so callers that pick the level
// dynamically (e.g. access logger choosing INFO vs WARN by status) can pass
// the same constants the formatter prints.
extern const char * const LVL_INFO;
extern const char * const LVL_WARN;
extern const char * const LVL_ERROR;
extern const char * const LVL_DEBUG;

// Global verbose gate. When false (default), log_debug() is a no-op so the
// noisy per-stage startup chatter ([mem] snapshots, backend names, vocab
// sizes, lazy-load timings) stays out of the default server log stream.
// Server flips this on via --verbose / TTS_VERBOSE; CLI flips it on
// unconditionally since CLI users expect the detailed timing view.
void set_verbose(bool v);
bool is_verbose();

__attribute__((format(printf, 1, 2)))
void log_info(const char * fmt, ...);

__attribute__((format(printf, 1, 2)))
void log_warn(const char * fmt, ...);

__attribute__((format(printf, 1, 2)))
void log_error(const char * fmt, ...);

__attribute__((format(printf, 1, 2)))
void log_debug(const char * fmt, ...);

__attribute__((format(printf, 2, 3)))
void log_req(const char * req_id, const char * fmt, ...);

__attribute__((format(printf, 2, 3)))
void log_req_warn(const char * req_id, const char * fmt, ...);

__attribute__((format(printf, 3, 4)))
void log_at(const char * level, const char * req_id, const char * fmt, ...);

// Route ggml's internal messages (ggml_vulkan: device discovery, etc.) through
// the same formatter so the log stream is uniform. ggml INFO is downgraded to
// DEBUG so it stays out of the default server log; ERROR/WARN keep their
// severity. Call once at process startup, before the first ggml backend init.
void install_ggml_log_bridge();

}  // namespace qwen3_tts
