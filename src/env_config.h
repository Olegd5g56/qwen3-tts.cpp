#pragma once

#include <string>

namespace qwen3_tts {

// Every QWEN3_TTS_* switch the engine reads, in one struct, read once.
//
// Before this existed they were 21 getenv() calls spread over six files, each
// with its own idea of what an unset or empty value meant, and several inside
// hot loops. Finding out what switches existed meant grepping the sources.
//
// Rules for adding one:
//   * add the field HERE, with the variable name in the comment, and parse it
//     in env_config.cpp - nowhere else;
//   * put it in the README table too. A switch nobody can find is a switch
//     nobody can use;
//   * prefer a plain bool. Tri-states (see pipeline, conv_t_f32) exist only
//     where "unset" has to mean something different from both on and off.
//
// The environment is read ONCE, on first use. Changing it afterwards - a
// setenv() from an FFI host, say - has no effect, which is deliberate: two
// halves of one request must not disagree about how it is being rendered.
struct env_config {
    // ---- placement and memory -------------------------------------------
    bool force_cpu      = false;  // QWEN3_TTS_FORCE_CPU=1     - skip every accelerator
    bool low_mem        = false;  // QWEN3_TTS_LOW_MEM         - lazy decoder + component unloads
    bool vocoder_on_cpu = false;  // QWEN3_TTS_VOCODER=cpu     - keep vocoder weights on the CPU

    // ---- pipeline shape --------------------------------------------------
    int  pipeline     = -1;       // QWEN3_TTS_PIPELINE=1/0    - overlap vocoder with generation
                                  //   -1 = decide from the backend
    int  decode_batch = 0;        // QWEN3_TTS_DECODE_BATCH    - frames per vocoder batch
                                  //   0 = each caller's own default
    int  prefix_cache = 8;        // QWEN3_TTS_PREFIX_CACHE    - voices in the prefill KV cache
    bool frame_budget = true;     // QWEN3_TTS_FRAME_BUDGET=0  - disable the per-request frame cap
    bool graph_reuse  = true;     // QWEN3_TTS_GRAPH_REUSE=0   - rebuild every step graph per frame

    // ---- vocoder conv_transpose route (docs/optimization.md) -------------
    bool conv_t_gemm = true;      // QWEN3_TTS_CONV_T_GEMM=0   - back to ggml's own kernel
    int  conv_t_f32  = -1;        // QWEN3_TTS_CONV_T_F32=1/0  - widen conv weights to F32
                                  //   -1 = decide from the backend

    // ---- CoreML code predictor (Apple only) ------------------------------
    bool        coreml_requested = false;  // QWEN3_TTS_USE_COREML set to anything
    bool        coreml_disabled  = false;  //   ... to 0/false/off/no
    std::string coreml_model;              // QWEN3_TTS_COREML_MODEL - .mlpackage path

    // ---- diagnostics -----------------------------------------------------
    bool profile_ops    = false;  // QWEN3_TTS_PROFILE_OPS  - per-(op,shape) timings
    bool probe_num      = false;  // QWEN3_TTS_PROBE_NUM    - activation magnitude probe
    int  probe_top      = 0;      // QWEN3_TTS_PROBE_TOP    - rows to print; 0 = the default 25
    bool keep_runaway   = false;  // QWEN3_TTS_KEEP_RUNAWAY - return runaway audio instead of failing
    bool skip_ref_codes = false;  // QWEN3_TTS_SKIP_REF_CODES - drop ICL reference frames
    bool dump_logits    = false;  // QWEN3_TTS_DUMP_LOGITS  - top-5 cb0 logits for the first frames

    // Dump destinations. Empty means off; a "%d" in the path is replaced with
    // a per-call counter where the writing site says so.
    std::string dump_stages;      // QWEN3_TTS_DUMP_STAGES   - codec encoder stage tensors
    std::string dump_features;    // QWEN3_TTS_DUMP_FEATURES - codec encoder features
    std::string dump_codes;       // QWEN3_TTS_DUMP_CODES    - generated speech codes
    std::string dump_prefill;     // QWEN3_TTS_DUMP_PREFILL  - prefill embeddings
};

// The process-wide configuration. First call reads the environment.
const env_config & env();

} // namespace qwen3_tts
