#include "env_config.h"

#include <algorithm>
#include <cstdlib>

namespace qwen3_tts {

namespace {

// A switch is "on" when it is set to something that is not 0. An empty value
// counts as unset - `FOO= command` should not turn anything on.
bool flag_on(const char * v) {
    return v && v[0] && v[0] != '0';
}

// Tri-state: -1 unset, 0 off, 1 on. Used where "unset" is a third answer
// (decide from the backend) and not just the default.
int tri(const char * v) {
    if (!v || !v[0]) return -1;
    return v[0] != '0' ? 1 : 0;
}

std::string str(const char * v) {
    return v && v[0] ? std::string(v) : std::string();
}

env_config read_env() {
    env_config c;
    const char * v = nullptr;

    // FORCE_CPU has always been exactly "1", not any truthy value. Kept, so a
    // script that sets it to 0 to mean "no" keeps working.
    v = std::getenv("QWEN3_TTS_FORCE_CPU");
    c.force_cpu = v && v[0] == '1';

    c.low_mem = flag_on(std::getenv("QWEN3_TTS_LOW_MEM"));

    // QWEN3_TTS_VOCODER names where the vocoder runs; only "cpu" means
    // anything, and only its first letter is looked at.
    v = std::getenv("QWEN3_TTS_VOCODER");
    c.vocoder_on_cpu = v && (v[0] == 'c' || v[0] == 'C');

    c.pipeline = tri(std::getenv("QWEN3_TTS_PIPELINE"));

    v = std::getenv("QWEN3_TTS_DECODE_BATCH");
    c.decode_batch = v ? std::atoi(v) : 0;
    if (c.decode_batch < 0) c.decode_batch = 0;

    v = std::getenv("QWEN3_TTS_PREFIX_CACHE");
    if (v) {
        c.prefix_cache = (int) std::strtol(v, nullptr, 10);
        if (c.prefix_cache < 0) c.prefix_cache = 0;
    }

    // The frame budget is on unless explicitly switched off.
    v = std::getenv("QWEN3_TTS_FRAME_BUDGET");
    c.frame_budget = !(v && v[0] == '0');

    // The GEMM route is the default; the variable exists to get back to
    // ggml's own kernel for comparison.
    v = std::getenv("QWEN3_TTS_CONV_T_GEMM");
    c.conv_t_gemm = !(v && v[0] == '0');
    c.conv_t_f32  = tri(std::getenv("QWEN3_TTS_CONV_T_F32"));

    v = std::getenv("QWEN3_TTS_USE_COREML");
    c.coreml_requested = v && v[0];
    if (c.coreml_requested) {
        std::string s = v;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char ch) { return (char) std::tolower(ch); });
        c.coreml_disabled = (s == "0" || s == "false" || s == "off" || s == "no");
    }
    c.coreml_model = str(std::getenv("QWEN3_TTS_COREML_MODEL"));

    c.profile_ops = flag_on(std::getenv("QWEN3_TTS_PROFILE_OPS"));
    c.probe_num   = flag_on(std::getenv("QWEN3_TTS_PROBE_NUM"));

    v = std::getenv("QWEN3_TTS_PROBE_TOP");
    if (v) {
        const long n = std::strtol(v, nullptr, 10);
        if (n > 0) c.probe_top = (int) n;
    }

    c.keep_runaway = flag_on(std::getenv("QWEN3_TTS_KEEP_RUNAWAY"));

    // These two are presence-only: any value at all, including 0, enables
    // them. That is what the code did before, and both are debugging aids
    // where "set it and see" is the whole point.
    c.skip_ref_codes = std::getenv("QWEN3_TTS_SKIP_REF_CODES") != nullptr;
    c.dump_logits    = std::getenv("QWEN3_TTS_DUMP_LOGITS")    != nullptr;

    c.dump_stages   = str(std::getenv("QWEN3_TTS_DUMP_STAGES"));
    c.dump_features = str(std::getenv("QWEN3_TTS_DUMP_FEATURES"));
    c.dump_codes    = str(std::getenv("QWEN3_TTS_DUMP_CODES"));
    c.dump_prefill  = str(std::getenv("QWEN3_TTS_DUMP_PREFILL"));

    return c;
}

} // namespace

const env_config & env() {
    // Function-local static: read on first use, initialised exactly once even
    // if two threads arrive together.
    static const env_config cfg = read_env();
    return cfg;
}

} // namespace qwen3_tts
