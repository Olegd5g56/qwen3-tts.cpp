#pragma once

#include <cstdint>
#include <string>

// Server runtime configuration. Defaults match a local dev setup; env vars
// override defaults; CLI flags override env.
struct server_params {
    std::string model;
    std::string vocoder;
    std::string hf_repo;     // e.g. "khimaros/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF:Q8_0"
    std::string hf_file;     // override filename within --hf-repo
    std::string hf_repo_v;   // vocoder HF repo
    std::string hf_file_v;   // override filename within --hf-repo-v
    std::string voices_dir = "./voices"; // persistent on-disk voice library
    std::string host      = "127.0.0.1";
    int         port      = 8080;
    int         n_threads = 4;
    bool        verbose   = false;
    int         idle_timeout_sec   = 0;  // 0 = disabled (keep model loaded forever)
    float       temperature        = 0.9f;
    int         top_k              = 50;
    float       repetition_penalty = 1.05f;
    int64_t     seed               = -1;
};

// Fill sp from TTS_* env vars. Returns false on invalid numeric values —
// env misconfig in a container should fail loudly rather than silently use
// the default.
bool load_env(server_params & sp);

// Parse CLI flags into sp; CLI overrides env. Returns false on bad input.
// "-h" / "--help" calls exit(0) directly.
bool parse_args(int argc, char ** argv, server_params & sp);

// --help banner to stderr.
void print_usage(const char * program);

// Resolve "user/Repo-GGUF:Q8_0" to a local file path by invoking the `hf`
// CLI. default_quant is used when no ":quant" suffix is present.
std::string hf_resolve(const std::string & repo_spec, const std::string & file_override,
                       const std::string & default_quant = "Q8_0");
