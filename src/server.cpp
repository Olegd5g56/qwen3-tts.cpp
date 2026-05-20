// openai-compatible tts server for qwen3-tts.cpp
//
// endpoints:
//   GET  /health              - health check
//   GET  /v1/models           - list loaded model
//   GET  /v1/audio/languages  - list supported languages
//   GET  /v1/audio/voices     - list available voices
//   POST /v1/audio/voices     - create custom voice from reference audio
//   DELETE /v1/audio/voices/X - delete custom voice
//   POST /v1/audio/speech     - synthesize speech (supports voice cloning)

#include "qwen3_tts.h"
#include "voice_store.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opusenc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using namespace qwen3_tts;

// supported languages and their model codec token IDs
static const std::vector<std::pair<std::string, int>> SUPPORTED_LANGUAGES = {
    {"en", 2050}, {"zh", 2055}, {"ja", 2058}, {"ko", 2064}, {"ru", 2069},
    {"de", 2053}, {"fr", 2061}, {"es", 2054}, {"it", 2070}, {"pt", 2071},
};

// language string to model language_id (returns -1 if unknown)
static int language_to_id(const std::string & lang) {
    if (lang.empty()) return 2050;
    for (const auto & [code, id] : SUPPORTED_LANGUAGES) {
        if (lang == code) return id;
    }
    return -1;
}

// encode float32 audio samples as a WAV byte buffer (16-bit PCM)
static std::string encode_wav(const std::vector<float> & samples, int sample_rate) {
    const int num_channels = 1;
    const int bits_per_sample = 16;
    const int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const int block_align = num_channels * bits_per_sample / 8;
    const int data_size = (int)samples.size() * block_align;
    const int file_size = 36 + data_size;

    std::string buf;
    buf.resize(44 + data_size);
    char * p = buf.data();

    auto write_u32 = [](char * dst, uint32_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
        dst[2] = (char)((v >> 16) & 0xff);
        dst[3] = (char)((v >> 24) & 0xff);
    };
    auto write_u16 = [](char * dst, uint16_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
    };

    // RIFF header
    memcpy(p, "RIFF", 4);      write_u32(p + 4, file_size);
    memcpy(p + 8, "WAVE", 4);

    // fmt chunk
    memcpy(p + 12, "fmt ", 4);  write_u32(p + 16, 16);
    write_u16(p + 20, 1);       // PCM
    write_u16(p + 22, num_channels);
    write_u32(p + 24, sample_rate);
    write_u32(p + 28, byte_rate);
    write_u16(p + 32, block_align);
    write_u16(p + 34, bits_per_sample);

    // data chunk
    memcpy(p + 36, "data", 4);  write_u32(p + 40, data_size);

    // convert float32 [-1,1] to int16
    int16_t * dst = reinterpret_cast<int16_t *>(p + 44);
    for (size_t i = 0; i < samples.size(); i++) {
        float s = samples[i];
        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        dst[i] = (int16_t)(s * 32767.0f);
    }

    return buf;
}

// stateful Ogg/Opus encoder. write() consumes float PCM (mono); whenever
// libopusenc finishes a page, on_page is invoked with the encoded bytes
// (raw page data, ready to ship). finish() drains pending samples and
// emits the final pages. Sample rate must be one of 8/12/16/24/48 kHz —
// 24 kHz matches the vocoder output, so no resampling is needed.
struct opus_streamer {
    OggOpusEnc *      enc      = nullptr;
    OggOpusComments * comments = nullptr;
    std::function<bool(const char *, size_t)> on_page;
    bool failed = false;

    bool open(int sample_rate) {
        comments = ope_comments_create();
        OpusEncCallbacks cb;
        cb.write = &opus_streamer::write_cb;
        cb.close = &opus_streamer::close_cb;
        int error = 0;
        enc = ope_encoder_create_callbacks(&cb, this, comments, sample_rate, 1, 0, &error);
        return enc != nullptr && error == OPE_OK;
    }
    bool write(const float * pcm, size_t n_samples) {
        if (!enc || failed) return false;
        return ope_encoder_write_float(enc, pcm, (int)n_samples) == OPE_OK && !failed;
    }
    void finish() {
        if (enc) ope_encoder_drain(enc);
    }
    ~opus_streamer() {
        if (enc) ope_encoder_destroy(enc);
        if (comments) ope_comments_destroy(comments);
    }

    static int write_cb(void * user, const unsigned char * ptr, opus_int32 len) {
        auto * s = static_cast<opus_streamer *>(user);
        if (s->failed) return 1;
        if (!s->on_page(reinterpret_cast<const char *>(ptr), (size_t)len)) {
            s->failed = true;
            return 1;
        }
        return 0;
    }
    static int close_cb(void *) { return 0; }
};

// encode float32 audio samples as a self-contained Ogg/Opus byte buffer
static std::string encode_opus(const std::vector<float> & samples, int sample_rate) {
    std::string out;
    opus_streamer s;
    s.on_page = [&out](const char * p, size_t n) {
        out.append(p, n);
        return true;
    };
    if (!s.open(sample_rate)) return {};
    if (!samples.empty()) s.write(samples.data(), samples.size());
    s.finish();
    return out;
}

// encode float32 audio samples as raw PCM (int16, little-endian)
static std::string encode_pcm(const std::vector<float> & samples) {
    std::string buf;
    buf.resize(samples.size() * sizeof(int16_t));
    int16_t * dst = reinterpret_cast<int16_t *>(buf.data());
    for (size_t i = 0; i < samples.size(); i++) {
        float s = samples[i];
        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        dst[i] = (int16_t)(s * 32767.0f);
    }
    return buf;
}

// emit a 44-byte WAV header with placeholder sizes for streaming.
// clients that tolerate non-finite RIFF/data sizes (ffmpeg, vlc, most players)
// can start playing before the full body arrives.
static std::string wav_streaming_header(int sample_rate) {
    const int num_channels = 1;
    const int bits_per_sample = 16;
    const int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const int block_align = num_channels * bits_per_sample / 8;

    std::string buf;
    buf.resize(44);
    char * p = buf.data();

    auto write_u32 = [](char * dst, uint32_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
        dst[2] = (char)((v >> 16) & 0xff);
        dst[3] = (char)((v >> 24) & 0xff);
    };
    auto write_u16 = [](char * dst, uint16_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
    };

    memcpy(p, "RIFF", 4);       write_u32(p + 4, 0xFFFFFFFF);
    memcpy(p + 8, "WAVE", 4);
    memcpy(p + 12, "fmt ", 4);  write_u32(p + 16, 16);
    write_u16(p + 20, 1);
    write_u16(p + 22, num_channels);
    write_u32(p + 24, sample_rate);
    write_u32(p + 28, byte_rate);
    write_u16(p + 32, block_align);
    write_u16(p + 34, bits_per_sample);
    memcpy(p + 36, "data", 4);  write_u32(p + 40, 0xFFFFFFFF);
    return buf;
}

// minimal RFC 4648 base64 encoder (no line wrapping)
static std::string base64_encode(const char * data, size_t len) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    int val = 0, valb = -6;
    for (size_t i = 0; i < len; i++) {
        val = (val << 8) + (uint8_t)data[i];
        valb += 8;
        while (valb >= 0) {
            out.push_back(tbl[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(tbl[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

// build the speech.audio.done SSE payload. usage mirrors openai (counts user
// content tokens only); timings mirrors llama.cpp's prompt/predicted schema
// for llama-swap compatibility. The tts-specific stages (speaker encoder,
// vocoder, text tokenizer) are surfaced as extra keys — llama-swap ignores
// unknown fields, and our own clients can render the full breakdown.
//
// Semantics:
//   usage.input_tokens   — tokens in the user's `input` text (maps to openai billing).
//   usage.output_tokens  — generated audio codec frames.
//   timings.prompt_n     — real transformer prefill length (text + instruct + ref_text
//                          + ref_codes + framing), i.e. work done before the first
//                          generated audio token.
//   timings.prompt_ms    — build_prefill_graph + forward_prefill wall time.
//   timings.predicted_n  — n_audio_tokens.
//   timings.predicted_ms — transformer autoregressive loop only (excludes vocoder and
//                          prefill), so predicted_per_second reflects pure transformer
//                          throughput comparable to llama-server.
static std::string build_done_event(const tts_result & result) {
    const int32_t input_tokens   = result.n_text_tokens;
    const int32_t output_tokens  = result.n_audio_tokens;
    const int32_t prefill_tokens = result.n_prefill_tokens;
    const int64_t prompt_ms      = result.t_prefill_ms;

    // transformer decode loop only. if get_last_prefill_ms() overshoots
    // t_generate_ms by rounding, clamp to 0 rather than emit a negative.
    int64_t predicted_ms = result.t_generate_ms - result.t_prefill_ms;
    if (predicted_ms < 0) predicted_ms = 0;

    const double pps = prompt_ms    > 0 ? (double)prefill_tokens * 1000.0 / (double)prompt_ms    : 0.0;
    const double tps = predicted_ms > 0 ? (double)output_tokens  * 1000.0 / (double)predicted_ms : 0.0;

    json ev = {
        {"type", "speech.audio.done"},
        {"usage", {
            {"input_tokens",  input_tokens},
            {"output_tokens", output_tokens},
            {"total_tokens",  input_tokens + output_tokens},
        }},
        {"timings", {
            {"prompt_n",             prefill_tokens},
            {"predicted_n",          output_tokens},
            {"prompt_ms",            prompt_ms},
            {"predicted_ms",         predicted_ms},
            {"prompt_per_second",    pps},
            {"predicted_per_second", tps},
            // project extras (llama-swap ignores unknown keys):
            {"tokenize_ms",          result.t_tokenize_ms},
            {"encode_ms",            result.t_encode_ms},
            {"generate_ms",          result.t_generate_ms},
            {"decode_ms",            result.t_decode_ms},
            {"total_ms",             result.t_total_ms},
            {"n_text_tokens",        input_tokens},
        }},
    };
    return ev.dump();
}

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
    float       temperature        = 0.9f;
    int         top_k              = 50;
    float       repetition_penalty = 1.05f;
    int64_t     seed               = -1;
};

// download a file from a huggingface repo, returns local cache path
static std::string hf_download(const std::string & repo, const std::string & filename) {
    std::string cmd = "hf download \"" + repo + "\" \"" + filename + "\" --quiet";
    FILE * fp = popen(cmd.c_str(), "r");
    if (!fp) return "";

    std::string path;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) {
        path = buf;
    }
    int status = pclose(fp);
    if (status != 0) return "";

    // trim trailing whitespace
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r' || path.back() == ' ')) {
        path.pop_back();
    }
    return path;
}

// resolve "user/RepoName-GGUF:QUANT" to a GGUF filename and download it.
// if file_override is set, use that filename instead of deriving one.
// default_quant is used when no :QUANT suffix is present.
static std::string hf_resolve(const std::string & repo_spec, const std::string & file_override,
                               const std::string & default_quant = "Q8_0") {
    std::string repo = repo_spec;
    std::string quant = default_quant;
    auto colon = repo.rfind(':');
    if (colon != std::string::npos) {
        quant = repo.substr(colon + 1);
        repo = repo.substr(0, colon);
    }

    std::vector<std::string> candidates;
    if (!file_override.empty()) {
        candidates.push_back(file_override);
    } else {
        // derive filename: strip "-GGUF" suffix (case-insensitive), try both quant cases
        std::string basename = repo;
        auto slash = basename.rfind('/');
        if (slash != std::string::npos) basename = basename.substr(slash + 1);
        if (basename.size() > 5) {
            std::string tail = basename.substr(basename.size() - 5);
            for (char & c : tail) c = (char)std::toupper((unsigned char)c);
            if (tail == "-GGUF") basename = basename.substr(0, basename.size() - 5);
        }
        candidates.push_back(basename + "-" + quant + ".gguf");
        std::string lquant = quant;
        for (char & c : lquant) c = (char)std::tolower((unsigned char)c);
        if (lquant != quant) candidates.push_back(basename + "-" + lquant + ".gguf");
    }

    for (const auto & gguf_file : candidates) {
        fprintf(stderr, "downloading %s/%s ...\n", repo.c_str(), gguf_file.c_str());
        std::string local_path = hf_download(repo, gguf_file);
        if (!local_path.empty()) return local_path;
    }
    fprintf(stderr, "fatal: failed to download from %s (tried:", repo.c_str());
    for (const auto & c : candidates) fprintf(stderr, " %s", c.c_str());
    fprintf(stderr, ")\n");
    return "";
}

static void print_usage(const char * program) {
    fprintf(stderr, "usage: %s [options] (-m <model.gguf> | -hf <repo:quant>)\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -m,  --model <file>             TTS model GGUF file\n");
    fprintf(stderr, "  -v,  --vocoder <file>           vocoder GGUF file (default: same dir as model)\n");
    fprintf(stderr, "  -hf, --hf-repo <repo[:quant]>   HuggingFace model repo (default quant: Q8_0)\n");
    fprintf(stderr, "       --hf-file <file>            override GGUF filename within --hf-repo\n");
    fprintf(stderr, "       --hf-repo-v <repo[:quant]>  HuggingFace vocoder repo\n");
    fprintf(stderr, "       --hf-file-v <file>          override GGUF filename within --hf-repo-v\n");
    fprintf(stderr, "  -H,  --host <host>              listen host (default: 127.0.0.1)\n");
    fprintf(stderr, "  -p,  --port <port>              listen port (default: 8080)\n");
    fprintf(stderr, "  -j,  --threads <n>              compute threads (default: 4)\n");
    fprintf(stderr, "       --voices-dir <dir>          persistent voice library dir (default: ./voices)\n");
    fprintf(stderr, "  -V,  --verbose                  print per-stage progress and timing\n");
    fprintf(stderr, "       --temperature <f>           sampling temperature default (default: 0.9)\n");
    fprintf(stderr, "       --top-k <n>                 top-k sampling default (default: 50)\n");
    fprintf(stderr, "       --repetition-penalty <f>    repetition penalty default (default: 1.05)\n");
    fprintf(stderr, "       --seed <n>                  default sampling seed (default: -1 = random)\n");
    fprintf(stderr, "  -h,  --help                     show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "environment variables (overridden by CLI flags):\n");
    fprintf(stderr, "  TTS_MODEL, TTS_VOCODER\n");
    fprintf(stderr, "  TTS_HF_REPO, TTS_HF_FILE, TTS_HF_REPO_V, TTS_HF_FILE_V\n");
    fprintf(stderr, "  TTS_HOST, TTS_PORT, TTS_THREADS, TTS_VERBOSE\n");
    fprintf(stderr, "  TTS_VOICES_DIR\n");
    fprintf(stderr, "  TTS_TEMPERATURE, TTS_TOP_K, TTS_REPETITION_PENALTY, TTS_SEED\n");
}

// parse a truthy/falsy env value: "1", "true", "yes", "on" (case-insensitive) => true
static bool env_truthy(const char * v) {
    if (!v || !*v) return false;
    std::string s = v;
    for (char & c : s) c = (char)std::tolower((unsigned char)c);
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

// fill server_params from TTS_* environment variables. called before parse_args
// so CLI flags take precedence over env, which takes precedence over defaults.
// invalid numeric values abort with a clear message rather than silently using
// the default — env misconfig in a container should fail loudly.
static bool load_env(server_params & sp) {
    auto get_str = [](const char * name, std::string & dst) {
        const char * v = std::getenv(name);
        if (v && *v) dst = v;
    };
    auto get_int = [](const char * name, int & dst) -> bool {
        const char * v = std::getenv(name);
        if (!v || !*v) return true;
        try { dst = std::stoi(v); } catch (...) {
            fprintf(stderr, "error: %s must be an integer (got '%s')\n", name, v);
            return false;
        }
        return true;
    };
    auto get_i64 = [](const char * name, int64_t & dst) -> bool {
        const char * v = std::getenv(name);
        if (!v || !*v) return true;
        try { dst = std::stoll(v); } catch (...) {
            fprintf(stderr, "error: %s must be an integer (got '%s')\n", name, v);
            return false;
        }
        return true;
    };
    auto get_float = [](const char * name, float & dst) -> bool {
        const char * v = std::getenv(name);
        if (!v || !*v) return true;
        try { dst = std::stof(v); } catch (...) {
            fprintf(stderr, "error: %s must be a number (got '%s')\n", name, v);
            return false;
        }
        return true;
    };

    get_str("TTS_MODEL",      sp.model);
    get_str("TTS_VOCODER",    sp.vocoder);
    get_str("TTS_HF_REPO",    sp.hf_repo);
    get_str("TTS_HF_FILE",    sp.hf_file);
    get_str("TTS_HF_REPO_V",  sp.hf_repo_v);
    get_str("TTS_HF_FILE_V",  sp.hf_file_v);
    get_str("TTS_VOICES_DIR", sp.voices_dir);
    get_str("TTS_HOST",       sp.host);

    if (!get_int  ("TTS_PORT",               sp.port))               return false;
    if (!get_int  ("TTS_THREADS",            sp.n_threads))          return false;
    if (!get_float("TTS_TEMPERATURE",        sp.temperature))        return false;
    if (!get_int  ("TTS_TOP_K",              sp.top_k))              return false;
    if (!get_float("TTS_REPETITION_PENALTY", sp.repetition_penalty)) return false;
    if (!get_i64  ("TTS_SEED",               sp.seed))               return false;

    if (const char * v = std::getenv("TTS_VERBOSE")) {
        if (*v) sp.verbose = env_truthy(v);
    }
    return true;
}

static bool parse_args(int argc, char ** argv, server_params & sp) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "error: missing model path\n"); return false; }
            sp.model = argv[i];
        } else if (arg == "-v" || arg == "--vocoder") {
            if (++i >= argc) { fprintf(stderr, "error: missing vocoder path\n"); return false; }
            sp.vocoder = argv[i];
        } else if (arg == "-H" || arg == "--host") {
            if (++i >= argc) { fprintf(stderr, "error: missing host\n"); return false; }
            sp.host = argv[i];
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) { fprintf(stderr, "error: missing port\n"); return false; }
            sp.port = std::stoi(argv[i]);
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) { fprintf(stderr, "error: missing threads\n"); return false; }
            sp.n_threads = std::stoi(argv[i]);
        } else if (arg == "--voices-dir") {
            if (++i >= argc) { fprintf(stderr, "error: missing voices-dir\n"); return false; }
            sp.voices_dir = argv[i];
        } else if (arg == "-V" || arg == "--verbose") {
            sp.verbose = true;
        } else if (arg == "-hf" || arg == "--hf-repo") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf repo\n"); return false; }
            sp.hf_repo = argv[i];
        } else if (arg == "--hf-file") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf file\n"); return false; }
            sp.hf_file = argv[i];
        } else if (arg == "--hf-repo-v") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf vocoder repo\n"); return false; }
            sp.hf_repo_v = argv[i];
        } else if (arg == "--hf-file-v") {
            if (++i >= argc) { fprintf(stderr, "error: missing hf vocoder file\n"); return false; }
            sp.hf_file_v = argv[i];
        } else if (arg == "--temperature") {
            if (++i >= argc) { fprintf(stderr, "error: missing temperature\n"); return false; }
            sp.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) { fprintf(stderr, "error: missing top-k\n"); return false; }
            sp.top_k = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) { fprintf(stderr, "error: missing repetition-penalty\n"); return false; }
            sp.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "--seed") {
            if (++i >= argc) { fprintf(stderr, "error: missing seed\n"); return false; }
            sp.seed = std::stoll(argv[i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (sp.model.empty() && sp.hf_repo.empty()) {
        fprintf(stderr, "error: -m <model> or --hf-repo <repo> is required\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    server_params sp;
    if (!load_env(sp)) {
        return 1;
    }
    if (!parse_args(argc, argv, sp)) {
        print_usage(argv[0]);
        return 1;
    }

    // resolve --hf-repo to local file paths
    if (!sp.hf_repo.empty()) {
        sp.model = hf_resolve(sp.hf_repo, sp.hf_file);
        if (sp.model.empty()) return 1;
        fprintf(stderr, "resolved model: %s\n", sp.model.c_str());
    }
    if (!sp.hf_repo_v.empty()) {
        sp.vocoder = hf_resolve(sp.hf_repo_v, sp.hf_file_v, "F16");
        if (sp.vocoder.empty()) return 1;
        fprintf(stderr, "resolved vocoder: %s\n", sp.vocoder.c_str());
    }

    // load models
    Qwen3TTS tts;
    fprintf(stderr, "loading model: %s\n", sp.model.c_str());
    if (!sp.vocoder.empty()) {
        fprintf(stderr, "loading vocoder: %s\n", sp.vocoder.c_str());
    }
    if (!tts.load_model_files(sp.model, sp.vocoder)) {
        fprintf(stderr, "fatal: %s\n", tts.get_error().c_str());
        return 1;
    }
    fprintf(stderr, "models loaded (type=%s, speakers=%zu)\n",
            tts.get_model_type().c_str(), tts.get_speaker_names().size());

    // derive model id from filename (e.g. "qwen3-tts-0.6b-f16" from path)
    std::string model_id = sp.model;
    auto slash = model_id.rfind('/');
    if (slash != std::string::npos) model_id = model_id.substr(slash + 1);
    auto dot = model_id.rfind('.');
    if (dot != std::string::npos) model_id = model_id.substr(0, dot);

    // synthesis is not thread-safe, serialize all requests
    std::mutex synth_mutex;

    // persistent on-disk voice library; manages embeddings and ICL caches.
    VoiceStore voice_store(sp.voices_dir, &tts, &synth_mutex);
    voice_store.refresh();
    {
        auto v = voice_store.list();
        fprintf(stderr, "voice library: %s (%zu voice%s)\n",
                sp.voices_dir.c_str(), v.size(), v.size() == 1 ? "" : "s");
    }

    httplib::Server svr;

    // log all requests
    svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        fprintf(stderr, "%s %s%s%s -> %d\n",
                req.method.c_str(), req.path.c_str(),
                req.params.empty() ? "" : "?",
                req.params.empty() ? "" : [&]() {
                    static thread_local std::string qs;
                    qs.clear();
                    for (auto & [k, v] : req.params) {
                        if (!qs.empty()) qs += '&';
                        qs += k + "=" + v;
                    }
                    return qs.c_str();
                }(),
                res.status);
    });

    // --- GET /health ---
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // --- GET /v1/models ---
    svr.Get("/v1/models", [&model_id](const httplib::Request &, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", json::array({
                {{"id", model_id}, {"object", "model"}, {"owned_by", "qwen"}},
            })},
        };
        res.set_content(models.dump(), "application/json");
    });

    // --- GET /v1/audio/languages ---
    svr.Get("/v1/audio/languages", [](const httplib::Request &, httplib::Response & res) {
        json lang_list = json::array();
        for (const auto & [code, id] : SUPPORTED_LANGUAGES) {
            lang_list.push_back({{"code", code}, {"id", id}});
        }
        res.set_content(json({{"languages", lang_list}}).dump(), "application/json");
    });

    // --- GET /v1/audio/voices ---
    svr.Get("/v1/audio/voices", [&model_id, &tts, &voice_store](const httplib::Request &, httplib::Response & res) {
        json voice_list = json::array({"default"});

        // add built-in speakers from model metadata (custom_voice models)
        for (auto & name : tts.get_speaker_names()) {
            voice_list.push_back(name);
        }

        // add on-disk cloned voices
        for (auto & id : voice_store.list()) {
            voice_list.push_back(id);
        }
        res.set_content(json({{model_id, voice_list}}).dump(), "application/json");
    });

    // --- POST /v1/audio/voices --- create or replace voice from reference audio
    svr.Post("/v1/audio/voices",
        [&tts, &voice_store](const httplib::Request & req, httplib::Response & res) {

        if (!voice_store.can_encode_new()) {
            res.status = 400;
            json err = {{"error", {
                {"message", "this model variant (" + tts.get_model_type() +
                            ") does not support voice cloning from audio; "
                            "use the Base variant, or pick a built-in voice via GET /v1/audio/voices"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // expect multipart form: name (string) + audio_sample (file) [+ ref_text]
        if (!req.has_file("audio_sample")) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'audio_sample' file is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }
        std::string name;
        if (req.has_param("name")) name = req.get_param_value("name");
        if (req.has_file("name")) name = req.get_file_value("name").content;
        if (name.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'name' is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }
        if (!VoiceStore::valid_id(name)) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid 'name': must be [A-Za-z0-9_.-]{1,64}, not '.'-prefixed, not 'default'","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        std::string ref_text;
        if (req.has_param("ref_text")) ref_text = req.get_param_value("ref_text");
        if (req.has_file("ref_text"))  ref_text = req.get_file_value("ref_text").content;

        auto audio_file = req.get_file_value("audio_sample");
        std::string err;
        if (!voice_store.create(name, audio_file.content, ref_text, err)) {
            res.status = 400;
            json e = {{"error", {{"message", "failed to create voice: " + err},
                                  {"type", "invalid_request_error"}}}};
            res.set_content(e.dump(), "application/json");
            return;
        }

        fprintf(stderr, "created voice '%s'%s\n", name.c_str(),
                ref_text.empty() ? "" : " (ICL mode)");
        json resp = {{"id", name}, {"name", name}};
        if (!ref_text.empty()) resp["mode"] = "icl";
        res.set_content(resp.dump(), "application/json");
    });

    // --- DELETE /v1/audio/voices/:id ---
    svr.Delete(R"(/v1/audio/voices/(.+))",
        [&voice_store](const httplib::Request & req, httplib::Response & res) {
        std::string voice_id = req.matches[1];
        std::string err;
        if (voice_store.remove(voice_id, err)) {
            res.set_content(R"({"deleted":true})", "application/json");
        } else {
            res.status = (err == "not found") ? 404 : 400;
            json e = {{"error", {{"message", err},
                                  {"type", res.status == 404 ? "not_found" : "invalid_request_error"}}}};
            res.set_content(e.dump(), "application/json");
        }
    });

    // --- POST /v1/audio/speech ---
    svr.Post("/v1/audio/speech",
        [&tts, &synth_mutex, &sp, &voice_store](const httplib::Request & req, httplib::Response & res) {

        // parse request body
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception &) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid JSON","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        // extract parameters
        std::string input = body.value("input", "");
        if (input.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'input' is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        // openai text limit is 4096 chars
        if (input.size() > 4096) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'input' exceeds 4096 characters","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        std::string response_format = body.value("response_format", "wav");
        std::string stream_format   = body.value("stream_format", "");
        std::string voice           = body.value("voice", "");
        std::string instructions    = body.value("instructions", "");
        std::string language        = body.value("language", "en");
        float       temperature     = body.value("temperature", sp.temperature);
        int         top_k           = body.value("top_k", sp.top_k);
        float       repetition_penalty = body.value("repetition_penalty", sp.repetition_penalty);
        int64_t     seed               = body.value("seed", sp.seed);
        int         stream_batch_size  = body.value("stream_batch_size", 0);
        if (stream_batch_size < 0) stream_batch_size = 0;
        if (stream_batch_size > 256) stream_batch_size = 256;

        fprintf(stderr, "request: voice=%s lang=%s fmt=%s temp=%.2f seed=%lld len=%zu\n",
                voice.empty() ? "default" : voice.c_str(),
                language.c_str(), response_format.c_str(),
                temperature, (long long)seed, input.size());

        // validate language
        int language_id = language_to_id(language);
        if (language_id < 0) {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported language '" + language +
                            "', see GET /v1/audio/languages"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // validate response format
        if (response_format != "wav" && response_format != "pcm" && response_format != "opus") {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported response_format '" + response_format +
                            "', supported: wav, pcm, opus"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // validate stream format (empty = one-shot, openai-spec values = chunked)
        if (!stream_format.empty() && stream_format != "audio" && stream_format != "sse") {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported stream_format '" + stream_format +
                            "', supported: audio, sse"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // resolve voice to speaker embedding (and optional ICL data)
        std::vector<float> voice_embedding;
        std::string voice_ref_text;
        std::vector<int32_t> voice_ref_codes;
        int32_t voice_n_ref_frames = 0;
        if (!voice.empty() && voice != "default") {
            // built-in speaker (CustomVoice models)
            if (tts.get_speaker_id(voice) >= 0) {
                std::lock_guard<std::mutex> lock(synth_mutex);
                if (!tts.get_speaker_embedding(voice, voice_embedding)) {
                    res.status = 500;
                    json err = {{"error", {
                        {"message", "failed to get speaker embedding: " + tts.get_error()},
                        {"type", "server_error"},
                    }}};
                    res.set_content(err.dump(), "application/json");
                    return;
                }
            } else {
                // on-disk cloned voice
                voice_entry ve;
                if (!voice_store.get(voice, ve)) {
                    res.status = 400;
                    json err = {{"error", {
                        {"message", "unknown voice '" + voice + "'"},
                        {"type", "invalid_request_error"},
                    }}};
                    res.set_content(err.dump(), "application/json");
                    return;
                }
                voice_embedding    = std::move(ve.embedding);
                voice_ref_text     = std::move(ve.ref_text);
                voice_ref_codes    = std::move(ve.ref_codes);
                voice_n_ref_frames = ve.n_ref_frames;
            }
        }

        // set up synthesis params
        tts_params params;
        params.n_threads          = sp.n_threads;
        params.temperature        = temperature;
        params.top_k              = top_k;
        params.repetition_penalty = repetition_penalty;
        params.seed               = seed;
        params.language_id        = language_id;
        params.print_progress     = sp.verbose;
        params.print_timing       = sp.verbose;
        params.instructions       = instructions;
        params.ref_text           = voice_ref_text;

        // live streaming path: when stream_format is set and stream_batch_size
        // > 0, synthesis runs INSIDE set_chunked_content_provider so PCM
        // batches flush to the wire as they're produced. stream_batch_size=0
        // preserves the legacy "synthesize-then-chunk" behavior for clients
        // that want a single delta event.
        const bool live_stream = !stream_format.empty() && stream_batch_size > 0;
        if (live_stream) {
            const bool is_sse  = (stream_format == "sse");
            const bool is_wav  = (response_format == "wav");
            const bool is_opus = (response_format == "opus");
            const char * ctype = is_sse ? "text/event-stream"
                                        : (is_wav  ? "audio/wav"
                                        : (is_opus ? "audio/ogg" : "audio/pcm"));

            // capture synthesis inputs; move into provider lambda below.
            res.set_chunked_content_provider(ctype,
                [this_tts = &tts, input = std::move(input), params = std::move(params),
                 voice_embedding = std::move(voice_embedding),
                 voice_ref_codes = std::move(voice_ref_codes),
                 voice_n_ref_frames,
                 stream_batch_size, is_sse, is_wav, is_opus,
                 synth_mutex = &synth_mutex, sample_rate_fallback = 24000]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    std::lock_guard<std::mutex> lock(*synth_mutex);

                    // wav header up front (audio mode only). for SSE, the wav
                    // bytes per-delta are raw pcm — clients reconstruct wav.
                    // opus emits its own ogg headers via libopusenc on first
                    // write, so no manual header is needed.
                    bool header_written = false;
                    auto ensure_header = [&]() {
                        if (!header_written && !is_sse && is_wav) {
                            std::string hdr = wav_streaming_header(sample_rate_fallback);
                            sink.write(hdr.data(), hdr.size());
                        }
                        header_written = true;
                    };

                    // opus encoder is created lazily so we don't pay the cost
                    // on a request that fails before producing any pcm.
                    opus_streamer opus;
                    bool opus_open = false;
                    auto ensure_opus = [&]() -> bool {
                        if (opus_open || !is_opus) return opus_open || !is_opus;
                        opus.on_page = [&](const char * p, size_t n) -> bool {
                            if (is_sse) {
                                json delta = {
                                    {"type", "speech.audio.delta"},
                                    {"audio", base64_encode(p, n)},
                                };
                                std::string frame = "event: speech.audio.delta\ndata: "
                                                  + delta.dump() + "\n\n";
                                return sink.write(frame.data(), frame.size());
                            }
                            return sink.write(p, n);
                        };
                        opus_open = opus.open(sample_rate_fallback);
                        return opus_open;
                    };

                    streaming_opts sopts;
                    sopts.batch_size = stream_batch_size;
                    sopts.on_pcm = [&](const float * pcm, size_t n) -> bool {
                        ensure_header();
                        if (is_opus) {
                            if (!ensure_opus()) return false;
                            return opus.write(pcm, n);
                        }
                        std::string bytes = encode_pcm(std::vector<float>(pcm, pcm + n));
                        if (is_sse) {
                            json delta = {
                                {"type", "speech.audio.delta"},
                                {"audio", base64_encode(bytes.data(), bytes.size())},
                            };
                            std::string frame = "event: speech.audio.delta\ndata: "
                                              + delta.dump() + "\n\n";
                            return sink.write(frame.data(), frame.size());
                        }
                        return sink.write(bytes.data(), bytes.size());
                    };

                    tts_result result;
                    if (!voice_ref_codes.empty()) {
                        result = this_tts->synthesize_with_embedding(
                            input, voice_embedding.data(), (int32_t)voice_embedding.size(),
                            params, voice_ref_codes.data(), voice_n_ref_frames, &sopts);
                    } else if (!voice_embedding.empty()) {
                        result = this_tts->synthesize_with_embedding(
                            input, voice_embedding.data(), (int32_t)voice_embedding.size(),
                            params, nullptr, 0, &sopts);
                    } else {
                        result = this_tts->synthesize(input, params, &sopts);
                    }

                    // ensure a header went out even if no pcm was produced.
                    ensure_header();
                    // flush any opus samples sitting in libopusenc's lookahead.
                    if (opus_open) opus.finish();

                    if (is_sse) {
                        std::string done_frame = "event: speech.audio.done\ndata: "
                                               + build_done_event(result) + "\n\n";
                        sink.write(done_frame.data(), done_frame.size());
                    }
                    sink.done();
                    return false;
                });
            return;
        }

        // synthesize (serialized), using voice embedding if provided
        tts_result result;
        {
            std::lock_guard<std::mutex> lock(synth_mutex);
            if (!voice_ref_codes.empty()) {
                result = tts.synthesize_with_embedding(
                    input, voice_embedding.data(), (int32_t)voice_embedding.size(), params,
                    voice_ref_codes.data(), voice_n_ref_frames);
            } else if (!voice_embedding.empty()) {
                result = tts.synthesize_with_embedding(
                    input, voice_embedding.data(), (int32_t)voice_embedding.size(), params);
            } else {
                result = tts.synthesize(input, params);
            }
        }

        if (!result.success) {
            res.status = 500;
            json err = {{"error", {
                {"message", "synthesis failed: " + result.error_msg},
                {"type", "server_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (result.audio.empty()) {
            res.status = 500;
            res.set_content(R"({"error":{"message":"synthesis produced no audio","type":"server_error"}})",
                            "application/json");
            return;
        }

        fprintf(stderr, "synthesized %.2fs audio (%zu samples) in %lldms\n",
                (float)result.audio.size() / result.sample_rate,
                result.audio.size(), (long long)result.t_total_ms);

        // one-shot (no stream_format): preserve legacy behavior
        if (stream_format.empty()) {
            if (response_format == "pcm") {
                res.set_content(encode_pcm(result.audio), "audio/pcm");
            } else if (response_format == "opus") {
                res.set_content(encode_opus(result.audio, result.sample_rate), "audio/ogg");
            } else {
                res.set_content(encode_wav(result.audio, result.sample_rate), "audio/wav");
            }
            return;
        }

        // stream_format=audio: raw chunked bytes in the chosen response_format.
        // wav uses a placeholder-size header so playback can start immediately.
        // opus produces a self-contained Ogg stream (no separate header).
        if (stream_format == "audio") {
            std::string header;
            std::string body_bytes;
            const char * ctype = "audio/pcm";
            if (response_format == "wav") {
                header     = wav_streaming_header(result.sample_rate);
                body_bytes = encode_pcm(result.audio);
                ctype      = "audio/wav";
            } else if (response_format == "opus") {
                body_bytes = encode_opus(result.audio, result.sample_rate);
                ctype      = "audio/ogg";
            } else {
                body_bytes = encode_pcm(result.audio);
            }

            res.set_chunked_content_provider(ctype,
                [header = std::move(header), body_bytes = std::move(body_bytes)]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    if (!header.empty()) {
                        sink.write(header.data(), header.size());
                    }
                    sink.write(body_bytes.data(), body_bytes.size());
                    sink.done();
                    return false;
                });
            return;
        }

        // stream_format=sse: emit speech.audio.delta + speech.audio.done.
        // response_format still selects the bytes carried inside delta (wav,
        // pcm, or opus). usage/timings on the done event are shaped to be
        // consumed by both openai clients and llama-swap's metrics_monitor.
        {
            std::string audio_bytes;
            if (response_format == "wav") {
                audio_bytes = encode_wav(result.audio, result.sample_rate);
            } else if (response_format == "opus") {
                audio_bytes = encode_opus(result.audio, result.sample_rate);
            } else {
                audio_bytes = encode_pcm(result.audio);
            }

            json delta = {
                {"type", "speech.audio.delta"},
                {"audio", base64_encode(audio_bytes.data(), audio_bytes.size())},
            };
            std::string delta_frame = "event: speech.audio.delta\ndata: " + delta.dump() + "\n\n";
            std::string done_frame  = "event: speech.audio.done\ndata: "
                                    + build_done_event(result)
                                    + "\n\n";

            res.set_chunked_content_provider("text/event-stream",
                [delta_frame = std::move(delta_frame), done_frame = std::move(done_frame)]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    sink.write(delta_frame.data(), delta_frame.size());
                    sink.write(done_frame.data(),  done_frame.size());
                    sink.done();
                    return false;
                });
            return;
        }
    });

    fprintf(stderr, "server listening on %s:%d\n", sp.host.c_str(), sp.port);
    if (!svr.listen(sp.host, sp.port)) {
        fprintf(stderr, "fatal: failed to bind to %s:%d\n", sp.host.c_str(), sp.port);
        return 1;
    }

    return 0;
}
