#include "server_args.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Download a file from a HuggingFace repo via the `hf` CLI. Returns the
// local cache path the CLI prints on stdout, or "" on failure.
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

    while (!path.empty() && (path.back() == '\n' || path.back() == '\r' || path.back() == ' ')) {
        path.pop_back();
    }
    return path;
}

std::string hf_resolve(const std::string & repo_spec, const std::string & file_override,
                       const std::string & default_quant) {
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

void print_usage(const char * program) {
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
    fprintf(stderr, "       --idle-timeout <sec>        unload model after N seconds idle, reload on demand (default: 0 = off)\n");
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
    fprintf(stderr, "  TTS_VOICES_DIR, TTS_IDLE_TIMEOUT\n");
    fprintf(stderr, "  TTS_TEMPERATURE, TTS_TOP_K, TTS_REPETITION_PENALTY, TTS_SEED\n");
}

// truthy/falsy env value: "1", "true", "yes", "on" (case-insensitive)
static bool env_truthy(const char * v) {
    if (!v || !*v) return false;
    std::string s = v;
    for (char & c : s) c = (char)std::tolower((unsigned char)c);
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

bool load_env(server_params & sp) {
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
    if (!get_int  ("TTS_IDLE_TIMEOUT",       sp.idle_timeout_sec))   return false;
    if (!get_float("TTS_TEMPERATURE",        sp.temperature))        return false;
    if (!get_int  ("TTS_TOP_K",              sp.top_k))              return false;
    if (!get_float("TTS_REPETITION_PENALTY", sp.repetition_penalty)) return false;
    if (!get_i64  ("TTS_SEED",               sp.seed))               return false;

    if (const char * v = std::getenv("TTS_VERBOSE")) {
        if (*v) sp.verbose = env_truthy(v);
    }
    return true;
}

bool parse_args(int argc, char ** argv, server_params & sp) {
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
        } else if (arg == "--idle-timeout") {
            if (++i >= argc) { fprintf(stderr, "error: missing idle-timeout\n"); return false; }
            sp.idle_timeout_sec = std::stoi(argv[i]);
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
