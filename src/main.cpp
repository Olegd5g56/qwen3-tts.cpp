#include "qwen3_tts.h"
#include "voice_store.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>

// Strict numeric parsers: a typo in a flag or env var must print an error,
// not abort with an uncaught std::invalid_argument.
static bool parse_int_val(const char * what, const char * s, int32_t & out) {
    try {
        size_t pos = 0;
        const int v = std::stoi(s, &pos);
        if (pos != std::string(s).size()) throw std::invalid_argument(s);
        out = v;
        return true;
    } catch (...) {
        fprintf(stderr, "Error: %s expects an integer (got '%s')\n", what, s);
        return false;
    }
}
static bool parse_i64_val(const char * what, const char * s, int64_t & out) {
    try {
        size_t pos = 0;
        const long long v = std::stoll(s, &pos);
        if (pos != std::string(s).size()) throw std::invalid_argument(s);
        out = v;
        return true;
    } catch (...) {
        fprintf(stderr, "Error: %s expects an integer (got '%s')\n", what, s);
        return false;
    }
}
static bool parse_float_val(const char * what, const char * s, float & out) {
    try {
        size_t pos = 0;
        const float v = std::stof(s, &pos);
        if (pos != std::string(s).size()) throw std::invalid_argument(s);
        out = v;
        return true;
    } catch (...) {
        fprintf(stderr, "Error: %s expects a number (got '%s')\n", what, s);
        return false;
    }
}

static bool parse_language(const std::string & lang, int32_t & out_id) {
    if      (lang == "en" || lang == "english")    out_id = 2050;
    else if (lang == "ru" || lang == "russian")    out_id = 2069;
    else if (lang == "zh" || lang == "chinese")    out_id = 2055;
    else if (lang == "ja" || lang == "japanese")   out_id = 2058;
    else if (lang == "ko" || lang == "korean")     out_id = 2064;
    else if (lang == "de" || lang == "german")     out_id = 2053;
    else if (lang == "fr" || lang == "french")     out_id = 2061;
    else if (lang == "es" || lang == "spanish")    out_id = 2054;
    else if (lang == "it" || lang == "italian")    out_id = 2070;
    else if (lang == "pt" || lang == "portuguese") out_id = 2071;
    else return false;
    return true;
}

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>     Talker gguf file, or directory containing talker+tokenizer ggufs (required)\n");
    fprintf(stderr, "      --vocoder <file>   Tokenizer/vocoder gguf (required when -m is a file; else auto-discovered)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (or piped via stdin)\n");
    fprintf(stderr, "  -o, --output <file>    Output file (default: output.wav). Format picked by extension:\n");
    fprintf(stderr, "                           .wav (PCM s16), .mp3 (LAME VBR -V 4), .opus / .ogg\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "      --ref-text <text>  Transcript of the reference audio (enables ICL cloning)\n");
    fprintf(stderr, "  -v, --voice <name>     Built-in speaker or a voice from --voices-dir\n");
    fprintf(stderr, "      --voices-dir <d>   Voice library directory (same layout as the server)\n");
    fprintf(stderr, "      --list-voices      List available voices (built-in + library) and exit\n");
    fprintf(stderr, "  --streaming-batch-size <n> Decode in n-frame batches while generating (default: 0 = off)\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  --codebooks <n>        RVQ codebooks to use, 1-16 (default: 16 = all).\n");
    fprintf(stderr, "                         Fewer is faster and coarser; 8 roughly halves\n");
    fprintf(stderr, "                         the code-predictor cost per frame\n");
    fprintf(stderr, "  --seed <n>             Sampling seed (default: -1 = random)\n");
    fprintf(stderr, "  -i, --instructions <s> Voice steering instructions\n");
    fprintf(stderr, "  -l, --language <lang>  Language: en,ru,zh,ja,ko,de,fr,es (default: en)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Environment variables (overridden by CLI flags):\n");
    fprintf(stderr, "  TTS_MODEL, TTS_VOCODER, TTS_THREADS\n");
    fprintf(stderr, "  TTS_VOICE, TTS_VOICES_DIR, TTS_LANGUAGE\n");
    fprintf(stderr, "  TTS_TEMPERATURE, TTS_TOP_K, TTS_REPETITION_PENALTY, TTS_SEED\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.mp3\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.mp3 -o cloned.opus\n", program);
    fprintf(stderr, "  echo \"Hello!\" | %s -m ./models -o hello.wav\n", program);
}

int main(int argc, char ** argv) {
    // CLI is for interactive use — show the full debug stream by default so
    // [mem] / backend / per-component timing lines stay visible. The server
    // gates this behind --verbose because admins want a quiet startup log.
    qwen3_tts::set_verbose(true);
    // Funnel ggml backend chatter through our formatter so it sits in the
    // same timestamped stream as everything else.
    qwen3_tts::install_ggml_log_bridge();

    std::string model_path;
    std::string vocoder_path;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string voice_name;
    std::string voices_dir;
    bool        list_voices = false;
    int32_t streaming_batch_size = 0;

    qwen3_tts::tts_params params;

    // Defaults from environment. CLI flags below take precedence.
    auto env_str = [](const char * name, std::string & dst) {
        if (const char * v = std::getenv(name)) dst = v;
    };
    env_str("TTS_MODEL",      model_path);
    env_str("TTS_VOCODER",    vocoder_path);
    env_str("TTS_VOICE",      voice_name);
    env_str("TTS_VOICES_DIR", voices_dir);
    if (const char * v = std::getenv("TTS_LANGUAGE")) {
        if (!parse_language(v, params.language_id)) {
            fprintf(stderr, "Error: TTS_LANGUAGE='%s' is not a supported language\n", v);
            return 1;
        }
    }
    if (const char * v = std::getenv("TTS_THREADS"))
        if (!parse_int_val("TTS_THREADS", v, params.n_threads)) return 1;
    if (const char * v = std::getenv("TTS_TEMPERATURE"))
        if (!parse_float_val("TTS_TEMPERATURE", v, params.temperature)) return 1;
    if (const char * v = std::getenv("TTS_TOP_K"))
        if (!parse_int_val("TTS_TOP_K", v, params.top_k)) return 1;
    if (const char * v = std::getenv("TTS_REPETITION_PENALTY"))
        if (!parse_float_val("TTS_REPETITION_PENALTY", v, params.repetition_penalty)) return 1;
    if (const char * v = std::getenv("TTS_SEED"))
        if (!parse_i64_val("TTS_SEED", v, params.seed)) return 1;
    if (const char * v = std::getenv("TTS_CODEBOOKS"))
        if (!parse_int_val("TTS_CODEBOOKS", v, params.n_codebooks)) return 1;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model path\n");
                return 1;
            }
            model_path = argv[i];
        } else if (arg == "--vocoder") {
            if (++i >= argc) { fprintf(stderr, "Error: missing vocoder path\n"); return 1; }
            vocoder_path = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "-v" || arg == "--voice") {
            if (++i >= argc) { fprintf(stderr, "Error: missing voice name\n"); return 1; }
            voice_name = argv[i];
        } else if (arg == "--voices-dir") {
            if (++i >= argc) { fprintf(stderr, "Error: missing voices dir\n"); return 1; }
            voices_dir = argv[i];
        } else if (arg == "--list-voices") {
            list_voices = true;
        } else if (arg == "--ref-text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing ref-text\n");
                return 1;
            }
            params.ref_text = argv[i];
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            if (!parse_float_val("--temperature", argv[i], params.temperature)) return 1;
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            if (!parse_int_val("--top-k", argv[i], params.top_k)) return 1;
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            if (!parse_float_val("--repetition-penalty", argv[i], params.repetition_penalty)) return 1;
        } else if (arg == "--codebooks") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing codebooks value\n");
                return 1;
            }
            if (!parse_int_val("--codebooks", argv[i], params.n_codebooks)) return 1;
            if (params.n_codebooks < 1 || params.n_codebooks > 16) {
                fprintf(stderr, "Error: --codebooks must be between 1 and 16\n");
                return 1;
            }
        } else if (arg == "--seed") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing seed value\n");
                return 1;
            }
            if (!parse_i64_val("--seed", argv[i], params.seed)) return 1;
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            if (!parse_language(argv[i], params.language_id)) {
                fprintf(stderr, "Error: unknown language '%s'. Supported: en,ru,zh,ja,ko,de,fr,es,it,pt\n", argv[i]);
                return 1;
            }
        } else if (arg == "-i" || arg == "--instructions") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing instructions value\n");
                return 1;
            }
            params.instructions = argv[i];
        } else if (arg == "--streaming-batch-size") {
            if (++i >= argc) { fprintf(stderr, "Error: missing streaming-batch-size value\n"); return 1; }
            if (!parse_int_val("--streaming-batch-size", argv[i], streaming_batch_size)) return 1;
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            if (!parse_int_val("--threads", argv[i], params.n_threads)) return 1;
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (model_path.empty()) {
        fprintf(stderr, "Error: -m <path> is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Read text from stdin when -t/--text was not given and stdin is piped.
    if (!list_voices && text.empty() && !isatty(fileno(stdin))) {
        std::string buf;
        char chunk[4096];
        while (size_t n = fread(chunk, 1, sizeof(chunk), stdin)) {
            buf.append(chunk, n);
        }
        while (!buf.empty() && (buf.back() == '\n' || buf.back() == '\r' ||
                                buf.back() == ' '  || buf.back() == '\t')) {
            buf.pop_back();
        }
        text = std::move(buf);
    }

    if (!list_voices && text.empty()) {
        fprintf(stderr, "Error: text is required (pass via -t or pipe to stdin)\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!list_voices) {
        size_t n_chars = qwen3_tts::utf8_codepoints(text);
        if (n_chars > qwen3_tts::MAX_INPUT_CHARS) {
            fprintf(stderr,
                    "Error: input text exceeds %zu characters (got %zu). "
                    "Split it into chunks and synthesize them separately.\n",
                    qwen3_tts::MAX_INPUT_CHARS, n_chars);
            return 1;
        }
    }
    // Mirror the server's contract: any input that passed the char limit must
    // be fully synthesized — the library's 2048-token default would clip.
    params.max_audio_tokens = qwen3_tts::MAX_AUDIO_TOKENS;

    if (!voice_name.empty() && !reference_audio.empty()) {
        fprintf(stderr, "Error: --voice and --reference are mutually exclusive\n");
        return 1;
    }
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    bool loaded;
    const auto t_load_start = std::chrono::steady_clock::now();
    bool is_file = std::filesystem::is_regular_file(model_path);
    if (is_file) {
        if (vocoder_path.empty()) {
            fprintf(stderr, "Error: --vocoder <file> is required when -m is a file\n");
            return 1;
        }
        fprintf(stderr, "Loading talker: %s\n", model_path.c_str());
        fprintf(stderr, "Loading vocoder: %s\n", vocoder_path.c_str());
        loaded = tts.load_model_files(model_path, vocoder_path);
    } else {
        fprintf(stderr, "Loading models from: %s\n", model_path.c_str());
        loaded = tts.load_models(model_path);
    }
    if (!loaded) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    const int64_t model_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_load_start).count();
    tts.set_n_threads(params.n_threads);

    // Cloned voices only make sense on Base — they're encoded by the Base
    // speaker_encoder and the embedding space doesn't transfer to other variants.
    const bool model_supports_cloned_voices = (tts.get_model_type() == "base");
    if (!voices_dir.empty() && !model_supports_cloned_voices) {
        fprintf(stderr, "warning: model type '%s' cannot use cloned voices; "
                        "ignoring --voices-dir. Use a Base model for voice cloning.\n",
                tts.get_model_type().c_str());
        voices_dir.clear();
    }

    // Voice library — same on-disk layout as the server, optional.
    std::mutex synth_mutex;
    std::function<qwen3_tts::Qwen3TTS*()> ensure_loaded = [&tts]() { return &tts; };
    qwen3_tts::VoiceStore voice_store(voices_dir, ensure_loaded,
                                       tts.has_speaker_encoder(), &synth_mutex);
    if (!voices_dir.empty()) {
        voice_store.refresh();
    }

    const std::vector<std::string> & builtin_speakers = tts.get_speaker_names();

    if (list_voices) {
        fprintf(stderr, "Built-in speakers (%zu):\n", builtin_speakers.size());
        for (const auto & n : builtin_speakers) printf("  %s\n", n.c_str());
        if (!voices_dir.empty()) {
            auto lib = voice_store.list();
            fprintf(stderr, "Library voices in %s (%zu):\n", voices_dir.c_str(), lib.size());
            for (const auto & n : lib) printf("  %s\n", n.c_str());
        }
        return 0;
    }

    // Resolve --voice to either a built-in speaker embedding or an on-disk entry.
    std::vector<float>   voice_embedding;
    std::vector<int32_t> voice_ref_codes;
    int32_t              voice_n_ref_frames = 0;
    if (!voice_name.empty() && voice_name != "default") {
        bool is_builtin = std::find(builtin_speakers.begin(), builtin_speakers.end(),
                                    voice_name) != builtin_speakers.end();
        if (is_builtin) {
            if (!tts.get_speaker_embedding(voice_name, voice_embedding)) {
                fprintf(stderr, "Error: failed to get speaker embedding for '%s': %s\n",
                        voice_name.c_str(), tts.get_error().c_str());
                return 1;
            }
        } else {
            if (voices_dir.empty()) {
                if (!model_supports_cloned_voices) {
                    fprintf(stderr, "Error: voice '%s' not found among built-in speakers; "
                                    "this model can't use cloned voices either\n", voice_name.c_str());
                } else {
                    fprintf(stderr, "Error: unknown voice '%s' (no --voices-dir set)\n", voice_name.c_str());
                }
                return 1;
            }
            qwen3_tts::voice_entry ve;
            // get() pays the cost of cache lookup or fresh encoding only when
            // it actually misses — but the CLI was previously silent during
            // that wait, so print a heads-up plus the timing afterwards.
            fprintf(stderr, "voice '%s': loading...\n", voice_name.c_str());
            auto t_voice0 = std::chrono::steady_clock::now();
            if (!voice_store.get(voice_name, ve)) {
                fprintf(stderr, "Error: unknown voice '%s'\n", voice_name.c_str());
                return 1;
            }
            auto t_voice_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_voice0).count();
            fprintf(stderr, "voice '%s': ready (%.1fs)\n",
                    voice_name.c_str(), (double)t_voice_ms / 1000.0);
            voice_embedding    = std::move(ve.embedding);
            params.ref_text    = std::move(ve.ref_text);
            voice_ref_codes    = std::move(ve.ref_codes);
            voice_n_ref_frames = ve.n_ref_frames;
        }
    }

    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });

    // Generate speech
    qwen3_tts::tts_result result;

    qwen3_tts::streaming_opts stream_opts;
    size_t stream_batches_seen = 0;
    if (streaming_batch_size > 0) {
        stream_opts.batch_size = streaming_batch_size;
        stream_opts.on_pcm = [&stream_batches_seen](const float * /*pcm*/, size_t n) {
            stream_batches_seen++;
            fprintf(stderr, "\r[stream] batch %zu: %zu samples", stream_batches_seen, n);
            return true;
        };
    }

    if (!voice_embedding.empty()) {
        fprintf(stderr, "Synthesizing with voice '%s': \"%s\"\n", voice_name.c_str(), text.c_str());
        if (!voice_ref_codes.empty()) {
            result = tts.synthesize_with_embedding(
                text, voice_embedding.data(), (int32_t)voice_embedding.size(), params,
                voice_ref_codes.data(), voice_n_ref_frames,
                streaming_batch_size > 0 ? &stream_opts : nullptr);
        } else {
            result = tts.synthesize_with_embedding(
                text, voice_embedding.data(), (int32_t)voice_embedding.size(), params,
                nullptr, 0,
                streaming_batch_size > 0 ? &stream_opts : nullptr);
        }
    } else if (reference_audio.empty()) {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        if (streaming_batch_size > 0) {
            result = tts.synthesize(text, params, &stream_opts);
        } else {
            result = tts.synthesize(text, params);
        }
    } else {
        fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        result = tts.synthesize_with_voice(text, reference_audio, params);
    }
    
    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }
    // synthesize() doesn't know about model loading — fill it in here so the
    // timing block below stops printing a hardcoded 0.
    result.t_load_ms = model_load_ms;
    
    fprintf(stderr, "\n");
    
    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
        return 1;
    }
    
    fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
    fprintf(stderr, "Audio duration: %.2f seconds\n", 
            (float)result.audio.size() / result.sample_rate);
    
    // Print timing
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Load:      %6lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }
    
    return 0;
}
