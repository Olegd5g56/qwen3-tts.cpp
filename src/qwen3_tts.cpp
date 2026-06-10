#include "qwen3_tts.h"
#include "gguf_loader.h"
#include "audio_streamers.h"
#include "log.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <mutex>

#include <mpg123.h>

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace qwen3_tts {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

static bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#else
    // /proc/self/statm gives the CURRENT resident set; ru_maxrss only reports
    // the lifetime peak, which made the start/end/peak log lines all show the
    // same number.
    if (FILE * f = fopen("/proc/self/statm", "r")) {
        long pages_total = 0, pages_rss = 0;
        const int n = fscanf(f, "%ld %ld", &pages_total, &pages_rss);
        fclose(f);
        if (n == 2 && pages_rss > 0) {
            out.rss_bytes = (uint64_t) pages_rss * (uint64_t) sysconf(_SC_PAGESIZE);
            out.phys_footprint_bytes = out.rss_bytes;
            return true;
        }
    }
    // fallback (non-procfs systems): peak RSS is better than nothing
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

static std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

static void log_memory_usage(const char * label) {
    process_memory_snapshot mem;
    if (!get_process_memory_snapshot(mem)) {
        log_debug("[mem] %-24s unavailable", label);
        return;
    }
    log_debug("[mem] %-24s rss=%s  phys=%s",
              label, format_bytes(mem.rss_bytes).c_str(),
              format_bytes(mem.phys_footprint_bytes).c_str());
}

static void resample_linear(const float * input, int input_len, int input_rate,
                            std::vector<float> & output, int output_rate) {
    double ratio = (double)input_rate / output_rate;
    int output_len = (int)((double)input_len / ratio);
    output.resize(output_len);
    
    for (int i = 0; i < output_len; ++i) {
        double src_idx = i * ratio;
        int idx0 = (int)src_idx;
        int idx1 = idx0 + 1;
        double frac = src_idx - idx0;
        
        if (idx1 >= input_len) {
            output[i] = input[input_len - 1];
        } else {
            output[i] = (float)((1.0 - frac) * input[idx0] + frac * input[idx1]);
        }
    }
}

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

bool Qwen3TTS::load_models(const std::string & model_dir) {
    // discover talker (any *.gguf not matching tokenizer) and vocoder (*tokenizer*.gguf).
    // prefer q8_0 over f16 for talker.
    namespace fs = std::filesystem;
    std::string tts_path, vocoder_path;
    std::string tts_q8, tts_f16, tts_other;
    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(model_dir, ec)) {
        if (ec || !entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        std::string lower = name;
        for (auto & c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower.find("tokenizer") != std::string::npos) {
            vocoder_path = entry.path().string();
        } else if (lower.find("q8_0") != std::string::npos) {
            tts_q8 = entry.path().string();
        } else if (lower.find("f16") != std::string::npos || lower.find("fp16") != std::string::npos) {
            tts_f16 = entry.path().string();
        } else {
            tts_other = entry.path().string();
        }
    }
    tts_path = !tts_q8.empty() ? tts_q8 : (!tts_f16.empty() ? tts_f16 : tts_other);
    if (tts_path.empty() || vocoder_path.empty()) {
        error_msg_ = "could not find talker and tokenizer ggufs in " + model_dir;
        return false;
    }
    return load_model_files(tts_path, vocoder_path);
}

bool Qwen3TTS::load_model_files(const std::string & tts_path,
                                 const std::string & vocoder_path) {
    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    transformer_.unload_model();
    audio_decoder_.unload_model();
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    decoder_loaded_ = false;

    tts_model_path_ = tts_path;

    // derive vocoder path from same directory if not specified
    if (vocoder_path.empty()) {
        auto slash = tts_path.rfind('/');
        std::string dir = (slash != std::string::npos) ? tts_path.substr(0, slash) : ".";
        decoder_model_path_ = dir + "/qwen3-tts-tokenizer-f16.gguf";
    } else {
        decoder_model_path_ = vocoder_path;
    }

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        log_debug("low-memory mode enabled (lazy decoder + component unloads)");
    }

    // Load TTS model (contains text tokenizer + transformer for generation)
    log_debug("loading TTS model from %s", tts_model_path_.c_str());

    // Load text tokenizer from TTS model
    int64_t t_tokenizer_start = get_time_ms();
    {
        GGUFLoader loader;
        if (!loader.open(tts_model_path_)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }

        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        log_debug("text tokenizer loaded: vocab_size=%d (%lld ms)",
                  tokenizer_.get_config().vocab_size,
                  (long long)(get_time_ms() - t_tokenizer_start));
    }
    log_memory_usage("load/after-tokenizer");

    // Speaker encoder is loaded lazily on first voice cloning request.
    log_debug("speaker encoder: deferred (lazy load)");

    // Load TTS transformer from TTS model
    int64_t t_transformer_start = get_time_ms();
    if (!transformer_.load_model(tts_model_path_)) {
        error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
        log_error("%s", error_msg_.c_str());
        return false;
    }
    transformer_loaded_ = true;
    log_debug("TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)",
              transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
              (long long)(get_time_ms() - t_transformer_start));
    log_memory_usage("load/after-transformer");

    if (!low_mem_mode_) {
        // Load vocoder (audio decoder) from tokenizer model
        log_debug("loading vocoder from %s", decoder_model_path_.c_str());
        int64_t t_decoder_start = get_time_ms();
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            log_error("%s", error_msg_.c_str());
            return false;
        }
        decoder_loaded_ = true;
        log_debug("vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)",
                  audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
                  (long long)(get_time_ms() - t_decoder_start));
        log_memory_usage("load/after-vocoder");
    } else {
        log_debug("vocoder: deferred (lazy load)");
    }

    models_loaded_ = true;

    int64_t t_end = get_time_ms();
    log_debug("all models loaded in %lld ms", (long long)(t_end - t_start));
    log_memory_usage("load/end");

    return true;
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params) {
    return synthesize(text, params, nullptr);
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params,
                                 const streaming_opts * stream) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    // For basic synthesis without voice cloning, we use a zero speaker embedding
    // This will use the model's default voice characteristics
    std::vector<float> zero_embedding(transformer_.get_config().hidden_size, 0.0f);

    return synthesize_internal(text, zero_embedding.data(), params, result,
                               nullptr, 0, stream);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const std::string & reference_audio,
                                            const tts_params & params) {
    tts_result result;
    
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }
    
    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        log_debug("resampling audio from %d Hz to %d Hz", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }
    
    return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const float * ref_samples, int32_t n_ref_samples,
                                            const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
            return result;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return result;
        }
        audio_encoder_.set_abort_callback(abort_cb_, abort_data_);
        encoder_loaded_ = true;
        if (params.print_timing) {
            log_info("speaker encoder lazy-loaded in %lld ms",
                     (long long)(get_time_ms() - t_encoder_load_start));
            log_memory_usage("voice/after-encoder-load");
        }
    }

    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    if (params.print_progress) {
        log_info("speaker embedding extracted: %zu floats", speaker_embedding.size());
    }

    // ICL mode: also encode reference audio to discrete speech codes
    if (!params.ref_text.empty()) {
        if (!codec_encoder_loaded_) {
            if (decoder_model_path_.empty()) {
                result.error_msg = "missing tokenizer model path for codec encoder";
                return result;
            }
            int64_t t0 = get_time_ms();
            if (!codec_encoder_.load_model(decoder_model_path_)) {
                result.error_msg = "failed to load codec encoder: " + codec_encoder_.get_error();
                return result;
            }
            codec_encoder_loaded_ = true;
            if (params.print_timing) {
                log_info("codec encoder lazy-loaded in %lld ms",
                         (long long)(get_time_ms() - t0));
            }
        }

        int64_t t0 = get_time_ms();
        std::vector<int32_t> ref_codes;
        int32_t n_ref_frames = 0;
        if (!codec_encoder_.encode(ref_samples, n_ref_samples, ref_codes, n_ref_frames)) {
            result.error_msg = "failed to encode reference audio: " + codec_encoder_.get_error();
            return result;
        }
        if (params.print_progress) {
            log_info("reference audio encoded: %d frames x 16 codebooks (ICL mode)", n_ref_frames);
        }
        if (params.print_timing) {
            log_info("codec encode: %lld ms", (long long)(get_time_ms() - t0));
        }

        return synthesize_internal(text, speaker_embedding.data(), params, result,
                                   ref_codes.data(), n_ref_frames);
    }

    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

bool Qwen3TTS::extract_speaker_embedding(const std::string & reference_audio,
                                           std::vector<float> & embedding) {
    std::vector<float> ref_samples;
    int ref_sample_rate = 0;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        error_msg_ = "Failed to load reference audio: " + reference_audio;
        return false;
    }
    return extract_speaker_embedding(ref_samples.data(), (int32_t)ref_samples.size(),
                                      ref_sample_rate, embedding);
}

bool Qwen3TTS::extract_speaker_embedding(const float * samples, int32_t n_samples,
                                           int sample_rate, std::vector<float> & embedding) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    const int target_rate = 24000;
    std::vector<float> resampled_storage;
    const float * use_samples = samples;
    int32_t use_n = n_samples;
    if (sample_rate != target_rate) {
        resample_linear(samples, (int)n_samples, sample_rate, resampled_storage, target_rate);
        use_samples = resampled_storage.data();
        use_n = (int32_t)resampled_storage.size();
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        if (!audio_encoder_.load_model(tts_model_path_)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        audio_encoder_.set_abort_callback(abort_cb_, abort_data_);
        encoder_loaded_ = true;
    }

    if (!audio_encoder_.encode(use_samples, use_n, embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }

    return true;
}

bool Qwen3TTS::encode_speech_codes(const float * samples, int32_t n_samples,
                                    std::vector<int32_t> & codes, int32_t & n_frames) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    if (!codec_encoder_loaded_) {
        if (decoder_model_path_.empty()) {
            error_msg_ = "missing tokenizer model path for codec encoder";
            return false;
        }
        if (!codec_encoder_.load_model(decoder_model_path_)) {
            error_msg_ = "failed to load codec encoder: " + codec_encoder_.get_error();
            return false;
        }
        codec_encoder_loaded_ = true;
    }

    if (!codec_encoder_.encode(samples, n_samples, codes, n_frames)) {
        error_msg_ = "failed to encode speech codes: " + codec_encoder_.get_error();
        return false;
    }

    return true;
}

tts_result Qwen3TTS::synthesize_with_embedding(const std::string & text,
                                                 const float * embedding, int32_t embedding_size,
                                                 const tts_params & params,
                                                 const int32_t * ref_codes,
                                                 int32_t n_ref_frames,
                                                 const streaming_opts * stream) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!embedding) {
        result.error_msg = "Speaker embedding is null";
        return result;
    }

    const int32_t expected_size = transformer_.get_config().hidden_size;
    if (embedding_size != expected_size) {
        result.error_msg = "Invalid embedding size: expected " + std::to_string(expected_size)
                         + ", got " + std::to_string(embedding_size);
        return result;
    }

    return synthesize_internal(text, embedding, params, result, ref_codes, n_ref_frames, stream);
}

bool Qwen3TTS::warmup_decoder_for_icl(const int32_t * ref_codes, int32_t n_ref_frames) {
    // Process-wide cache: the server destroys and recreates Qwen3TTS on
    // idle unload, but the snapshots (host-side float buffers) stay valid
    // as long as the same vocoder weights are used — so the vocoder path
    // is mixed into the key instead of tying the cache to this instance.
    static std::mutex cache_mutex;
    static std::unordered_map<uint64_t, AudioTokenizerDecoder::stream_state> cache;

    // FNV-1a over the raw codes + vocoder path. A handful of voices live
    // in the cache at most, so collisions are not a realistic concern.
    const int n_cb = audio_decoder_.get_config().n_codebooks;
    const size_t n_bytes = (size_t) n_ref_frames * n_cb * sizeof(int32_t);
    const uint8_t * p = (const uint8_t *) ref_codes;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n_bytes; ++i) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    for (char c : decoder_model_path_) {
        h = (h ^ (uint8_t) c) * 1099511628211ull;
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(h);
        if (it != cache.end()) {
            audio_decoder_.stream_state_restore(it->second);
            log_debug("[icl] decoder warm-up cache hit (%d ref frames)", n_ref_frames);
            return true;
        }
    }

    audio_decoder_.stream_reset();
    std::vector<float> warmup_pcm;
    if (!audio_decoder_.stream_decode(ref_codes, n_ref_frames, warmup_pcm)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    // ~10 MB of host state per voice; bound the cache crudely rather than
    // tracking LRU order for what is normally 1-3 entries.
    if (cache.size() >= 16) {
        cache.clear();
    }
    audio_decoder_.stream_state_save(cache[h]);
    return true;
}

tts_result Qwen3TTS::synthesize_internal(const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & params,
                                          tts_result & result,
                                          const int32_t * ref_codes,
                                          int32_t n_ref_frames,
                                          const streaming_opts * stream) {
    int64_t t_total_start = get_time_ms();
    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            return;
        }
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        if (mem.rss_bytes > result.mem_rss_peak_bytes) {
            result.mem_rss_peak_bytes = mem.rss_bytes;
        }
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        }
        if (params.print_timing) {
            log_info("[mem] %-24s rss=%s  phys=%s",
                     stage,
                     format_bytes(mem.rss_bytes).c_str(),
                     format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };
    sample_memory("synth/start");
    
    // Step 2: Tokenize input text and optional instruction prompt
    int64_t t_tokenize_start = get_time_ms();
    std::vector<int32_t> text_tokens = tokenizer_.encode_for_tts(text);
    std::vector<int32_t> instruct_tokens;
    if (!params.instructions.empty() && transformer_.get_config().model_size != "0b6") {
        instruct_tokens = tokenizer_.encode_instruct(params.instructions);
    }
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    result.n_text_tokens = (int32_t)text_tokens.size();
    sample_memory("synth/after-tokenize");

    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    
    if (params.print_progress) {
        std::string toks;
        for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
            toks += std::to_string(text_tokens[i]);
            toks += ' ';
        }
        if (text_tokens.size() > 10) toks += "...";
        log_info("text tokenized: %zu tokens (%s)", text_tokens.size(), toks.c_str());
    }
    
    // Step 3: Generate speech codes using TTS transformer
    int64_t t_generate_start = get_time_ms();
    if (!transformer_loaded_) {
        int64_t t_reload_start = get_time_ms();
        if (!transformer_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to reload TTS transformer: " + transformer_.get_error();
            return result;
        }
        transformer_.set_abort_callback(abort_cb_, abort_data_);
        transformer_loaded_ = true;
        if (params.print_timing) {
            log_info("transformer reloaded in %lld ms",
                     (long long)(get_time_ms() - t_reload_start));
            sample_memory("synth/after-transformer-reload");
        }
    }
    transformer_.clear_kv_cache();
    transformer_.set_verbose(params.print_progress);
    if (params.seed >= 0) {
        transformer_.set_seed((uint64_t)params.seed);
    }

    // tokenize ref_text for ICL mode
    std::vector<int32_t> ref_text_tokens;
    if (ref_codes && n_ref_frames > 0 && !params.ref_text.empty()) {
        ref_text_tokens = tokenizer_.encode_for_tts(params.ref_text);
        // python _build_ref_text wraps as <|im_start|>assistant\n{text}<|im_end|>\n
        // and slices ref_id[:, 3:-2], yielding just the content tokens. our
        // encode_for_tts uses the longer assistant wrap with a trailing
        // <|im_start|>assistant\n (8 framing tokens), so we drop 3 prefix + 5
        // suffix tokens to land on the same content-only window. leaving the
        // boundary tokens in causes the talker to treat ref+new as separate
        // turns, producing hangs and ref-text interjection.
        if ((int)ref_text_tokens.size() > 8) {
            ref_text_tokens = std::vector<int32_t>(
                ref_text_tokens.begin() + 3,
                ref_text_tokens.end() - 5
            );
        } else {
            ref_text_tokens.clear();
        }
    }

    // streaming: install per-frame callback that batches codes and live-decodes
    // via the vocoder's streaming path. we must also ensure the decoder is
    // loaded before generate() so its stream_decode can be called from within
    // the callback. ICL warm-up (ref_codes) is fed as a discarded chunk below.
    const bool streaming = stream && stream->batch_size > 0 && stream->on_pcm;
    std::vector<int32_t> stream_buf;
    size_t stream_cb_count = 0;
    bool stream_cb_aborted = false;
    if (streaming) {
        if (!decoder_loaded_) {
            int64_t t_decoder_load_start = get_time_ms();
            if (decoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing vocoder model path";
                return result;
            }
            if (!audio_decoder_.load_model(decoder_model_path_)) {
                result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
                return result;
            }
            audio_decoder_.set_abort_callback(abort_cb_, abort_data_);
            decoder_loaded_ = true;
            if (params.print_timing) {
                log_info("vocoder lazy-loaded in %lld ms",
                         (long long)(get_time_ms() - t_decoder_load_start));
                sample_memory("synth/after-vocoder-load-stream");
            }
        }
        audio_decoder_.stream_reset();
        const int n_cb = transformer_.get_config().n_codebooks;

        // ICL warm-up: run ref_codes through the streaming decoder (cached
        // per voice) so the live PCM stream starts in the reference timbre.
        if (ref_codes && n_ref_frames > 0 && !params.ref_text.empty()) {
            if (!warmup_decoder_for_icl(ref_codes, n_ref_frames)) {
                result.error_msg = "Failed to warm-up vocoder with ref codes: " + audio_decoder_.get_error();
                return result;
            }
        }

        stream_buf.reserve((size_t) stream->batch_size * n_cb);
        transformer_.set_frame_callback(
            [this, stream, &stream_buf, &stream_cb_count, &stream_cb_aborted, n_cb, &result]
            (int32_t /*frame_idx*/, const int32_t * frame_codes) -> bool {
                for (int c = 0; c < n_cb; ++c) stream_buf.push_back(frame_codes[c]);
                const int frames_buffered = (int) (stream_buf.size() / n_cb);
                if (frames_buffered >= stream->batch_size) {
                    std::vector<float> pcm;
                    if (!audio_decoder_.stream_decode(stream_buf.data(), frames_buffered, pcm)) {
                        stream_cb_aborted = true;
                        return false;
                    }
                    stream_buf.clear();
                    result.audio.insert(result.audio.end(), pcm.begin(), pcm.end());
                    stream_cb_count++;
                    if (!stream->on_pcm(pcm.data(), pcm.size())) {
                        stream_cb_aborted = true;
                        return false;
                    }
                }
                return true;
            });
    }

    std::vector<int32_t> speech_codes;
    if (!transformer_.generate(text_tokens.data(), (int32_t)text_tokens.size(),
                               speaker_embedding, params.max_audio_tokens, speech_codes,
                               params.language_id, params.repetition_penalty,
                               params.temperature, params.top_k,
                               instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                               (int32_t)instruct_tokens.size(),
                               ref_text_tokens.empty() ? nullptr : ref_text_tokens.data(),
                               (int32_t)ref_text_tokens.size(),
                               ref_codes, n_ref_frames)) {
        if (streaming) transformer_.set_frame_callback({});
        result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
        return result;
    }
    if (streaming) {
        transformer_.set_frame_callback({});
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
    result.n_prefill_tokens = transformer_.get_last_n_prefill_tokens();
    result.t_prefill_ms     = transformer_.get_last_prefill_ms();
    sample_memory("synth/after-generate");

    if (is_aborted()) {
        result.error_msg = "Aborted";
        return result;
    }

    int n_codebooks = transformer_.get_config().n_codebooks;
    int n_frames = (int)speech_codes.size() / n_codebooks;
    result.n_audio_tokens = n_frames;

    if (params.print_progress) {
        log_info("speech codes generated: %d frames x %d codebooks", n_frames, n_codebooks);
    }
    
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    if (low_mem_mode_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
        sample_memory("synth/after-transformer-unload");
    }
    
    // Step 4: Decode speech codes to waveform using vocoder
    int64_t t_decode_start = get_time_ms();

    // streaming path: frames were already decoded live in the frame callback.
    // flush any residual (< batch_size) frames now, then skip the one-shot
    // decode that follows.
    if (streaming) {
        const int n_cb = transformer_.get_config().n_codebooks;
        const int leftover = (int) (stream_buf.size() / n_cb);
        if (leftover > 0 && !stream_cb_aborted) {
            std::vector<float> pcm;
            if (!audio_decoder_.stream_decode(stream_buf.data(), leftover, pcm)) {
                result.error_msg = "Failed to flush streaming vocoder: " + audio_decoder_.get_error();
                return result;
            }
            stream_buf.clear();
            result.audio.insert(result.audio.end(), pcm.begin(), pcm.end());
            if (!stream->on_pcm(pcm.data(), pcm.size())) {
                stream_cb_aborted = true;
            }
        }
        result.t_decode_ms = get_time_ms() - t_decode_start;
        sample_memory("synth/after-stream-decode");
        if (params.print_progress) {
            log_info("streaming: %zu batches dispatched, %zu total samples",
                     stream_cb_count, result.audio.size());
        }
        result.sample_rate = audio_decoder_.get_config().sample_rate;
        result.success = !stream_cb_aborted;
        if (stream_cb_aborted && result.error_msg.empty()) {
            result.error_msg = "Streaming consumer aborted";
        }
        result.t_total_ms = get_time_ms() - t_total_start;
        sample_memory("synth/end");
        return result;
    }

    if (!decoder_loaded_) {
        int64_t t_decoder_load_start = get_time_ms();
        if (decoder_model_path_.empty()) {
            result.error_msg = "Internal error: missing vocoder model path";
            return result;
        }
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return result;
        }
        audio_decoder_.set_abort_callback(abort_cb_, abort_data_);
        decoder_loaded_ = true;
        if (params.print_timing) {
            log_info("vocoder lazy-loaded in %lld ms",
                     (long long)(get_time_ms() - t_decoder_load_start));
            sample_memory("synth/after-vocoder-load");
        }
    }
    
    // Chunked decode via the streaming vocoder path. The one-shot decode()
    // builds a single graph over all n_frames; the conv pyramid upsamples
    // 12.5Hz → 24kHz (~480x), so activation memory balloons past 13 GB for
    // ~3k frames. Chunking caps the working set to one DECODE_BATCH worth of
    // frames; the persistent KV / tail-ring state stitches chunks together
    // seamlessly (same mechanism as the live-stream path above).
    //
    // ICL: feed ref_codes first and discard their PCM so result.audio starts
    // at the first generated frame. Equivalent to the old prepend-then-trim
    // trick, but the streaming path lets us drop the cut math.
    audio_decoder_.stream_reset();
    const bool icl = ref_codes && n_ref_frames > 0 && !params.ref_text.empty();
    if (icl) {
        if (!warmup_decoder_for_icl(ref_codes, n_ref_frames)) {
            result.error_msg = "Failed to warm-up vocoder with ref codes: "
                               + audio_decoder_.get_error();
            return result;
        }
    }
    if (params.print_timing) {
        log_info("[icl] ref_frames=%d new_frames=%d prepended=%d",
                 n_ref_frames, n_frames, (int)icl);
    }
    if (const char * dp = std::getenv("QWEN3_TTS_DUMP_CODES")) {
        static std::atomic<int> dump_counter{0};
        int idx = dump_counter.fetch_add(1);
        std::string path = std::string(dp);
        if (path.find("%d") != std::string::npos) {
            char buf[1024];
            snprintf(buf, sizeof(buf), dp, idx);
            path = buf;
        }
        FILE * fp = fopen(path.c_str(), "wb");
        if (fp) {
            int32_t hdr[3] = { n_ref_frames, n_frames, n_codebooks };
            fwrite(hdr, sizeof(int32_t), 3, fp);
            if (ref_codes && n_ref_frames > 0) {
                fwrite(ref_codes, sizeof(int32_t), (size_t)n_ref_frames * n_codebooks, fp);
            }
            fwrite(speech_codes.data(), sizeof(int32_t), speech_codes.size(), fp);
            fclose(fp);
            log_info("dumped ref+new codes to %s", path.c_str());
        }
    }

    constexpr int32_t DECODE_BATCH = 100;
    for (int32_t off = 0; off < n_frames; off += DECODE_BATCH) {
        int32_t batch = std::min(DECODE_BATCH, n_frames - off);
        if (!audio_decoder_.stream_decode(
                speech_codes.data() + (size_t)off * n_codebooks,
                batch, result.audio)) {
            result.error_msg = "Failed to decode speech codes: "
                               + audio_decoder_.get_error();
            return result;
        }
        if (is_aborted()) {
            result.error_msg = "Aborted";
            return result;
        }
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }
    
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");
    
    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        log_info("timing: tokenize=%lld ms  encode=%lld ms  generate=%lld ms  decode=%lld ms  total=%lld ms",
                 (long long)result.t_tokenize_ms,
                 (long long)result.t_encode_ms,
                 (long long)result.t_generate_ms,
                 (long long)result.t_decode_ms,
                 (long long)result.t_total_ms);
        log_info("audio: %.2fs duration  %.2fx realtime (RTF=%.3f)",
                 audio_sec, x_realtime, realtime_factor);
        log_info("memory rss:  start=%s  end=%s  peak=%s",
                 format_bytes(result.mem_rss_start_bytes).c_str(),
                 format_bytes(result.mem_rss_end_bytes).c_str(),
                 format_bytes(result.mem_rss_peak_bytes).c_str());
        log_info("memory phys: start=%s  end=%s  peak=%s",
                 format_bytes(result.mem_phys_start_bytes).c_str(),
                 format_bytes(result.mem_phys_end_bytes).c_str(),
                 format_bytes(result.mem_phys_peak_bytes).c_str());
    }
    
    return result;
}

int32_t Qwen3TTS::get_hidden_size() const {
    return transformer_.get_config().hidden_size;
}

const std::string & Qwen3TTS::get_model_type() const {
    return transformer_.get_config().model_type;
}

const std::vector<std::string> & Qwen3TTS::get_speaker_names() const {
    return transformer_.get_config().speaker_names;
}

const std::vector<int32_t> & Qwen3TTS::get_speaker_ids() const {
    return transformer_.get_config().speaker_ids;
}

bool Qwen3TTS::has_speaker_encoder() const {
    return transformer_.get_config().has_speaker_encoder;
}

int32_t Qwen3TTS::get_speaker_id(const std::string & name) const {
    auto & names = transformer_.get_config().speaker_names;
    auto & ids = transformer_.get_config().speaker_ids;
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == name) return ids[i];
    }
    return -1;
}

bool Qwen3TTS::get_speaker_embedding(const std::string & name, std::vector<float> & embedding) {
    int32_t spk_id = get_speaker_id(name);
    if (spk_id < 0) {
        error_msg_ = "unknown speaker: " + name;
        return false;
    }
    if (!transformer_.get_codec_embedding(spk_id, embedding)) {
        error_msg_ = "failed to get speaker embedding: " + transformer_.get_error();
        return false;
    }
    return true;
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

void Qwen3TTS::set_abort_callback(ggml_abort_callback callback, void * data) {
    abort_cb_ = callback;
    abort_data_ = data;
    transformer_.set_abort_callback(callback, data);
    audio_encoder_.set_abort_callback(callback, data);
    audio_decoder_.set_abort_callback(callback, data);
}

void Qwen3TTS::set_n_threads(int32_t n_threads) {
    n_threads_ = n_threads;
    // Push into every component — whether or not it's currently loaded.
    // Each stores the value and (re)applies it on next backend init, so
    // lazy-loaded components inherit the setting automatically.
    transformer_.set_n_threads(n_threads);
    audio_encoder_.set_n_threads(n_threads);
    codec_encoder_.set_n_threads(n_threads);
    audio_decoder_.set_n_threads(n_threads);
}

// Mix an interleaved multi-channel buffer down to mono, storing results in `out`.
// InputT must be convertible to float; `scale` is applied before summing.
template<typename InputT>
static void mix_to_mono(const InputT * interleaved, int n_samples, int num_channels,
                        float scale, std::vector<float> & out) {
    out.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < num_channels; ++c) {
            sum += static_cast<float>(interleaved[i * num_channels + c]) * scale;
        }
        out[i] = sum / num_channels;
    }
}

// Get lowercase file extension from path
static std::string get_file_extension(const std::string & path) {
    auto dot_pos = path.rfind('.');
    if (dot_pos == std::string::npos) return "";
    std::string ext = path.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

// WAV stream loading (16-bit PCM, 32-bit PCM, or 32-bit IEEE float).
// Takes an already-opened FILE* so both real files and fmemopen()-wrapped
// byte buffers go through the same parser. Caller owns the FILE*.
static bool load_wav_stream(FILE * f, std::vector<float> & samples,
                            int & sample_rate) {
    // Read RIFF header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        log_error("not a RIFF file");
        return false;
    }

    uint32_t file_size;
    if (fread(&file_size, 4, 1, f) != 1) {
        return false;
    }

    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        log_error("not a WAVE file");
        return false;
    }

    // Find fmt and data chunks
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;

    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;

        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) break;
            if (fread(&num_channels, 2, 1, f) != 1) break;
            if (fread(&sr, 4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR);  // Skip byte rate and block align
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;

            // Handle WAVE_FORMAT_EXTENSIBLE: read actual format from SubFormat GUID
            if (audio_format == 0xFFFE && chunk_size >= 40) {
                fseek(f, 8, SEEK_CUR);  // Skip cbSize(2) + validBitsPerSample(2) + channelMask(4)
                uint16_t sub_format = 0;
                if (fread(&sub_format, 2, 1, f) != 1) break;
                audio_format = sub_format;
                // Skip remaining SubFormat GUID bytes and any extra data
                fseek(f, chunk_size - 26, SEEK_CUR);
            }
            // Skip any extra format bytes for non-extensible formats
            else if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
            // RIFF chunks are word-aligned: odd-sized chunks carry a pad byte
            if (chunk_size & 1) fseek(f, 1, SEEK_CUR);
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            // A zero channel count would divide by zero below (SIGFPE — not
            // an exception, the whole process dies). Covers both a malformed
            // fmt chunk and a data chunk that precedes fmt entirely.
            if (num_channels == 0) {
                log_error("invalid WAV: zero channels (or data chunk before fmt)");
                return false;
            }
            // Reference audio is seconds long; a multi-GB data chunk is a
            // corrupt header, not a real request — refuse before allocating.
            if (chunk_size > (1u << 30)) {
                log_error("invalid WAV: data chunk claims %u bytes", chunk_size);
                return false;
            }
            sample_rate = sr;

            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n_samples = chunk_size / (2 * num_channels);
                    std::vector<int16_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 2, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        return false;
                    }
                    mix_to_mono(raw.data(), n_samples, num_channels, 1.0f / 32768.0f, samples);
                }
                else if (bits_per_sample == 32) {
                    int n_samples = chunk_size / (4 * num_channels);
                    std::vector<int32_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        return false;
                    }
                    mix_to_mono(raw.data(), n_samples, num_channels, 1.0f / 2147483648.0f, samples);
                }
                else {
                    log_error("unsupported bits per sample: %d", bits_per_sample);
                    return false;
                }
            }
            else if (audio_format == 3) {  // IEEE float
                int n_samples = chunk_size / (4 * num_channels);
                std::vector<float> raw(n_samples * num_channels);
                if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                    return false;
                }
                mix_to_mono(raw.data(), n_samples, num_channels, 1.0f, samples);
            }
            else {
                log_error("unsupported audio format: %d", audio_format);
                return false;
            }

            return true;
        }
        else {
            // Skip unknown chunk (+ pad byte: RIFF chunks are word-aligned)
            fseek(f, chunk_size + (chunk_size & 1), SEEK_CUR);
        }
    }

    log_error("no data chunk found");
    return false;
}

// Path-based WAV loader: opens the file and delegates to load_wav_stream.
static bool load_wav_file(const std::string & path, std::vector<float> & samples,
                          int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        log_error("cannot open WAV file: %s", path.c_str());
        return false;
    }
    bool ok = load_wav_stream(f, samples, sample_rate);
    fclose(f);
    return ok;
}

// Bytes-based WAV loader: wraps the buffer with fmemopen so the existing
// parser can read from memory unchanged.
static bool load_wav_bytes(const void * data, size_t len,
                           std::vector<float> & samples, int & sample_rate) {
    FILE * f = fmemopen(const_cast<void*>(data), len, "rb");
    if (!f) {
        log_error("fmemopen failed for WAV bytes");
        return false;
    }
    bool ok = load_wav_stream(f, samples, sample_rate);
    fclose(f);
    return ok;
}

// Initialise mpg123 once per process and return a handle preconfigured to
// emit float32 mono/stereo at whatever rate the source frame announces.
// Caller takes ownership of the handle (mpg123_close + mpg123_delete).
static mpg123_handle * mpg123_make_float_handle() {
    static std::once_flag mpg123_init_flag;
    std::call_once(mpg123_init_flag, []() { mpg123_init(); });

    int err = MPG123_OK;
    mpg123_handle * h = mpg123_new(nullptr, &err);
    if (!h) {
        log_error("mpg123_new failed: %s", mpg123_plain_strerror(err));
        return nullptr;
    }

    // Restrict allowed output formats to float32 BEFORE the first decode —
    // once a frame is parsed the format is locked, and later format calls
    // are silently ignored.
    const long * rates_list = nullptr;
    size_t n_rates = 0;
    mpg123_rates(&rates_list, &n_rates);
    mpg123_format_none(h);
    for (size_t i = 0; i < n_rates; i++) {
        mpg123_format(h, rates_list[i], MPG123_MONO | MPG123_STEREO, MPG123_ENC_FLOAT_32);
    }
    return h;
}

// Drain a fully-prepared mpg123 handle into mono float samples at the
// source rate. Common tail of the file and the in-memory MP3 paths.
static bool mpg123_drain_to_mono(mpg123_handle * h, std::vector<float> & samples,
                                  int & sample_rate) {
    long rate = 0;
    int channels = 0;
    int encoding = 0;
    if (mpg123_getformat(h, &rate, &channels, &encoding) != MPG123_OK) {
        log_error("mpg123_getformat failed: %s", mpg123_strerror(h));
        return false;
    }
    if (encoding != MPG123_ENC_FLOAT_32) {
        log_error("mpg123 negotiated encoding %d, expected float32", encoding);
        return false;
    }

    std::vector<float> interleaved;
    const size_t chunk_floats = 16384;
    std::vector<unsigned char> buf(chunk_floats * sizeof(float));
    size_t done = 0;
    int rc;
    while (true) {
        rc = mpg123_read(h, buf.data(), buf.size(), &done);
        if (done > 0) {
            const float * f = reinterpret_cast<const float *>(buf.data());
            interleaved.insert(interleaved.end(), f, f + done / sizeof(float));
        }
        if (rc == MPG123_DONE || rc == MPG123_NEED_MORE) break;
        if (rc == MPG123_OK || rc == MPG123_NEW_FORMAT) continue;
        log_error("mpg123_read failed: %s", mpg123_strerror(h));
        return false;
    }

    if (channels <= 0 || interleaved.empty()) {
        log_error("MP3 decoded to zero samples");
        return false;
    }

    const int n_samples = (int)(interleaved.size() / channels);
    mix_to_mono(interleaved.data(), n_samples, channels, 1.0f, samples);
    sample_rate = (int)rate;
    return true;
}

// MP3 file loading via libmpg123.
static bool load_mp3_file(const std::string & path, std::vector<float> & samples,
                          int & sample_rate) {
    mpg123_handle * h = mpg123_make_float_handle();
    if (!h) return false;

    if (mpg123_open(h, path.c_str()) != MPG123_OK) {
        log_error("cannot open MP3 file '%s': %s", path.c_str(), mpg123_strerror(h));
        mpg123_delete(h);
        return false;
    }

    const bool ok = mpg123_drain_to_mono(h, samples, sample_rate);
    mpg123_close(h);
    mpg123_delete(h);
    return ok;
}

// MP3 in-memory loading via libmpg123 feed API. Used for HTTP uploads where
// the bytes never hit the filesystem.
static bool load_mp3_bytes(const void * data, size_t len,
                           std::vector<float> & samples, int & sample_rate) {
    mpg123_handle * h = mpg123_make_float_handle();
    if (!h) return false;

    if (mpg123_open_feed(h) != MPG123_OK) {
        log_error("mpg123_open_feed failed: %s", mpg123_strerror(h));
        mpg123_delete(h);
        return false;
    }
    if (mpg123_feed(h, static_cast<const unsigned char *>(data), len) != MPG123_OK) {
        log_error("mpg123_feed failed: %s", mpg123_strerror(h));
        mpg123_close(h);
        mpg123_delete(h);
        return false;
    }

    const bool ok = mpg123_drain_to_mono(h, samples, sample_rate);
    mpg123_close(h);
    mpg123_delete(h);
    return ok;
}

// Audio file loading - dispatches to format-specific loaders based on file extension
bool load_audio_file(const std::string & path, std::vector<float> & samples,
                     int & sample_rate) {
    std::string ext = get_file_extension(path);

    if (ext == ".wav") {
        return load_wav_file(path, samples, sample_rate);
    } else if (ext == ".mp3") {
        return load_mp3_file(path, samples, sample_rate);
    } else {
        log_error("unsupported audio format '%s'. Supported formats: .wav, .mp3", ext.c_str());
        return false;
    }
}

// In-memory audio loading. Sniffs the first few bytes to pick a decoder so
// callers (the voice-upload endpoint) don't need to track the source format.
bool load_audio_bytes(const void * data, size_t len,
                      std::vector<float> & samples, int & sample_rate) {
    if (len < 4) {
        log_error("audio buffer too small (%zu bytes)", len);
        return false;
    }
    const uint8_t * p = static_cast<const uint8_t *>(data);

    // RIFF....WAVE — wav
    if (p[0] == 'R' && p[1] == 'I' && p[2] == 'F' && p[3] == 'F') {
        return load_wav_bytes(data, len, samples, sample_rate);
    }
    // "ID3" tag header — mp3 with metadata
    if (p[0] == 'I' && p[1] == 'D' && p[2] == '3') {
        return load_mp3_bytes(data, len, samples, sample_rate);
    }
    // Raw MPEG frame sync: 0xFF and top 3 bits of next byte set
    if (p[0] == 0xFF && (p[1] & 0xE0) == 0xE0) {
        return load_mp3_bytes(data, len, samples, sample_rate);
    }

    log_error("unrecognised audio format (magic bytes: %02X %02X %02X %02X)",
              p[0], p[1], p[2], p[3]);
    return false;
}

// Encode float32 mono PCM as a complete WAV byte buffer (16-bit PCM at sample_rate).
std::string encode_wav(const std::vector<float> & samples, int sample_rate) {
    const int num_channels    = 1;
    const int bits_per_sample = 16;
    const int byte_rate       = sample_rate * num_channels * bits_per_sample / 8;
    const int block_align     = num_channels * bits_per_sample / 8;
    const int data_size       = (int)samples.size() * block_align;
    const int file_size       = 36 + data_size;

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

    memcpy(p,      "RIFF", 4); write_u32(p + 4,  file_size);
    memcpy(p + 8,  "WAVE", 4);
    memcpy(p + 12, "fmt ", 4); write_u32(p + 16, 16);
    write_u16(p + 20, 1); // PCM
    write_u16(p + 22, num_channels);
    write_u32(p + 24, sample_rate);
    write_u32(p + 28, byte_rate);
    write_u16(p + 32, block_align);
    write_u16(p + 34, bits_per_sample);
    memcpy(p + 36, "data", 4); write_u32(p + 40, data_size);

    int16_t * dst = reinterpret_cast<int16_t *>(p + 44);
    for (size_t i = 0; i < samples.size(); i++) {
        float s = samples[i];
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        dst[i] = (int16_t)(s * 32767.0f);
    }
    return buf;
}

// Encode float32 mono PCM as a self-contained MP3 byte buffer.
// VBR -V 4 speech-tuned (~70 kbps mono) — no client-facing knob (OpenAI spec doesn't expose one).
// Input samples are expected in IEEE float range [-1.0, +1.0].
// Thin wrapper over mp3_streamer with a string-appending sink so the encoder
// parameters live in one place.
std::string encode_mp3(const std::vector<float> & samples, int sample_rate) {
    std::string out;
    mp3_streamer enc;
    enc.on_page = [&out](const char * p, size_t n) {
        out.append(p, n);
        return true;
    };
    if (!enc.open(sample_rate)) return {};
    if (!samples.empty() && !enc.write(samples.data(), samples.size())) return {};
    enc.finish();
    if (enc.failed) return {};

    // Overwrite LAME's placeholder Info/Xing frame (emitted at init_params time)
    // with the real one. Without this, players can't report duration and
    // libmpg123 spits "part2_3_length too large" on the first frames because
    // the placeholder's ancillary bytes corrupt the bit-reservoir state of
    // the following frame.
    unsigned char tag[7200];
    size_t tag_size = lame_get_lametag_frame(enc.lame, tag, sizeof(tag));
    if (tag_size > 0 && tag_size <= out.size()) {
        std::memcpy(&out[0], tag, tag_size);
    }
    return out;
}

// Encode float32 mono PCM as a self-contained Ogg/Opus byte buffer.
// sample_rate must be 8/12/16/24/48 kHz (libopusenc constraint).
std::string encode_opus(const std::vector<float> & samples, int sample_rate) {
    std::string out;
    opus_streamer enc;
    enc.on_page = [&out](const char * p, size_t n) {
        out.append(p, n);
        return true;
    };
    if (!enc.open(sample_rate)) return {};
    if (!samples.empty() && !enc.write(samples.data(), samples.size())) return {};
    enc.finish();
    if (enc.failed) return {};
    return out;
}

static std::string str_tolower(std::string s) {
    for (auto & c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Save audio to disk, dispatching on file extension:
//   .wav         -> 16-bit PCM WAV
//   .mp3         -> LAME VBR -V 4 (speech-tuned)
//   .opus / .ogg -> Ogg/Opus
// Unknown extensions are rejected loudly — the CLI is a tool, not a guesser.
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    auto dot = path.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? "" : str_tolower(path.substr(dot + 1));

    std::string bytes;
    if (ext == "wav") {
        bytes = encode_wav(samples, sample_rate);
    } else if (ext == "mp3") {
        bytes = encode_mp3(samples, sample_rate);
    } else if (ext == "opus" || ext == "ogg") {
        bytes = encode_opus(samples, sample_rate);
    } else {
        log_error("unsupported output extension '.%s' — use .wav, .mp3, .opus, or .ogg",
                  ext.c_str());
        return false;
    }
    if (bytes.empty()) {
        log_error("failed to encode audio for %s", path.c_str());
        return false;
    }

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        log_error("cannot create output file: %s", path.c_str());
        return false;
    }
    size_t w = fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
    if (w != bytes.size()) {
        log_error("short write to %s (%zu/%zu)", path.c_str(), w, bytes.size());
        return false;
    }
    return true;
}

} // namespace qwen3_tts
