#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_codec_encoder.h"
#include "audio_tokenizer_decoder.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <cstddef>

namespace qwen3_tts {

// Public API limits shared by the CLI and the HTTP server. OpenAI's TTS API
// caps `input` at 4096 characters (codepoints, not bytes); we mirror that
// exactly so requests that pass us also pass the upstream service. The audio
// token budget is sized so any valid 4096-char request can be fully
// synthesized without hitting the per-call cap (1.5x covers worst-case slow
// voices / long pauses at ~12.5 Hz codec rate).
inline constexpr size_t  MAX_INPUT_CHARS  = 4096;
inline constexpr int32_t MAX_AUDIO_TOKENS = 6144;

// Count UTF-8 codepoints in a string. Assumes valid UTF-8 (JSON parser
// guarantees that on the server path; CLI reads from stdin/argv where we
// trust the OS encoding). Starter bytes are anything that isn't a
// continuation byte (10xxxxxx).
inline size_t utf8_codepoints(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++n;
    }
    return n;
}

// Runaway guard. Qwen3-TTS base models sometimes never emit EOS and keep
// generating until the caller's cap: upstream QwenLM/Qwen3-TTS#118 reports
// ~0.5% of calls and was closed as "not planned", and #211 notes it gets more
// likely on long input. Measured here at 1 in 15 on a 4067-char Russian text
// and 0 in 140 on a 727-char one. See docs/known-issues.md #11.
//
// A per-request budget derived from the input keeps a runaway from costing
// minutes: on a short line the guard fires in about a second, which is what
// makes an automatic retry cheap enough to be worth doing.
//
// Constants: measured 0.70 codec frames per character of Russian on this fork
// (727 chars -> 508 frames; 4067 chars -> 2830 frames), worst case 0.76. Han,
// kana and hangul pack a whole syllable into one codepoint, so they need
// several frames each. Both constants sit ~1.8x above the worst case observed
// for that script — this is a guard against a broken generation, not a length
// predictor, and firing it on a legitimately slow voice would be far worse
// than letting a rare runaway run a little longer.
inline int32_t audio_token_budget(const std::string & text, int32_t hard_cap) {
    size_t chars = 0;
    size_t dense = 0;  // codepoints that carry a full syllable
    for (size_t i = 0; i < text.size(); ) {
        const unsigned char c = text[i];
        uint32_t cp = c;
        size_t len = 1;
        if      ((c & 0x80) == 0x00) { cp = c;        len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        for (size_t k = 1; k < len && i + k < text.size(); ++k) {
            cp = (cp << 6) | (text[i + k] & 0x3F);
        }
        i += len;
        ++chars;
        if ((cp >= 0x3040 && cp <= 0x30FF) ||   // kana
            (cp >= 0x3400 && cp <= 0x9FFF) ||   // han
            (cp >= 0xAC00 && cp <= 0xD7AF) ||   // hangul
            (cp >= 0xF900 && cp <= 0xFAFF)) {   // han compatibility
            ++dense;
        }
    }
    // One dense codepoint is worth far more audio than one letter, so mixed
    // text is budgeted per script rather than by picking a single constant.
    const double frames = (double) (chars - dense) * 1.4 + (double) dense * 8.0;
    // Floor: very short inputs still need room for lead-in silence and a tail.
    int32_t budget = (int32_t) frames + 128;
    if (budget > hard_cap) budget = hard_cap;
    return budget;
}

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate
    int32_t max_audio_tokens = 2048;
    
    // Temperature for sampling (0 = greedy)
    float temperature = 0.9f;

    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;
    
    // Number of threads
    int32_t n_threads = 4;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Number of RVQ codebooks to predict and decode per frame (1..16).
    // 0 = all. Each dropped codebook removes one code-predictor pass per
    // frame — the main speed/fidelity dial that does not require a different
    // model. Below ~8 the voice audibly coarsens.
    int32_t n_codebooks = 0;

    // Language ID for codec (2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
    int32_t language_id = 2050;

    // Voice steering instruction (e.g. "speak slowly in a calm tone")
    std::string instructions;

    // Reference text for ICL voice cloning (when set, enables ICL mode instead of x-vector)
    std::string ref_text;

    // Sampling seed. < 0 means leave RNG as-is (non-deterministic); >= 0 reseeds
    // the transformer's RNG so runs are reproducible.
    int64_t seed = -1;
};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;
    
    // Sample rate
    int32_t sample_rate = 24000;
    
    // Success flag
    bool success = false;

    // Generation stopped because it hit the per-request frame budget instead
    // of emitting EOS — i.e. the model ran away (docs/known-issues.md #11).
    // The audio that came back is unusable: the tail is noise and the end of
    // the text is missing. Callers that can afford it should retry.
    bool hit_token_budget = false;
    
    // Error message if failed
    std::string error_msg;
    
    // Token counts (real, not approximated)
    int32_t n_text_tokens = 0;      // user-content text tokens (maps to openai usage.input_tokens)
    int32_t n_prefill_tokens = 0;   // total positions the transformer prefilled
                                    // (text + instruct + ref_text + ref_codes + framing)
    int32_t n_audio_tokens = 0;     // codec frames produced by the transformer

    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;        // speaker encoder (voice cloning)
    int64_t t_generate_ms = 0;      // full transformer generate() wall time
    int64_t t_prefill_ms = 0;       // subset of t_generate_ms: build_prefill + forward_prefill
    int64_t t_decode_ms = 0;        // vocoder decode
    int64_t t_total_ms = 0;

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;
    
};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Streaming decode options. When batch_size > 0, the transformer emits
// audio codes in frame batches that are decoded live via the audio
// decoder's streaming path, and each decoded PCM batch is forwarded to
// `on_pcm`. A final flush drains any trailing partial batch. The
// aggregate PCM is also accumulated into `tts_result::audio` for parity
// with the non-streaming path, but consumers that only care about wire
// bytes can ignore it.
struct streaming_opts {
    int32_t batch_size = 0;
    std::function<bool(const float * pcm, size_t n_samples)> on_pcm;
};

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();
    
    // Load all models from directory (auto-detects q8_0 vs f16)
    // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
    bool load_models(const std::string & model_dir);

    // Load models from explicit file paths
    // tts_model_path: path to the TTS GGUF (tokenizer + transformer + encoder)
    // vocoder_model_path: path to the vocoder GGUF (if empty, looks in same directory)
    bool load_model_files(const std::string & tts_model_path,
                          const std::string & vocoder_model_path = "");
    
    // Generate speech from text
    // text: input text to synthesize
    // params: generation parameters
    tts_result synthesize(const std::string & text,
                          const tts_params & params = tts_params());
    
    // Generate speech with voice cloning
    // text: input text to synthesize
    // reference_audio: path to reference audio file (WAV, 24kHz)
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params & params = tts_params());
    
    // Generate speech with voice cloning from samples
    // text: input text to synthesize
    // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
    // n_ref_samples: number of reference samples
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params & params = tts_params());

    // Extract speaker embedding from reference audio file (without synthesis)
    // reference_audio: path to reference audio file (WAV or MP3)
    // embedding: output vector of 1024 float32 values
    bool extract_speaker_embedding(const std::string & reference_audio,
                                    std::vector<float> & embedding);

    // Same, but from already-decoded samples (resampled to 24kHz internally).
    bool extract_speaker_embedding(const float * samples, int32_t n_samples,
                                    int sample_rate, std::vector<float> & embedding);

    // Encode audio to discrete speech codes for ICL voice cloning
    // samples: audio samples (24kHz, mono, normalized to [-1, 1])
    // n_samples: number of samples
    // codes: output vector of codes (n_frames * 16 interleaved)
    // n_frames: output number of frames
    bool encode_speech_codes(const float * samples, int32_t n_samples,
                              std::vector<int32_t> & codes, int32_t & n_frames);

    // Generate speech with pre-extracted speaker embedding
    // text: input text to synthesize
    // embedding: pre-extracted speaker embedding (1024 float32 values)
    // embedding_size: number of elements in embedding (must be 1024)
    // params: generation parameters
    tts_result synthesize_with_embedding(const std::string & text,
                                          const float * embedding, int32_t embedding_size,
                                          const tts_params & params = tts_params(),
                                          const int32_t * ref_codes = nullptr,
                                          int32_t n_ref_frames = 0,
                                          const streaming_opts * stream = nullptr);

    // Streaming overload for non-voice-clone synthesis. See streaming_opts.
    tts_result synthesize(const std::string & text,
                          const tts_params & params,
                          const streaming_opts * stream);

    // Query model info
    int32_t get_hidden_size() const;
    const std::string & get_model_type() const;
    const std::vector<std::string> & get_speaker_names() const;
    const std::vector<int32_t> & get_speaker_ids() const;
    bool has_speaker_encoder() const;

    // Look up speaker token ID by name (-1 if not found)
    int32_t get_speaker_id(const std::string & name) const;

    // Get speaker embedding for a built-in speaker (from codec_embd at speaker token ID)
    bool get_speaker_embedding(const std::string & name, std::vector<float> & embedding);

    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);

    // Set abort callback on all loaded component backends (thread-safe).
    // The callback is stored and automatically re-applied after lazy load/reload.
    void set_abort_callback(ggml_abort_callback callback, void * data);

    // Set CPU thread count for all components. Stored and re-applied to
    // any lazy-loaded component on its first load. Pass 0 to leave ggml's
    // built-in default in effect.
    void set_n_threads(int32_t n_threads);

    // Get error message
    const std::string & get_error() const { return error_msg_; }

    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }
    
private:
    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const tts_params & params,
                                   tts_result & result,
                                   const int32_t * ref_codes = nullptr,
                                   int32_t n_ref_frames = 0,
                                   const streaming_opts * stream = nullptr);

    bool is_aborted() const { return abort_cb_ && abort_cb_(abort_data_); }

    // Prepare the streaming decoder for an ICL request: restore a cached
    // warm-up state for these ref_codes, or decode them once and cache the
    // resulting state. On failure audio_decoder_.get_error() is set.
    bool warmup_decoder_for_icl(const int32_t * ref_codes, int32_t n_ref_frames);


    TextTokenizer tokenizer_;
    TTSTransformer transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioCodecEncoder codec_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    
    bool models_loaded_ = false;
    bool encoder_loaded_ = false;
    bool codec_encoder_loaded_ = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;
    tts_progress_callback_t progress_callback_;
    ggml_abort_callback abort_cb_ = nullptr;
    void * abort_data_ = nullptr;
    int32_t n_threads_ = 0;
};

// Utility: Load audio file (WAV or MP3, dispatched by extension)
bool load_audio_file(const std::string & path, std::vector<float> & samples,
                     int & sample_rate);

// Utility: Load audio from a memory buffer (WAV or MP3, sniffed by magic bytes)
bool load_audio_bytes(const void * data, size_t len,
                      std::vector<float> & samples, int & sample_rate);

// Utility: Save audio file. Dispatches on path extension:
//   .wav         -> 16-bit PCM WAV
//   .mp3         -> LAME VBR -V 4 (speech-tuned, ~70 kbps mono)
//   .opus / .ogg -> Ogg/Opus
// Unknown extensions are rejected with an error message.
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate);

// In-memory audio encoders. samples are float32 mono in range [-1.0, +1.0].
// On encoder failure return an empty string.
std::string encode_wav (const std::vector<float> & samples, int sample_rate);
std::string encode_mp3 (const std::vector<float> & samples, int sample_rate);
std::string encode_opus(const std::vector<float> & samples, int sample_rate);

} // namespace qwen3_tts
