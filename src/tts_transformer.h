#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "coreml_code_predictor.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#ifdef QWEN3_TTS_TIMING
#include <chrono>
#endif

namespace qwen3_tts {

#ifdef QWEN3_TTS_TIMING
struct tts_timing {
    // Prefill phase
    double t_prefill_build_ms = 0;      // build_prefill_graph (embedding lookups, text projection)
    double t_prefill_forward_ms = 0;    // forward_prefill total
    double t_prefill_graph_build_ms = 0;  // build_prefill_forward_graph
    double t_prefill_graph_alloc_ms = 0;  // sched_alloc_graph
    double t_prefill_compute_ms = 0;      // sched_graph_compute
    double t_prefill_data_ms = 0;         // tensor_set + tensor_get + reset

    // Talker forward_step totals (accumulated across all frames)
    double t_talker_forward_ms = 0;       // total time in forward_step()
    double t_talker_graph_build_ms = 0;   // build_step_graph
    double t_talker_graph_alloc_ms = 0;   // sched_alloc_graph
    double t_talker_compute_ms = 0;       // sched_graph_compute
    double t_talker_data_ms = 0;          // tensor_set + tensor_get + reset

    // Code predictor totals (accumulated across all frames)
    double t_code_pred_ms = 0;            // total predict_codes_autoregressive
    double t_code_pred_init_ms = 0;       // init/clear KV cache + CB0 embed lookup
    double t_code_pred_prefill_ms = 0;    // code pred prefill (2-token, per frame)
    double t_code_pred_steps_ms = 0;      // code pred autoregressive steps (14 steps, per frame)
    double t_code_pred_graph_build_ms = 0;  // graph build (prefill + steps combined)
    double t_code_pred_graph_alloc_ms = 0;  // sched_alloc_graph
    double t_code_pred_compute_ms = 0;      // sched_graph_compute
    double t_code_pred_data_ms = 0;         // tensor_set + tensor_get + reset
    double t_code_pred_coreml_ms = 0;       // CoreML predictor compute + I/O

    // Embed lookups in generate() loop
    double t_embed_lookup_ms = 0;

    int32_t n_frames = 0;
    double t_generate_total_ms = 0;
};
#endif

#define QWEN3_TTS_MAX_NODES 16384

// Capacity of a *cached* step graph. The scratch buffer above is shared and
// sized once for the largest graph there is; a cached graph keeps a buffer of
// its own for the whole request, so seventeen of them at the shared size would
// be a hundred megabytes of host memory for nothing. The largest graph that
// gets cached is the talker's step (1235 nodes on the 1.7B); anything above
// this cap simply falls back to the scheduler.
#define QWEN3_TTS_STEP_MAX_NODES 2048

// KV window granularity: per-step attention views are padded up to a multiple
// of this, so the step graph keeps one shape across many consecutive steps
// instead of growing by one row every frame.
#define QWEN3_TTS_KV_STEP 64

// TTS Transformer configuration (Qwen2-based Talker)
struct tts_transformer_config {
    // Model variant: "base", "custom_voice", or "voice_design"
    std::string model_type = "base";
    std::string model_size;  // "0b6", "1b7"

    // Speaker presets (custom_voice models only)
    std::vector<std::string> speaker_names;
    std::vector<int32_t> speaker_ids;
    std::vector<std::string> speaker_dialects;

    // Language map
    std::vector<std::string> language_names;
    std::vector<int32_t> language_ids;

    bool has_speaker_encoder = false;

    // Text embedding
    int32_t text_vocab_size = 151936;
    int32_t text_embd_dim = 2048;

    // Talker transformer
    int32_t hidden_size = 1024;
    int32_t n_layers = 28;
    int32_t n_attention_heads = 16;
    int32_t n_key_value_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;

    // M-RoPE sections [time, freq, channel] = [24, 20, 20]
    int32_t mrope_section[3] = {24, 20, 20};

    // Codec vocabulary
    int32_t codec_vocab_size = 3072;  // talker.codec_embd/codec_head
    int32_t n_codebooks = 16;

    // Code predictor
    int32_t code_pred_layers = 5;
    int32_t code_pred_vocab_size = 2048;  // Per-codebook vocab
    int32_t code_pred_hidden_size = 0;    // 0 = same as hidden_size (0.6B), otherwise separate (1.7B)
    int32_t code_pred_intermediate_size = 0; // 0 = same as intermediate_size

    // Special codec tokens
    int32_t codec_pad_id = 2148;
    int32_t codec_bos_id = 2149;
    int32_t codec_eos_id = 2150;

    int32_t tts_bos_token_id = 151672;
    int32_t tts_eos_token_id = 151673;
    int32_t tts_pad_token_id = 151671;

    int32_t codec_think_id = 2154;
    int32_t codec_nothink_id = 2155;
    int32_t codec_think_bos_id = 2156;
    int32_t codec_think_eos_id = 2157;

    int32_t english_language_id = 2050;
};

// Transformer layer weights
struct transformer_layer {
    struct ggml_tensor * attn_norm = nullptr;
    
    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    
    struct ggml_tensor * ffn_norm = nullptr;
    
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

// TTS Transformer model weights
struct tts_transformer_model {
    tts_transformer_config config;
    
    // Text embedding and projection
    struct ggml_tensor * text_embd = nullptr;      // [text_embd_dim, text_vocab_size]
    struct ggml_tensor * text_proj_fc1 = nullptr;  // [text_embd_dim, text_embd_dim]
    struct ggml_tensor * text_proj_fc1_bias = nullptr;
    struct ggml_tensor * text_proj_fc2 = nullptr;  // [text_embd_dim, hidden_size]
    struct ggml_tensor * text_proj_fc2_bias = nullptr;
    
    // Codec embedding (for autoregressive input)
    struct ggml_tensor * codec_embd = nullptr;     // [hidden_size, codec_vocab_size]
    
    // Talker transformer layers
    std::vector<transformer_layer> layers;
    
    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr;    // [hidden_size]
    
    // Codec head (for first codebook prediction)
    struct ggml_tensor * codec_head = nullptr;     // [hidden_size, codec_vocab_size]
    
     // Code predictor layers
     std::vector<transformer_layer> code_pred_layers;
     
     // Code predictor output norm (final RMS norm before lm_head)
     struct ggml_tensor * code_pred_output_norm = nullptr;  // [hidden_size]
     
     // Code predictor per-codebook embeddings and heads (15 codebooks, 0 uses talker output)
     std::vector<struct ggml_tensor *> code_pred_embd;  // [hidden_size, code_pred_vocab_size] x 15
     std::vector<struct ggml_tensor *> code_pred_head;  // [hidden_size, code_pred_vocab_size] x 15

     // MTP projection (optional, 1.7B only: projects talker hidden to code_pred hidden)
     struct ggml_tensor * mtp_proj_weight = nullptr;  // [code_pred_hidden, talker_hidden]
     struct ggml_tensor * mtp_proj_bias = nullptr;    // [code_pred_hidden]
    
    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;
    
    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
    
    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// KV cache for autoregressive generation
struct tts_kv_cache {
    std::vector<struct ggml_tensor *> k_cache;
    std::vector<struct ggml_tensor *> v_cache;
    
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int32_t n_ctx = 0;
    int32_t n_used = 0;
    int32_t head_dim = 128;
    int32_t n_kv_heads = 8;
    int32_t n_layers = 28;
};

// TTS Transformer state
struct tts_transformer_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    tts_kv_cache cache;           // Talker KV cache (28 layers)
    tts_kv_cache code_pred_cache; // Code predictor KV cache (5 layers)

    int32_t n_threads = 0;        // 0 = leave ggml default, otherwise applied to CPU backend(s)
};

// A graph whose shape does not change from frame to frame, kept built and kept
// allocated for as long as that shape holds.
//
// This exists because ggml_backend_sched holds exactly one allocation. A frame
// pushes seventeen different graphs through it - the talker's step and the
// code predictor's sixteen - so every one of them finds the allocation planned
// for a different graph and re-plans it from scratch. Running with
// GGML_SCHED_DEBUG_REALLOC=1 aborts on the first frame. Giving each graph its
// own allocator and running it straight on the backend takes that work out of
// the per-frame path entirely; nothing about the arithmetic changes.
struct tts_cached_graph {
    std::vector<uint8_t> meta;             // the graph's own metadata buffer
    struct ggml_cgraph * gf     = nullptr; // lives inside meta
    ggml_gallocr_t       galloc = nullptr;
    int32_t              key    = -1;      // shape this was built for; -1 = empty
};

// TTS Transformer class
class TTSTransformer {
public:
    TTSTransformer();
    ~TTSTransformer();
    
    // Load model from GGUF file
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Initialize KV cache
    bool init_kv_cache(int32_t n_ctx);
    
    // Clear KV cache
    void clear_kv_cache();
    
    // Initialize code predictor KV cache (5 layers, max 16 context)
    bool init_code_pred_kv_cache(int32_t n_ctx);
    
    // Clear code predictor KV cache
    void clear_code_pred_kv_cache();
    
    // Forward pass for text tokens (prefill phase)
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size] (optional, can be nullptr)
    // n_past: number of tokens already in KV cache
    // output: hidden states [n_tokens, hidden_size]
    bool forward_text(const int32_t * text_tokens, int32_t n_tokens,
                      const float * speaker_embd, int32_t n_past,
                      std::vector<float> & output);

    bool forward_prefill(const float * prefill_embd, int32_t n_tokens,
                         int32_t n_past, std::vector<float> & output,
                         std::vector<float> * logits_out = nullptr);
    
    // Forward pass for codec tokens (generation phase)
    // codec_token: single codec token for first codebook
    // n_past: number of tokens already in KV cache
    // output: logits for next codec token [codec_vocab_size]
    bool forward_codec(int32_t codec_token, int32_t n_past,
                       std::vector<float> & output);

    bool forward_step(const float * step_embd, int32_t n_past,
                      std::vector<float> & output,
                      std::vector<float> * hidden_out = nullptr);
    
    // Get hidden states from last forward pass (for code predictor)
    bool get_hidden_states(std::vector<float> & hidden) const;
    
    // Run code predictor to get all 16 codebook predictions
    // hidden: hidden states from talker [hidden_size]
    // prev_codes: previous codes for codebooks 1-15 (can be nullptr for first step)
    // output: logits for all 16 codebooks [16, code_pred_vocab_size]
    bool predict_codes(const float * hidden, const int32_t * prev_codes,
                       std::vector<float> & output);
    
    // Run code predictor autoregressively to generate 15 codes (codebooks 1-15)
    // hidden: hidden states from talker [hidden_size]
    // codebook_0_token: the codebook 0 token (used to create 2-token prefill input)
    // output: generated codes for codebooks 1-15 [15]
    // Number of codebooks to predict per frame (1..n_codebooks); <= 0 means all.
    // Each codebook past the first costs one full code-predictor pass, and the
    // chain is residual, so cutting it short is the one lever that reduces
    // per-frame cost without changing the model.
    void set_active_codebooks(int32_t n) { active_codebooks_ = n; }
    int32_t get_active_codebooks() const { return active_codebooks_; }

    bool predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token, 
                                       std::vector<int32_t> & output,
                                       float temperature = 0.9f,
                                       int32_t top_k = 50);
    
    // Generate speech codes autoregressively
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size]
    // max_len: maximum number of frames to generate
    // output: generated speech codes [n_frames, n_codebooks]
    bool generate(const int32_t * text_tokens, int32_t n_tokens,
                  const float * speaker_embd, int32_t max_len,
                  std::vector<int32_t> & output,
                  int32_t language_id = 2050,
                  float repetition_penalty = 1.05f,
                  float temperature = 0.9f,
                  int32_t top_k = 50,
                  const int32_t * instruct_tokens = nullptr,
                  int32_t n_instruct_tokens = 0,
                  const int32_t * ref_text_tokens = nullptr,
                  int32_t n_ref_text_tokens = 0,
                  const int32_t * ref_codes = nullptr,
                  int32_t n_ref_frames = 0);
    
    const tts_transformer_config & get_config() const { return model_.config; }

    // extract a single row from codec_embd for a given token ID
    bool get_codec_embedding(int32_t token_id, std::vector<float> & output);

    const std::string & get_error() const { return error_msg_; }

    // Set abort callback checked before each graph compute (thread-safe)
    void set_abort_callback(ggml_abort_callback callback, void * data);

    // Set the CPU thread count for this component's CPU backend(s). Safe to
    // call before or after load_model() — the value is stored and reapplied
    // whenever the backends are (re)created.
    void set_n_threads(int32_t n_threads);

    // Fired inside generate() after each frame's 16 codebook codes are
    // pushed onto the output vector. The callback receives the frame
    // index and a pointer to the new frame's codes (valid only for the
    // duration of the call). Returning false aborts generation.
    using frame_emit_fn = std::function<bool(int32_t frame_idx,
                                             const int32_t * frame_codes)>;
    void set_frame_callback(frame_emit_fn cb) { frame_cb_ = std::move(cb); }

    // Enable per-stage progress prints inside generate() (prefill + decode loop)
    void set_verbose(bool v) { verbose_ = v; }

    // Reseed the sampling RNG (for reproducible generation).
    void set_seed(uint64_t seed) { rng_.seed(seed); }

    // Check if abort has been requested
    bool is_aborted() const;

    // Stats from the most recent generate() call. Prefill = build_prefill_graph
    // + forward_prefill (everything before the autoregressive loop). Decode
    // timing is the loop itself; wall-clock of generate() = prefill + decode
    // (plus negligible overhead).
    int32_t get_last_n_prefill_tokens() const { return last_n_prefill_tokens_; }
    int64_t get_last_prefill_ms()       const { return last_prefill_ms_; }
    int64_t get_last_decode_ms()        const { return last_decode_ms_; }
    
    // Legacy interface for compatibility
    bool forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                 std::vector<float> & output);
    
    bool forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                            const float * audio_embd, int32_t n_audio,
                            int32_t audio_start_pos, int32_t n_past,
                            std::vector<float> & output);
    
private:
    // The one sampler. Temperature, then top-k, then softmax, then a
    // multinomial draw from rng_; temperature <= 0 means greedy. Called by the
    // talker's loop in generate() and by both code-predictor paths, which used
    // to carry three byte-identical copies of it.
    //
    // What deliberately stays OUTSIDE it: the talker also suppresses the top
    // 1024 codec ids (except end-of-speech) and applies a repetition penalty
    // over the codebook-0 tokens it has already emitted. Those two belong to
    // the talker alone - in the reference implementation they are passed in
    // `talker_kwargs` while `code_predictor.generate()` gets only
    // do_sample/top_k/top_p/temperature - so generate() applies them to the
    // logits before calling this. Do not "fix" that asymmetry by moving them
    // in here; it is the model's, not ours.
    //
    // `probs` is caller-owned scratch, sized >= vocab_size, so the hot loop
    // does not allocate per frame.
    int32_t sample_token(float * logits, int32_t vocab_size,
                         float temperature, int32_t top_k,
                         std::vector<float> & probs);

    bool try_init_coreml_code_predictor(const std::string & model_path);
    bool predict_codes_autoregressive_coreml(const float * hidden, int32_t codebook_0_token,
                                             std::vector<int32_t> & output,
                                             float temperature,
                                             int32_t top_k);

    bool build_prefill_graph(const int32_t * text_tokens, int32_t n_tokens,
                             const float * speaker_embd, int32_t language_id,
                             std::vector<float> & prefill_embd,
                             std::vector<float> & trailing_text_hidden,
                             std::vector<float> & tts_pad_embed,
                             const int32_t * instruct_tokens = nullptr,
                             int32_t n_instruct_tokens = 0,
                             const int32_t * ref_text_tokens = nullptr,
                             int32_t n_ref_text_tokens = 0,
                             const int32_t * ref_codes = nullptr,
                             int32_t n_ref_frames = 0,
                             int32_t * reusable_prefix_len = nullptr);

    // ── Per-voice prefill reuse ──────────────────────────────────────
    //
    // Two parts of an ICL prefill depend on the voice alone and are rebuilt
    // from scratch on every request otherwise:
    //
    //   * the reference block the prompt feeds the model - one summed row per
    //     reference frame over all 16 codebooks, plus the reference
    //     transcript's projection. Pure host-side assembly, and the codebook
    //     sums cost one device read per codebook per frame.
    //   * the KV of the prompt's head: instructions, role tokens, the codec
    //     overlay and the reference transcript.
    //
    // What is deliberately NOT in the second cache: the reference frames.
    // They sit at the *end* of the prompt, after the line being spoken (see
    // build_prefill_graph), so every layer above the first mixes the target
    // text into their K and V. Their KV is different for every request and
    // reusing it would be wrong, not merely stale. Only the head - everything
    // up to and including the reference transcript - is position- and
    // content-stable, which is 46 of 171 tokens on the benchmark voice.
    //
    // Both caches are capped and evicted least-recently-used: a library of
    // 198 voices must not be able to pin a gigabyte. QWEN3_TTS_PREFIX_CACHE
    // sets the cap in voices; 0 disables both.
    struct voice_prefix_entry {
        uint64_t             key = 0;
        std::vector<int32_t> ref_codes;      // re-checked on hit, not just hashed
        std::vector<int32_t> ref_text;       //   "
        std::vector<float>   codec_section;  // (1 + n_ref_frames) * hidden_size
        std::vector<float>   ref_text_proj;  // n_ref_text * hidden_size
    };

    struct prefix_kv_entry {
        uint64_t             key = 0;
        int32_t              n_tokens = 0;
        std::vector<float>   embd;           // re-checked on hit, not just hashed
        std::vector<uint8_t> k;              // n_layers * n_tokens * row bytes
        std::vector<uint8_t> v;
    };

    // Cap in voices, read once from QWEN3_TTS_PREFIX_CACHE.
    int32_t prefix_cache_cap() const;

    // Both return nullptr when the cache is disabled or the entry is absent.
    voice_prefix_entry * voice_prefix_find(uint64_t key,
                                           const int32_t * ref_codes, size_t n_ref_codes,
                                           const int32_t * ref_text,  size_t n_ref_text);
    voice_prefix_entry * voice_prefix_insert(voice_prefix_entry entry);

    // Snapshot / restore KV rows [0, n_tokens) of the talker cache. The token
    // index is the cache tensors' slowest axis, so a prefix is one contiguous
    // byte range per layer and survives a change of n_ctx untouched.
    bool prefix_kv_save(uint64_t key, const float * embd, int32_t n_tokens);
    bool prefix_kv_load(uint64_t key, const float * embd, int32_t n_tokens);

    std::vector<voice_prefix_entry> voice_prefix_cache_;  // front = most recent
    std::vector<prefix_kv_entry>    prefix_kv_cache_;     // front = most recent
    mutable int32_t                 prefix_cache_cap_ = -1;

    struct ggml_cgraph * build_prefill_forward_graph(int32_t n_tokens, int32_t n_past);

    struct ggml_cgraph * build_step_graph(int32_t n_past);

    bool project_text_tokens(const int32_t * text_tokens, int32_t n_tokens,
                             std::vector<float> & output);

    bool lookup_embedding_rows(struct ggml_tensor * embedding, const int32_t * token_ids,
                               int32_t n_tokens, const char * input_name,
                               const char * output_name, std::vector<float> & output);
    bool lookup_single_embedding_row(struct ggml_tensor * embedding, int32_t token_id,
                                     float * out_row);
    
    // Build computation graph for code predictor
    struct ggml_cgraph * build_code_pred_graph(int32_t n_prev_codes);
    
    // Build computation graph for single-step autoregressive code predictor
    // n_past: number of tokens already in KV cache (0-14)
    // generation_step: which codebook we're predicting (0-14)
    struct ggml_cgraph * build_code_pred_step_graph(int32_t n_past, int32_t generation_step);
    
    // Build computation graph for 2-token prefill of code predictor
    // Processes [past_hidden, codec_embd(codebook_0_token)] together
    struct ggml_cgraph * build_code_pred_prefill_graph();

    // Starts a graph in whichever scratch buffer is active - the shared one,
    // or a cache's own while that cache is being filled.
    struct ggml_cgraph * new_graph(struct ggml_context ** ctx0_out);

    // The cached graph for `key`, built on first use and rebuilt when the key
    // changes. nullptr means "not cacheable" - the caller falls back to the
    // scheduler, which is also what happens when reuse is switched off.
    struct ggml_cgraph * cached_graph(tts_cached_graph & cg, int32_t key,
                                      const std::function<struct ggml_cgraph *()> & build);
    static void release_cached_graph(tts_cached_graph & cg);
    void release_cached_graphs();

    
    // Parse hyperparameters from GGUF
    bool parse_config(struct gguf_context * ctx);
    
    // Create tensor structures
    bool create_tensors(struct gguf_context * ctx);
    
    // Load tensor data from file
    bool load_tensor_data(const std::string & path, struct gguf_context * ctx);
    
    tts_transformer_model model_;
    tts_transformer_state state_;
    std::string error_msg_;
    ggml_abort_callback abort_cb_ = nullptr;
    void * abort_data_ = nullptr;
    bool verbose_ = false;
    frame_emit_fn frame_cb_;

    // Stats populated by generate()
    int32_t last_n_prefill_tokens_ = 0;
    int64_t last_prefill_ms_ = 0;
    int64_t last_decode_ms_ = 0;

    // Cached hidden states from last forward pass
    std::vector<float> last_hidden_;
    std::vector<ggml_fp16_t> embd_row_fp16_scratch_;
    std::mt19937 rng_{std::random_device{}()};
    int32_t      active_codebooks_ = 0;  // 0 = all
    CoreMLCodePredictor coreml_code_predictor_;
    bool use_coreml_code_predictor_ = false;
    std::string coreml_code_predictor_path_;
    bool skip_ggml_code_pred_layers_ = false;

    // Kept step graphs. talker_graph_ is keyed by the padded KV window, so it
    // is rebuilt once every QWEN3_TTS_KV_STEP frames; the code predictor's are
    // keyed by nothing at all - their KV window is a fixed 16 - and are built
    // once per model load.
    tts_cached_graph              talker_graph_;
    tts_cached_graph              code_pred_prefill_graph_;
    std::vector<tts_cached_graph> code_pred_step_graphs_;
    bool                          graph_reuse_ = false;
    std::vector<uint8_t> *        active_meta_ = nullptr;  // set only while filling a cache

#ifdef QWEN3_TTS_TIMING
    tts_timing * timing_ = nullptr;
#endif
};

// Free model resources
void free_transformer_model(tts_transformer_model & model);

// Free KV cache resources
void free_tts_kv_cache(tts_kv_cache & cache);

} // namespace qwen3_tts
