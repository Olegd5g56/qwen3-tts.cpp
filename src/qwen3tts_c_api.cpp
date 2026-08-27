/* qwen3tts_c_api.cpp — C ABI over qwen3_tts::Qwen3TTS.
 *
 * The struct definitions live in the header and are included from here rather
 * than restated, so the two cannot drift apart.
 *
 * Synthesis calls use @autoreleasepool on macOS to drain Metal Objective-C
 * objects when called from background threads. */

#include "qwen3_tts.h"
#include "qwen3tts_c_api.h"

#ifdef __APPLE__
#include <objc/objc.h>
#include <objc/message.h>
// Minimal autorelease pool without importing Foundation
static void * new_autorelease_pool() {
    id pool = ((id(*)(id, SEL))objc_msgSend)(
        (id)objc_getClass("NSAutoreleasePool"),
        sel_registerName("new"));
    return (void *)pool;
}
static void drain_autorelease_pool(void * pool) {
    ((void(*)(id, SEL))objc_msgSend)((id)pool, sel_registerName("drain"));
}
#define AUTORELEASE_BEGIN void * _pool = new_autorelease_pool();
#define AUTORELEASE_END   drain_autorelease_pool(_pool);
#else
#define AUTORELEASE_BEGIN
#define AUTORELEASE_END
#endif

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

// Opaque handle — backs the C typedef
struct Qwen3Tts {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
};

// A prepared voice. ref_codes is empty when the voice was built without a
// transcript: then it is an embedding-only likeness, not ICL.
struct Qwen3TtsVoice {
    std::vector<float>   embedding;
    std::vector<int32_t> ref_codes;
    int32_t              n_ref_frames = 0;
    std::string          ref_text;
};

// A struct whose struct_size we do not recognise was compiled against a
// different version of this header. Reading it would mean reading fields that
// may not be there, so every entry point refuses instead. NULL params are
// still fine and mean "defaults".
static bool params_ok(Qwen3Tts * tts, const Qwen3TtsParams * p) {
    if (!p) return true;
    if (p->struct_size == (int32_t)sizeof(Qwen3TtsParams)) return true;
    if (tts) {
        tts->last_error = "Qwen3TtsParams.struct_size is " + std::to_string(p->struct_size) +
                          ", expected " + std::to_string(sizeof(Qwen3TtsParams)) +
                          " - the caller was built against a different qwen3tts_c_api.h. "
                          "Call qwen3_tts_default_params() before filling it in.";
    }
    return false;
}

static bool request_ok(Qwen3Tts * tts, const Qwen3TtsRequest * r) {
    if (!r) return false;
    if (r->struct_size == (int32_t)sizeof(Qwen3TtsRequest)) return true;
    if (tts) {
        tts->last_error = "Qwen3TtsRequest.struct_size is " + std::to_string(r->struct_size) +
                          ", expected " + std::to_string(sizeof(Qwen3TtsRequest)) +
                          " - the caller was built against a different qwen3tts_c_api.h. "
                          "Call qwen3_tts_default_request() before filling it in.";
    }
    return false;
}

// Helper: convert C params to C++ params. Assumes params_ok() passed.
static qwen3_tts::tts_params to_cpp_params(const Qwen3TtsParams * p) {
    qwen3_tts::tts_params params;
    params.max_audio_tokens = qwen3_tts::MAX_AUDIO_TOKENS;
    if (p) {
        params.max_audio_tokens   = p->max_audio_tokens;
        params.temperature        = p->temperature;
        // p->top_p intentionally unused: the sampler is temperature/top-k/
        // repetition-penalty only; the field stays in the struct for ABI.
        params.top_k              = p->top_k;
        params.n_threads          = p->n_threads;
        params.repetition_penalty = p->repetition_penalty;
        params.language_id        = p->language_id;
        params.n_codebooks        = p->n_codebooks;
        params.seed               = p->seed;
        if (p->instructions) params.instructions = p->instructions;
    }
    return params;
}

// Helper: convert C++ result to heap-allocated C audio struct
static Qwen3TtsAudio * to_c_audio(const qwen3_tts::tts_result & result) {
    if (!result.success || result.audio.empty()) {
        return nullptr;
    }
    auto * out = new Qwen3TtsAudio;
    auto * buf = new float[result.audio.size()];
    std::memcpy(buf, result.audio.data(), result.audio.size() * sizeof(float));
    out->samples     = buf;
    out->n_samples   = (int32_t)result.audio.size();
    out->sample_rate = result.sample_rate;
    return out;
}

// Shared body of voice_from_file / voice_from_samples: 24 kHz samples plus an
// optional transcript in, a prepared voice out. Mirrors what the server's
// VoiceStore does, deliberately - a voice prepared here and the same voice
// prepared there must clone identically.
static Qwen3TtsVoice * build_voice(Qwen3Tts * tts,
                                   std::vector<float> & samples_24k,
                                   const std::vector<float> & embedding,
                                   const char * ref_text) {
    auto * voice = new Qwen3TtsVoice;
    voice->embedding = embedding;
    if (ref_text && ref_text[0]) {
        voice->ref_text = ref_text;
        if (!tts->engine.encode_speech_codes(samples_24k.data(), (int32_t)samples_24k.size(),
                                             voice->ref_codes, voice->n_ref_frames)) {
            tts->last_error = "encode_speech_codes failed: " + tts->engine.get_error();
            delete voice;
            return nullptr;
        }
    }
    return voice;
}

// ============================================================
// C API implementation
// ============================================================

extern "C" {

void qwen3_tts_default_params(Qwen3TtsParams * params) {
    if (!params) return;
    params->struct_size        = (int32_t)sizeof(Qwen3TtsParams);
    // The same hard cap both front ends use. It is an upper bound only: the
    // core estimates a tighter per-request budget from the text length.
    params->max_audio_tokens   = qwen3_tts::MAX_AUDIO_TOKENS;
    params->temperature        = 0.9f;
    params->top_p              = 1.0f;
    params->top_k              = 50;
    params->n_threads          = 4;
    params->repetition_penalty = 1.05f;
    params->language_id        = 2050; // en
    params->n_codebooks        = 0;    // all 16
    params->seed               = -1;   // non-deterministic
    params->instructions       = nullptr;
}

void qwen3_tts_default_request(Qwen3TtsRequest * request) {
    if (!request) return;
    request->struct_size       = (int32_t)sizeof(Qwen3TtsRequest);
    request->text              = nullptr;
    request->voice             = nullptr;
    request->embedding         = nullptr;
    request->embedding_size    = 0;
    request->stream_batch_size = 0;
    request->on_pcm            = nullptr;
    request->on_pcm_user       = nullptr;
}

Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads) {
    if (!model_dir) return nullptr;
    auto * tts = new Qwen3Tts;
    if (!tts->engine.load_models(model_dir)) {
        delete tts;
        return nullptr;
    }
    if (n_threads > 0) {
        tts->engine.set_n_threads(n_threads);
    }
    return tts;
}

Qwen3Tts * qwen3_tts_create_files(const char * model_path, const char * vocoder_path,
                                  int32_t n_threads) {
    if (!model_path) return nullptr;
    auto * tts = new Qwen3Tts;
    if (!tts->engine.load_model_files(model_path, vocoder_path ? vocoder_path : "")) {
        delete tts;
        return nullptr;
    }
    if (n_threads > 0) {
        tts->engine.set_n_threads(n_threads);
    }
    return tts;
}

int qwen3_tts_is_loaded(const Qwen3Tts * tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

int32_t qwen3_tts_sample_rate(const Qwen3Tts * tts) {
    (void)tts;
    return 24000;
}

void qwen3_tts_free_audio(Qwen3TtsAudio * audio) {
    if (!audio) return;
    delete[] audio->samples;
    delete audio;
}

void qwen3_tts_destroy(Qwen3Tts * tts) {
    delete tts;
}

const char * qwen3_tts_get_error(const Qwen3Tts * tts) {
    if (!tts) return "";
    return tts->last_error.c_str();
}

// ---- voices ------------------------------------------------------------

Qwen3TtsVoice * qwen3_tts_voice_from_file(Qwen3Tts * tts, const char * audio_path,
                                          const char * ref_text) {
    if (!tts || !audio_path) return nullptr;
    AUTORELEASE_BEGIN
    Qwen3TtsVoice * voice = nullptr;
    std::vector<float> embedding;
    if (!tts->engine.extract_speaker_embedding(audio_path, embedding)) {
        tts->last_error = "extract_speaker_embedding failed: " + tts->engine.get_error();
    } else {
        std::vector<float> samples;
        int sr = 0;
        bool ok = true;
        if (ref_text && ref_text[0]) {
            ok = qwen3_tts::load_audio_file(audio_path, samples, sr);
            if (!ok) tts->last_error = std::string("load_audio_file failed: ") + audio_path;
            else     qwen3_tts::resample_to_24k(samples, sr);
        }
        if (ok) voice = build_voice(tts, samples, embedding, ref_text);
    }
    AUTORELEASE_END
    return voice;
}

Qwen3TtsVoice * qwen3_tts_voice_from_samples(Qwen3Tts * tts, const float * samples,
                                             int32_t n_samples, int32_t sample_rate,
                                             const char * ref_text) {
    if (!tts || !samples || n_samples <= 0) return nullptr;
    AUTORELEASE_BEGIN
    Qwen3TtsVoice * voice = nullptr;
    std::vector<float> embedding;
    if (!tts->engine.extract_speaker_embedding(samples, n_samples, sample_rate, embedding)) {
        tts->last_error = "extract_speaker_embedding failed: " + tts->engine.get_error();
    } else {
        std::vector<float> pcm(samples, samples + n_samples);
        qwen3_tts::resample_to_24k(pcm, sample_rate);
        voice = build_voice(tts, pcm, embedding, ref_text);
    }
    AUTORELEASE_END
    return voice;
}

int32_t qwen3_tts_voice_n_ref_frames(const Qwen3TtsVoice * voice) {
    return voice ? voice->n_ref_frames : 0;
}

int32_t qwen3_tts_voice_embedding(const Qwen3TtsVoice * voice, float * out, int32_t max_size) {
    if (!voice) return -1;
    const int32_t n = (int32_t)voice->embedding.size();
    if (!out) return n;
    const int32_t copied = n < max_size ? n : max_size;
    if (copied > 0) std::memcpy(out, voice->embedding.data(), (size_t)copied * sizeof(float));
    return copied;
}

int32_t qwen3_tts_voice_ref_codes(const Qwen3TtsVoice * voice, int32_t * out, int32_t max_size) {
    if (!voice) return -1;
    const int32_t n = (int32_t)voice->ref_codes.size();
    if (!out) return n;
    const int32_t copied = n < max_size ? n : max_size;
    if (copied > 0) std::memcpy(out, voice->ref_codes.data(), (size_t)copied * sizeof(int32_t));
    return copied;
}

Qwen3TtsVoice * qwen3_tts_voice_from_parts(const float * embedding, int32_t embedding_size,
                                           const int32_t * ref_codes, int32_t n_ref_frames,
                                           const char * ref_text) {
    if (!embedding || embedding_size <= 0) return nullptr;
    const bool has_codes = ref_codes && n_ref_frames > 0;
    // Codes are 16 interleaved per frame, and without the transcript they are
    // not usable for ICL - refuse the half-built voice rather than silently
    // cloning worse than the caller expects.
    if (has_codes && !(ref_text && ref_text[0])) return nullptr;
    auto * voice = new Qwen3TtsVoice;
    voice->embedding.assign(embedding, embedding + embedding_size);
    if (has_codes) {
        voice->ref_codes.assign(ref_codes, ref_codes + (size_t)n_ref_frames * 16);
        voice->n_ref_frames = n_ref_frames;
        voice->ref_text     = ref_text;
    }
    return voice;
}

void qwen3_tts_voice_free(Qwen3TtsVoice * voice) {
    delete voice;
}

// ---- synthesis ---------------------------------------------------------

Qwen3TtsAudio * qwen3_tts_synthesize_request(Qwen3Tts * tts,
                                             const Qwen3TtsRequest * request,
                                             const Qwen3TtsParams * params) {
    if (!tts) return nullptr;
    if (!request_ok(tts, request) || !params_ok(tts, params)) return nullptr;
    if (!request->text) {
        tts->last_error = "Qwen3TtsRequest.text is NULL";
        return nullptr;
    }

    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);

    // The transcript rides with the voice, not with the request: it describes
    // the reference clip, and pairing it with the wrong audio is the failure
    // in docs/known-issues.md #23.
    if (request->voice) cpp_params.ref_text = request->voice->ref_text;

    qwen3_tts::streaming_opts stream;
    const bool streaming = request->on_pcm != nullptr;
    if (streaming) {
        stream.batch_size = request->stream_batch_size > 0 ? request->stream_batch_size : 16;
        auto cb   = request->on_pcm;
        void * ud = request->on_pcm_user;
        stream.on_pcm = [cb, ud](const float * pcm, size_t n) -> bool {
            return cb(pcm, (int32_t)n, ud) != 0;
        };
    }
    const qwen3_tts::streaming_opts * sopts = streaming ? &stream : nullptr;

    qwen3_tts::tts_result result;
    if (request->voice) {
        const auto & v = *request->voice;
        result = tts->engine.synthesize_with_embedding(
            request->text, v.embedding.data(), (int32_t)v.embedding.size(), cpp_params,
            v.ref_codes.empty() ? nullptr : v.ref_codes.data(), v.n_ref_frames, sopts);
    } else if (request->embedding && request->embedding_size > 0) {
        result = tts->engine.synthesize_with_embedding(
            request->text, request->embedding, request->embedding_size, cpp_params,
            nullptr, 0, sopts);
    } else {
        result = tts->engine.synthesize(request->text, cpp_params, sopts);
    }

    if (!result.success) {
        tts->last_error = result.error_msg;
    } else if (result.hit_token_budget) {
        // The audio is unusable - the tail is noise and the end of the text is
        // missing (docs/known-issues.md #11). The server retries with a
        // different seed; a library cannot decide that for its caller, so it
        // reports rather than returns rubbish.
        tts->last_error = "the model did not signal end of speech and ran to the frame "
                          "budget - retry with a different seed (docs/known-issues.md #11)";
        AUTORELEASE_END
        return nullptr;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize(
        Qwen3Tts * tts, const char * text,
        const Qwen3TtsParams * params) {
    Qwen3TtsRequest req;
    qwen3_tts_default_request(&req);
    req.text = text;
    return qwen3_tts_synthesize_request(tts, &req, params);
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
        Qwen3Tts * tts, const char * text,
        const char * reference_audio_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !reference_audio_path) return nullptr;
    if (!params_ok(tts, params)) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_voice(text, reference_audio_path, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
        Qwen3Tts * tts, const char * text,
        const float * ref_samples, int32_t n_ref_samples,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_samples || n_ref_samples <= 0) return nullptr;
    if (!params_ok(tts, params)) return nullptr;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    auto result = tts->engine.synthesize_with_voice(text, ref_samples, n_ref_samples, cpp_params);
    if (!result.success) {
        tts->last_error = result.error_msg;
    }
    auto * out = to_c_audio(result);
    AUTORELEASE_END
    return out;
}

int32_t qwen3_tts_extract_embedding_file(
        Qwen3Tts * tts, const char * reference_audio_path,
        float * embedding_out, int32_t max_size) {
    if (!tts || !reference_audio_path || !embedding_out || max_size <= 0) return -1;

    AUTORELEASE_BEGIN
    std::vector<float> embedding;
    if (!tts->engine.extract_speaker_embedding(reference_audio_path, embedding)) {
        tts->last_error = tts->engine.get_error();
        AUTORELEASE_END
        return -1;
    }
    AUTORELEASE_END

    int32_t emb_size = (int32_t)embedding.size();
    if (emb_size > max_size) emb_size = max_size;
    std::memcpy(embedding_out, embedding.data(), emb_size * sizeof(float));
    return emb_size;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params) {
    Qwen3TtsRequest req;
    qwen3_tts_default_request(&req);
    req.text           = text;
    req.embedding      = embedding;
    req.embedding_size = embedding_size;
    return qwen3_tts_synthesize_request(tts, &req, params);
}

} // extern "C"
