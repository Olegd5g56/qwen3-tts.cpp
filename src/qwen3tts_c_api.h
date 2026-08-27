/* qwen3tts_c_api.h — C ABI over the qwen3-tts core, for FFI consumers
 * (Nim, Python/ctypes, Rust, Go, …).
 *
 * ABI CHANGED 2026-08-27. The library was behind the C++ core: no ICL voice
 * cloning, no streaming, and a frame cap that did not match either front end.
 * Catching it up meant growing Qwen3TtsParams, which moves the struct layout,
 * so anything compiled against the old header must be rebuilt. Every struct
 * here now starts with `struct_size` and every entry point checks it, so the
 * next addition is detected instead of read as garbage.
 *
 * The short version of what to call:
 *
 *   plain speech        qwen3_tts_synthesize()
 *   clone a voice       qwen3_tts_voice_from_file() + qwen3_tts_synthesize_request()
 *   audio as it renders  fill Qwen3TtsRequest::on_pcm and call the same
 *
 * qwen3_tts_synthesize_request() is the full entry point; the older
 * qwen3_tts_synthesize* functions are kept and are thin wrappers over it.
 */
#ifndef QWEN3TTS_C_API_H
#define QWEN3TTS_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct Qwen3Tts Qwen3Tts;

/* A prepared voice: the speaker embedding, and — when a reference transcript
 * was supplied — the ICL reference codes that go with it. Built once, reused
 * for every line that voice speaks. */
typedef struct Qwen3TtsVoice Qwen3TtsVoice;

/* Generation parameters. ALWAYS fill this with qwen3_tts_default_params()
 * first: it sets struct_size, which every call validates. */
typedef struct Qwen3TtsParams {
    int32_t struct_size;         /* sizeof(Qwen3TtsParams); set by default_params */
    int32_t max_audio_tokens;    /* hard cap on frames; default 6144, as in the
                                  * CLI and the server. The effective cap is
                                  * estimated per request from the text length,
                                  * so this only bounds it from above. */
    float   temperature;         /* default: 0.9, 0=greedy (unstable, see docs) */
    float   top_p;               /* unused (kept for ABI); sampler is temperature/top_k only */
    int32_t top_k;               /* default: 50, 0=disabled */
    int32_t n_threads;           /* default: 4 */
    float   repetition_penalty;  /* default: 1.05 */
    int32_t language_id;         /* 2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko,
                                  * 2053=de, 2061=fr, 2054=es */
    int32_t n_codebooks;         /* RVQ codebooks per frame, 0 = all 16. Below
                                  * ~12 the text itself degrades - see
                                  * docs/known-issues.md #7. */
    int64_t seed;                /* < 0 = non-deterministic (default); >= 0
                                  * reseeds so a run reproduces exactly */
    const char * instructions;   /* voice steering, e.g. "speak slowly"; NULL = none */
} Qwen3TtsParams;

/* Generated audio result */
typedef struct Qwen3TtsAudio {
    const float * samples; /* PCM float32 mono */
    int32_t n_samples;
    int32_t sample_rate;   /* always 24000 */
} Qwen3TtsAudio;

/* Called with each decoded batch of PCM while synthesis is still running.
 * Return 0 to abort the request, non-zero to continue. `user_data` is passed
 * through untouched. The pointer is only valid for the duration of the call. */
typedef int (*Qwen3TtsPcmCallback)(const float * pcm, int32_t n_samples, void * user_data);

/* One synthesis request. Fill with qwen3_tts_default_request(), then set what
 * you need. Voice selection, in order of precedence:
 *   voice != NULL           -> that prepared voice (ICL when it carries codes)
 *   embedding != NULL       -> a raw speaker embedding
 *   neither                 -> the model's default voice
 * Streaming is on when on_pcm != NULL. */
typedef struct Qwen3TtsRequest {
    int32_t struct_size;             /* sizeof(Qwen3TtsRequest); set by default_request */
    const char * text;               /* required */
    const Qwen3TtsVoice * voice;     /* optional */
    const float * embedding;         /* optional, ignored when voice is set */
    int32_t embedding_size;
    int32_t stream_batch_size;       /* frames per decoded batch; 0 = 16 when
                                      * on_pcm is set. Larger batches decode
                                      * slightly cheaper and start later. */
    Qwen3TtsPcmCallback on_pcm;      /* optional; NULL = no streaming */
    void * on_pcm_user;
} Qwen3TtsRequest;

/* Fill with defaults. Both are required before use — they stamp struct_size. */
void qwen3_tts_default_params(Qwen3TtsParams * params);
void qwen3_tts_default_request(Qwen3TtsRequest * request);

/* Create TTS engine and load models from a directory.
 * Returns NULL on failure. */
Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads);

/* Create from explicit file paths. vocoder_path may be NULL to look for the
 * vocoder next to the model. Returns NULL on failure. */
Qwen3Tts * qwen3_tts_create_files(const char * model_path,
                                  const char * vocoder_path,
                                  int32_t n_threads);

/* Check if models are loaded */
int qwen3_tts_is_loaded(const Qwen3Tts * tts);

/* Get sample rate (always 24000) */
int32_t qwen3_tts_sample_rate(const Qwen3Tts * tts);

/* Destroy TTS engine */
void qwen3_tts_destroy(Qwen3Tts * tts);

/* Free generated audio */
void qwen3_tts_free_audio(Qwen3TtsAudio * audio);

/* Get last error message (or empty string) */
const char * qwen3_tts_get_error(const Qwen3Tts * tts);

/* ---- voices ------------------------------------------------------------ */

/* Prepare a voice from a reference clip (WAV or MP3, any sample rate).
 *
 * ref_text is the transcript of that clip. Pass it whenever you have it: with
 * a transcript the voice clones in ICL mode, which is what actually reproduces
 * a speaker. Without one only the x-vector embedding is used and the result is
 * a much weaker likeness.
 *
 * ref_text MUST match what is said in the audio. A transcript that does not
 * match drives generation to the frame cap and produces noise
 * (docs/known-issues.md #23). Keep the clip under ~15 s; only the first 150
 * frames are used.
 *
 * Returns NULL on failure; the reason is in qwen3_tts_get_error(). */
Qwen3TtsVoice * qwen3_tts_voice_from_file(Qwen3Tts * tts,
                                          const char * audio_path,
                                          const char * ref_text);

/* Same, from samples already in memory. sample_rate may be anything; the
 * samples are resampled to 24 kHz internally. Mono, normalized to [-1, 1]. */
Qwen3TtsVoice * qwen3_tts_voice_from_samples(Qwen3Tts * tts,
                                             const float * samples,
                                             int32_t n_samples,
                                             int32_t sample_rate,
                                             const char * ref_text);

/* Number of ICL reference frames the voice carries (0 when it was built
 * without a transcript, i.e. embedding-only). */
int32_t qwen3_tts_voice_n_ref_frames(const Qwen3TtsVoice * voice);

/* Copy out the speaker embedding. Returns the number of floats written, or -1.
 * Pass out=NULL to query the size. The size is the model's hidden size - 2048
 * for the 1.7B, 1024 for the 0.6B - so never hard-code it, and never feed a
 * voice prepared by one model variant to another. */
int32_t qwen3_tts_voice_embedding(const Qwen3TtsVoice * voice,
                                  float * out, int32_t max_size);

/* Copy out the ICL reference codes (n_ref_frames * 16 interleaved). Returns
 * the number of int32 written, or -1. Pass out=NULL to query the size, which
 * is 0 for a voice built without a transcript. */
int32_t qwen3_tts_voice_ref_codes(const Qwen3TtsVoice * voice,
                                  int32_t * out, int32_t max_size);

/* Rebuild a voice from parts previously copied out, skipping the encoder.
 * Preparing a voice costs a speaker-encoder pass and a codec-encoder pass
 * (~1.4 s here); a caller that serves many lines per voice should do it once
 * and keep the parts, which is exactly what the server's voice cache does.
 *
 * ref_codes/n_ref_frames/ref_text may be NULL/0 for an embedding-only voice,
 * but all three go together: codes without their transcript do not clone.
 * Returns NULL if the parts are inconsistent. */
Qwen3TtsVoice * qwen3_tts_voice_from_parts(const float * embedding, int32_t embedding_size,
                                           const int32_t * ref_codes, int32_t n_ref_frames,
                                           const char * ref_text);

void qwen3_tts_voice_free(Qwen3TtsVoice * voice);

/* ---- synthesis --------------------------------------------------------- */

/* The full entry point: voice cloning, streaming, everything.
 * Returns NULL on failure — including the runaway case, where the model never
 * signalled end of speech and ran to the frame budget: the audio behind that
 * is unusable (noise tail, missing sentences), so it is reported as an error
 * and a retry with a different seed is the fix (docs/known-issues.md #11).
 * Caller must free the result with qwen3_tts_free_audio().
 * With on_pcm set, the complete audio is STILL returned — the callback is an
 * addition, not a replacement, so a caller that only wants the wire bytes can
 * ignore the return value (but must still free it). */
Qwen3TtsAudio * qwen3_tts_synthesize_request(Qwen3Tts * tts,
                                             const Qwen3TtsRequest * request,
                                             const Qwen3TtsParams * params);

/* Synthesize text to audio with the model's default voice.
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio * qwen3_tts_synthesize(
    Qwen3Tts * tts,
    const char * text,
    const Qwen3TtsParams * params);

/* Synthesize with voice cloning from an audio file.
 * Embedding-only: no transcript, so no ICL. For a real likeness use
 * qwen3_tts_voice_from_file() with a transcript and synthesize_request(). */
Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
    Qwen3Tts * tts,
    const char * text,
    const char * reference_audio_path,
    const Qwen3TtsParams * params);

/* Same, from raw 24 kHz mono samples in [-1, 1]. Embedding-only, as above. */
Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
    Qwen3Tts * tts,
    const char * text,
    const float * ref_samples,
    int32_t n_ref_samples,
    const Qwen3TtsParams * params);

/* Extract a speaker embedding from an audio file, for caching.
 * embedding_out: caller-allocated buffer; max_size: its size in floats.
 * Returns the embedding size (the model's hidden size: 2048 for the 1.7B,
 * 1024 for the 0.6B), or -1 on failure.
 *
 * An embedding alone is the weak likeness; for real cloning prepare a voice
 * with a transcript instead. */
int32_t qwen3_tts_extract_embedding_file(
    Qwen3Tts * tts,
    const char * reference_audio_path,
    float * embedding_out,
    int32_t max_size);

/* Synthesize with a pre-computed speaker embedding (skips the encoder). */
Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
    Qwen3Tts * tts,
    const char * text,
    const float * embedding,
    int32_t embedding_size,
    const Qwen3TtsParams * params);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_C_API_H */
