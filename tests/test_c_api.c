/* test_c_api.c — smoke test for the C ABI (libqwen3tts).
 *
 * Deliberately written in C, not C++: half the point is that the header
 * actually compiles as C, which is what every FFI consumer will do.
 *
 * What it checks:
 *   - a params struct from a different header version is refused, not misread
 *   - a voice prepared from a clip + transcript carries ICL reference frames
 *   - streaming and non-streaming produce the SAME audio, byte for byte
 *   - a voice cached as parts and rebuilt produces the same audio again
 *   - reference codes without their transcript are refused
 *
 * Needs a model: set QWEN3_TTS_TEST_MODEL and QWEN3_TTS_TEST_VOCODER. Without
 * them it exits 77 (ctest "skipped"), like the other model-dependent tests.
 * The reference clip and text are in the repo, so nothing else is needed.
 */
#include "qwen3tts_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SKIP 77

static int n_chunks = 0;
static int32_t n_streamed = 0;

static int on_pcm(const float * pcm, int32_t n, void * user) {
    (void)pcm; (void)user;
    n_chunks++;
    n_streamed += n;
    return 1;
}

/* Reads a file whole. The trailing newline is KEPT: the text is part of the
 * prompt, and dropping a byte changes the token stream and with it the audio. */
static char * read_file(const char * path, long * out_len) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char * buf = (char *) malloc((size_t) n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) { free(buf); fclose(f); return NULL; }
    buf[n] = 0;
    fclose(f);
    if (out_len) *out_len = n;
    return buf;
}

static char * read_transcript(const char * path) {
    long n = 0;
    char * s = read_file(path, &n);
    if (!s) return NULL;
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' ' || s[n-1] == '\t')) {
        s[--n] = 0;
    }
    return s;
}

static int fail(const char * what, Qwen3Tts * tts) {
    printf("FAIL: %s%s%s\n", what,
           tts ? ": " : "", tts ? qwen3_tts_get_error(tts) : "");
    return 1;
}

int main(void) {
    const char * model   = getenv("QWEN3_TTS_TEST_MODEL");
    const char * vocoder = getenv("QWEN3_TTS_TEST_VOCODER");
    if (!model || !vocoder) {
        printf("SKIP: set QWEN3_TTS_TEST_MODEL and QWEN3_TTS_TEST_VOCODER to run\n");
        return SKIP;
    }

    char * text       = read_file("benchmarks/bench_ru_short.txt", NULL);
    char * transcript = read_transcript("benchmarks/voice/sample.txt");
    if (!text || !transcript) {
        printf("SKIP: run from the repo root (benchmarks/ not found)\n");
        return SKIP;
    }

    Qwen3Tts * tts = qwen3_tts_create_files(model, vocoder, 4);
    if (!tts) return fail("create_files", NULL);
    if (!qwen3_tts_is_loaded(tts)) return fail("is_loaded", tts);
    if (qwen3_tts_sample_rate(tts) != 24000) return fail("sample_rate", tts);
    printf("ok   loaded\n");

    Qwen3TtsParams params;
    qwen3_tts_default_params(&params);
    params.seed = 1234;  /* pinned: two runs below must match exactly */

    /* A struct from another build of the header must be rejected outright. */
    Qwen3TtsParams stale = params;
    stale.struct_size = 32;
    if (qwen3_tts_synthesize(tts, text, &stale)) return fail("stale struct_size accepted", tts);
    printf("ok   stale struct_size refused\n");

    /* ICL voice: clip + matching transcript. */
    Qwen3TtsVoice * voice =
        qwen3_tts_voice_from_file(tts, "benchmarks/voice/sample.mp3", transcript);
    if (!voice) return fail("voice_from_file", tts);
    const int32_t n_frames = qwen3_tts_voice_n_ref_frames(voice);
    const int32_t n_emb    = qwen3_tts_voice_embedding(voice, NULL, 0);
    const int32_t n_codes  = qwen3_tts_voice_ref_codes(voice, NULL, 0);
    if (n_frames <= 0)            return fail("voice carries no ICL frames", tts);
    if (n_codes != n_frames * 16) return fail("ref codes are not 16 per frame", tts);
    printf("ok   voice: %d ICL frames, %d embedding floats\n", n_frames, n_emb);

    Qwen3TtsRequest req;
    qwen3_tts_default_request(&req);
    req.text  = text;
    req.voice = voice;
    Qwen3TtsAudio * plain = qwen3_tts_synthesize_request(tts, &req, &params);
    if (!plain) return fail("ICL synthesis", tts);
    printf("ok   ICL synthesis: %d samples\n", plain->n_samples);

    /* Streaming must be the same audio, not merely similar. */
    Qwen3TtsRequest sreq = req;
    sreq.on_pcm            = on_pcm;
    sreq.stream_batch_size = 16;
    Qwen3TtsAudio * streamed = qwen3_tts_synthesize_request(tts, &sreq, &params);
    if (!streamed) return fail("streaming synthesis", tts);
    if (n_chunks < 2) return fail("streaming produced fewer than two batches", tts);
    if (n_streamed != streamed->n_samples) return fail("streamed sample count != returned", tts);
    if (streamed->n_samples != plain->n_samples ||
        memcmp(streamed->samples, plain->samples,
               (size_t) plain->n_samples * sizeof(float)) != 0) {
        return fail("streaming audio differs from non-streaming", tts);
    }
    printf("ok   streaming: %d batches, byte-identical to one-shot\n", n_chunks);

    /* Cache the voice as parts and rebuild it — what a service does. */
    float   * emb   = (float *)   malloc((size_t) n_emb   * sizeof(float));
    int32_t * codes = (int32_t *) malloc((size_t) n_codes * sizeof(int32_t));
    if (!emb || !codes) return fail("out of memory", tts);
    if (qwen3_tts_voice_embedding(voice, emb, n_emb) != n_emb)     return fail("copy embedding", tts);
    if (qwen3_tts_voice_ref_codes(voice, codes, n_codes) != n_codes) return fail("copy ref codes", tts);

    Qwen3TtsVoice * rebuilt =
        qwen3_tts_voice_from_parts(emb, n_emb, codes, n_frames, transcript);
    if (!rebuilt) return fail("voice_from_parts", tts);
    Qwen3TtsRequest rreq;
    qwen3_tts_default_request(&rreq);
    rreq.text  = text;
    rreq.voice = rebuilt;
    Qwen3TtsAudio * again = qwen3_tts_synthesize_request(tts, &rreq, &params);
    if (!again) return fail("synthesis from rebuilt voice", tts);
    if (again->n_samples != plain->n_samples ||
        memcmp(again->samples, plain->samples,
               (size_t) plain->n_samples * sizeof(float)) != 0) {
        return fail("rebuilt voice produced different audio", tts);
    }
    printf("ok   voice_from_parts: byte-identical to the encoded voice\n");

    /* Codes without their transcript do not clone; refuse the half-voice. */
    if (qwen3_tts_voice_from_parts(emb, n_emb, codes, n_frames, NULL)) {
        return fail("parts accepted codes without a transcript", tts);
    }
    printf("ok   codes without a transcript refused\n");

    /* Default voice, no cloning at all. */
    Qwen3TtsAudio * def = qwen3_tts_synthesize(tts, text, &params);
    if (!def) return fail("default-voice synthesis", tts);
    printf("ok   default voice: %d samples\n", def->n_samples);

    qwen3_tts_free_audio(plain);
    qwen3_tts_free_audio(streamed);
    qwen3_tts_free_audio(again);
    qwen3_tts_free_audio(def);
    qwen3_tts_voice_free(voice);
    qwen3_tts_voice_free(rebuilt);
    qwen3_tts_destroy(tts);
    free(emb); free(codes); free(text); free(transcript);

    printf("PASS\n");
    return 0;
}
