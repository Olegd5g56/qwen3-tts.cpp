#include "audio_tokenizer_decoder.h"
#include "test_fixtures.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

namespace {

void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s --tokenizer <path.gguf> [--frames N] [--chunk K] [--seed S] [--tol F]\n", argv0);
}

}

int main(int argc, char ** argv) {
    const char * tokenizer_path = nullptr;
    int n_frames = 64;
    int chunk = 16;
    int seed = 1;
    float tol = 1e-4f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) n_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) tol = (float) atof(argv[++i]);
        else { print_usage(argv[0]); return 1; }
    }
    // Falls back to the shared test env var so ctest can register this without
    // a hardcoded path: this test needs only a vocoder, no reference dumps.
    const std::string tokenizer = tokenizer_path
        ? std::string(tokenizer_path)
        : qwen3_tts_test::model_path_from_env("QWEN3_TTS_TEST_VOCODER", "");
    if (tokenizer.empty()) {
        printf("  SKIP: no vocoder given (--tokenizer or QWEN3_TTS_TEST_VOCODER)\n");
        print_usage(argv[0]);
        return qwen3_tts_test::SKIP_EXIT_CODE;
    }
    if (!qwen3_tts_test::fixtures_present({tokenizer})) {
        return qwen3_tts_test::SKIP_EXIT_CODE;
    }

    qwen3_tts::AudioTokenizerDecoder decoder;
    if (!decoder.load_model(tokenizer)) {
        fprintf(stderr, "load failed: %s\n", decoder.get_error().c_str());
        return 2;
    }
    // Optional: change the CPU thread count. Different thread counts change
    // the order reductions accumulate in, so this measures how much of any
    // difference above is plain floating-point noise.
    if (const char * t = std::getenv("QWEN3_TTS_TEST_THREADS")) {
        decoder.set_n_threads(atoi(t));
        printf("threads: %d\n", atoi(t));
    }

    const auto & cfg = decoder.get_config();

    // random but deterministic codes in [0, codebook_size).
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, cfg.codebook_size - 1);
    std::vector<int32_t> codes((size_t) n_frames * cfg.n_codebooks);
    for (auto & c : codes) c = dist(rng);

    std::vector<float> pcm_oneshot;
    if (!decoder.decode(codes.data(), n_frames, pcm_oneshot)) {
        fprintf(stderr, "one-shot decode failed: %s\n", decoder.get_error().c_str());
        return 3;
    }
    printf("one-shot: %zu samples\n", pcm_oneshot.size());

    // causality check: decode(first half) should equal decode(full)[0:prefix_len]
    // if the decoder is strictly causal. this isolates streaming bugs from
    // non-causal behavior in the underlying decoder.
    size_t prefix_len = 0;
    if (n_frames >= 2) {
        int half = n_frames / 2;
        std::vector<float> pcm_half;
        if (decoder.decode(codes.data(), half, pcm_half)) {
            prefix_len = pcm_half.size();
            size_t cmp_n = pcm_half.size();
            if (cmp_n > pcm_oneshot.size()) cmp_n = pcm_oneshot.size();
            double max_diff = 0.0;
            size_t first_bad = SIZE_MAX;
            for (size_t i = 0; i < cmp_n; ++i) {
                double d = std::fabs((double) pcm_half[i] - (double) pcm_oneshot[i]);
                if (d > max_diff) max_diff = d;
                if (d > tol && first_bad == SIZE_MAX) first_bad = i;
            }
            // Informational only. Two decodes of different lengths run
            // different attention shapes, so the softmax sums its terms in a
            // different order and the last bits differ. The conv tower then
            // amplifies that. See docs/known-issues.md 1.
            printf("length-sensitivity (informational): decode(%d)[0:%zu] vs "
                   "decode(%d)[0:%zu] max_diff=%.3e%s\n",
                   half, cmp_n, n_frames, cmp_n, max_diff,
                   (first_bad != SIZE_MAX) ? "" : " (bit-identical)");
            if (first_bad != SIZE_MAX) {
                printf("  first deviation at %zu: half=%.6f full=%.6f\n",
                       first_bad, pcm_half[first_bad], pcm_oneshot[first_bad]);
            }
        }
    }

    // Causality check B — the discriminating one.
    //
    // Check A above compares decodes of *different* lengths, so the tensor
    // shapes differ and so does the order the backend accumulates its matmuls.
    // A deviation there can be plain floating-point noise rather than a leak.
    //
    // Here the length is identical and only the codes *after* the prefix
    // change. Same shapes, same kernel decomposition, same arithmetic order —
    // so any deviation inside the prefix is the future genuinely leaking into
    // the past, with nothing else it could be.
    if (n_frames >= 2) {
        const int half = n_frames / 2;
        std::vector<int32_t> codes_b = codes;
        std::mt19937 rng_b(seed + 1000);
        for (size_t i = (size_t) half * cfg.n_codebooks; i < codes_b.size(); ++i) {
            codes_b[i] = dist(rng_b);
        }

        std::vector<float> pcm_b;
        if (decoder.decode(codes_b.data(), n_frames, pcm_b)) {
            // Exactly the samples the unchanged prefix is responsible for:
            // the length decode(half) produces on its own.
            size_t cmp_n = prefix_len ? prefix_len : pcm_oneshot.size() / 2;
            if (cmp_n > pcm_b.size()) cmp_n = pcm_b.size();
            double max_diff = 0.0;
            size_t first_bad = SIZE_MAX;
            for (size_t i = 0; i < cmp_n; ++i) {
                double d = std::fabs((double) pcm_b[i] - (double) pcm_oneshot[i]);
                if (d > max_diff) max_diff = d;
                if (d > tol && first_bad == SIZE_MAX) first_bad = i;
            }
            printf("future-leak: same length, codes[%d:] changed, first %zu samples "
                   "max_diff=%.3e%s\n", half, cmp_n, max_diff,
                   (first_bad != SIZE_MAX) ? " (LEAK)" : " (clean)");
            if (first_bad != SIZE_MAX) {
                printf("  first deviation at %zu: a=%.6f b=%.6f\n",
                       first_bad, pcm_oneshot[first_bad], pcm_b[first_bad]);
            }
        }
    }

    // Repeatability: same input twice. Anything non-zero here means the
    // backend itself is non-deterministic and every number above is noise.
    {
        std::vector<float> pcm_again;
        if (decoder.decode(codes.data(), n_frames, pcm_again) &&
            pcm_again.size() == pcm_oneshot.size()) {
            double max_diff = 0.0;
            for (size_t i = 0; i < pcm_again.size(); ++i) {
                double d = std::fabs((double) pcm_again[i] - (double) pcm_oneshot[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("repeatability: identical input twice max_diff=%.3e\n", max_diff);
        }
    }

    std::vector<float> pcm_stream;
    decoder.stream_reset();
    for (int off = 0; off < n_frames; off += chunk) {
        int k = std::min(chunk, n_frames - off);
        const int32_t * ptr = codes.data() + (size_t) off * cfg.n_codebooks;
        if (!decoder.stream_decode(ptr, k, pcm_stream)) {
            fprintf(stderr, "stream_decode failed at off=%d: %s\n", off, decoder.get_error().c_str());
            return 4;
        }
    }
    printf("streaming: %zu samples (%d chunks of %d frames)\n", pcm_stream.size(),
           (n_frames + chunk - 1) / chunk, chunk);

    if (pcm_stream.size() != pcm_oneshot.size()) {
        fprintf(stderr, "FAIL: length mismatch: stream=%zu oneshot=%zu\n",
                pcm_stream.size(), pcm_oneshot.size());
        return 5;
    }

    double max_abs = 0.0, sum_sq = 0.0, sig_sq = 0.0;
    size_t first_bad = SIZE_MAX;
    for (size_t i = 0; i < pcm_oneshot.size(); ++i) {
        double d = std::fabs((double) pcm_stream[i] - (double) pcm_oneshot[i]);
        if (d > max_abs) max_abs = d;
        sum_sq += d * d;
        sig_sq += (double) pcm_oneshot[i] * (double) pcm_oneshot[i];
        if (d > tol && first_bad == SIZE_MAX) first_bad = i;
    }
    const double rms     = std::sqrt(sum_sq / pcm_oneshot.size());
    const double sig_rms = std::sqrt(sig_sq / pcm_oneshot.size());
    const double rel_db  = (rms > 0.0 && sig_rms > 0.0)
                         ? 20.0 * std::log10(rms / sig_rms) : -999.0;

    // Chunked decode cannot be bit-identical to one-shot: each chunk runs
    // attention over a different number of keys, so the softmax accumulates
    // its terms in a different order. Frame 0 is always exact (its softmax has
    // a single term); everything after inherits last-bit differences that the
    // conv tower amplifies. So the criterion is the error *floor* relative to
    // the signal, not an absolute sample-wise tolerance.
    printf("stream vs one-shot: max_abs=%.3e rms=%.3e signal_rms=%.3e (%.1f dB)\n",
           max_abs, rms, sig_rms, rel_db);
    if (first_bad != SIZE_MAX) {
        printf("first deviation at sample %zu: stream=%.6f oneshot=%.6f\n",
               first_bad, pcm_stream[first_bad], pcm_oneshot[first_bad]);
    }
    // Where the limit comes from: the floor measured here is a steady ~-55 dB
    // across 64..512 frames, on *random* codes, which drive the vocoder well
    // off-distribution and amplify more than speech does (real utterances
    // measure around -67 dB). A genuine state-threading bug shows up as a seam
    // — a localised burst tens of dB louder — so -45 dB separates the two with
    // room to spare rather than tracking the floor.
    const double rel_db_limit = -45.0;
    if (rel_db > rel_db_limit) {
        fprintf(stderr, "FAIL: streaming error %.1f dB below signal, limit %.1f dB\n",
                rel_db, rel_db_limit);
        return 6;
    }
    // A single click can hide under a healthy rms. Gate the worst sample
    // against the signal peak as well.
    double peak = 0.0;
    for (float v : pcm_oneshot) {
        const double a = std::fabs((double) v);
        if (a > peak) peak = a;
    }
    if (peak > 0.0 && max_abs > 0.05 * peak) {
        fprintf(stderr, "FAIL: worst sample %.3e is %.1f%% of peak %.3e\n",
                max_abs, 100.0 * max_abs / peak, peak);
        return 7;
    }

    printf("PASS\n");
    return 0;
}
