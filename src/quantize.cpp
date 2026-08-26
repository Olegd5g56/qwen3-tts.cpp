// qwen3-tts-quantize - re-quantise a Qwen3-TTS GGUF to a different weight type.
//
// Why this exists: scripts/convert_tts_to_gguf.py can only write the types the
// Python `gguf` package can encode (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/...).
// K-quants live in ggml-quants.c and the package implements them for
// dequantisation only. llama.cpp solves this with llama-quantize, which we
// cannot use: it goes through llama.cpp's model loader and our architecture is
// a custom `qwen3-tts` it does not know. ggml_quantize_chunk() is public ggml
// API though, and ggml is our submodule - so walking the GGUF tensor by tensor
// ourselves is all it takes. See docs/quantisation.md.
//
// Any input type works: each tensor is dequantised to F32 and requantised, so
// bf16 -> q4_K is exact-in, and q8_0 -> q4_K is possible but lossier. Feed it a
// bf16 file when you care.
//
// Usage:
//   qwen3-tts-quantize <in.gguf> <out.gguf> <type> [options]
//     --tensor-type SUBSTR=TYPE   override the type for tensors whose name
//                                 contains SUBSTR (repeatable, last match wins)
//     --threads N                 worker threads (default: hardware_concurrency)
//     --keep-all                  quantise every eligible tensor, including the
//                                 embeddings/heads normally left alone
//     --verify                    dequantise each result and report the error
//                                 against the source, per tensor and overall

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

namespace {

struct override_rule {
    std::string    substr;
    enum ggml_type type;
};

// Mirrors _should_quantize() in scripts/convert_tts_to_gguf.py. Embeddings,
// norms, biases and the output heads stay at the source type - they are the
// tensors where low-bit noise is most audible, and the norms are F32 anyway.
bool should_quantize_name(const std::string & name) {
    if (name.find("_embd")      != std::string::npos) return false;
    if (name.find("codebook")   != std::string::npos) return false;
    if (name.find("_norm")      != std::string::npos) return false;
    if (name.find("lm_head")    != std::string::npos) return false;
    if (name.find("codec_head") != std::string::npos) return false;
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0) return false;
    return true;
}

bool parse_type(const std::string & s, enum ggml_type & out) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const auto t = (enum ggml_type) i;
        const char * n = ggml_type_name(t);
        if (n == nullptr) continue;
        if (s.size() != strlen(n)) continue;
        bool eq = true;
        for (size_t k = 0; k < s.size(); k++) {
            if (tolower((unsigned char) s[k]) != tolower((unsigned char) n[k])) { eq = false; break; }
        }
        if (eq) { out = t; return true; }
    }
    return false;
}

// Dequantise `t` into `dst` (ggml_nelements(t) floats).
void to_f32(const struct ggml_tensor * t, float * dst) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, n * sizeof(float));
        return;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    if (traits->to_float == nullptr) {
        fprintf(stderr, "error: no dequantiser for %s (tensor %s)\n", ggml_type_name(t->type), t->name);
        exit(1);
    }
    traits->to_float(t->data, dst, n);
}

// Quantise nrows x n_per_row floats, splitting the rows across `nthread`.
void quantize_rows(enum ggml_type type, const float * src, void * dst,
                   int64_t nrows, int64_t n_per_row, int nthread) {
    if (nthread <= 1 || nrows < 4) {
        ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, nullptr);
        return;
    }
    const int64_t per = (nrows + nthread - 1) / nthread;
    std::vector<std::thread> workers;
    for (int64_t r0 = 0; r0 < nrows; r0 += per) {
        const int64_t cnt = std::min(per, nrows - r0);
        workers.emplace_back([=]() {
            ggml_quantize_chunk(type, src, dst, r0 * n_per_row, cnt, n_per_row, nullptr);
        });
    }
    for (auto & w : workers) w.join();
}

// How much of the tensor the quantiser threw away, as a fraction of the
// tensor's own scale: rms(quantised - original) / rms(original). Scale-free, so
// it is comparable across tensors and across models, and it is the only ranking
// of weight types available here - this model's autoregressive loop turns any
// numerical difference into a completely different token sequence, so comparing
// generated audio ranks nothing (docs/known-issues.md #16).
double relative_rms_error(enum ggml_type type, const float * src, const void * quantised,
                          int64_t nrows, int64_t n_per_row, std::vector<float> & scratch) {
    const auto * traits = ggml_get_type_traits(type);
    if (traits->to_float == nullptr) return -1.0;

    const int64_t n = nrows * n_per_row;
    scratch.resize((size_t) n);
    traits->to_float(quantised, scratch.data(), n);

    double se = 0.0, s2 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double d = (double) scratch[i] - (double) src[i];
        se += d * d;
        s2 += (double) src[i] * (double) src[i];
    }
    if (s2 == 0.0) return 0.0;
    return sqrt(se / s2);
}

const char * size_str(size_t bytes, char * buf, size_t buflen) {
    snprintf(buf, buflen, "%.2f MiB", bytes / (1024.0 * 1024.0));
    return buf;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <in.gguf> <out.gguf> <type> [--tensor-type SUBSTR=TYPE] [--threads N]\n"
            "                     [--keep-all] [--verify]\n"
            "\n"
            "  <type> is a ggml type name: q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k\n"
            "         iq4_nl iq4_xs f16 bf16 f32 ...\n"
            "\n"
            "example: %s 1.7B-bf16.gguf 1.7B-q4km.gguf q4_k --tensor-type ffn_down=q6_k\n",
            argv[0], argv[0]);
        return 1;
    }

    const char * fname_in  = argv[1];
    const char * fname_out = argv[2];

    enum ggml_type target;
    if (!parse_type(argv[3], target)) {
        fprintf(stderr, "error: unknown type '%s'\n", argv[3]);
        return 1;
    }

    std::vector<override_rule> overrides;
    int  nthread  = (int) std::thread::hardware_concurrency();
    bool keep_all = false;
    bool verify   = false;

    for (int i = 4; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--tensor-type" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t eq = spec.find('=');
            if (eq == std::string::npos) {
                fprintf(stderr, "error: --tensor-type wants SUBSTR=TYPE, got '%s'\n", spec.c_str());
                return 1;
            }
            override_rule r;
            r.substr = spec.substr(0, eq);
            if (!parse_type(spec.substr(eq + 1), r.type)) {
                fprintf(stderr, "error: unknown type in '%s'\n", spec.c_str());
                return 1;
            }
            overrides.push_back(r);
        } else if (arg == "--threads" && i + 1 < argc) {
            nthread = atoi(argv[++i]);
        } else if (arg == "--keep-all") {
            keep_all = true;
        } else if (arg == "--verify") {
            verify = true;
        } else {
            fprintf(stderr, "error: unrecognised argument '%s'\n", arg.c_str());
            return 1;
        }
    }
    if (nthread < 1) nthread = 1;

    if (ggml_quantize_requires_imatrix(target)) {
        fprintf(stderr, "error: %s needs an importance matrix, which this tool does not build\n",
                ggml_type_name(target));
        return 1;
    }

    // no_alloc = false: the whole source file lands in RAM. The largest model
    // here is under 4 GiB, so this buys a lot of simplicity for little cost.
    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };

    struct gguf_context * gguf_in = gguf_init_from_file(fname_in, params);
    if (gguf_in == nullptr) {
        fprintf(stderr, "error: failed to read %s\n", fname_in);
        return 1;
    }

    struct gguf_context * gguf_out = gguf_init_empty();
    gguf_set_kv(gguf_out, gguf_in);

    const int64_t n_tensors = gguf_get_n_tensors(gguf_in);
    printf("%s: %" PRId64 " tensors, target %s, %d threads\n",
           fname_in, n_tensors, ggml_type_name(target), nthread);

    // Quantised payloads have to stay alive until gguf_write_to_file runs.
    std::vector<std::vector<uint8_t>> payloads;
    payloads.reserve(n_tensors);
    std::vector<float> f32;
    std::vector<float> verify_scratch;
    double      err_sum = 0.0, err_worst = 0.0;
    int64_t     err_n = 0;
    const char * err_worst_name = "";

    size_t bytes_in = 0, bytes_out = 0;
    int64_t n_converted = 0, n_skipped_name = 0, n_skipped_shape = 0, n_skipped_dims = 0;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_in, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_data, name);

        gguf_add_tensor(gguf_out, t);
        bytes_in += ggml_nbytes(t);

        enum ggml_type want = target;
        bool named_explicitly = false;
        for (const auto & r : overrides) {
            if (strstr(name, r.substr.c_str()) != nullptr) {
                want = r.type;
                named_explicitly = true;
            }
        }

        const char * skip = nullptr;
        if (ggml_n_dims(t) < 2) {
            skip = "1-D";                      // norms and biases: F32, leave them
            n_skipped_dims++;
        } else if (!keep_all && !named_explicitly && !should_quantize_name(name)) {
            // The name list is a blanket policy; --tensor-type is the operator
            // saying they mean this one. Explicit beats implicit, or there is
            // no way to ask a single kept tensor to be quantised without
            // --keep-all taking every other one with it.
            skip = "kept by name";
            n_skipped_name++;
        } else if (t->ne[0] % ggml_blck_size(want) != 0) {
            // A row that is not a whole number of blocks cannot hold this type.
            // A few speaker-encoder convolutions are shaped that way, and every
            // K-quant needs 256 where Q8_0 needs 32, so this fires far more
            // often for K-quants. Leaving them at the source type is what
            // llama.cpp does too.
            skip = "row not a multiple of the block";
            n_skipped_shape++;
        }

        if (skip != nullptr || want == t->type) {
            bytes_out += ggml_nbytes(t);
            payloads.emplace_back();           // keep indices aligned
            if (skip != nullptr && want != t->type && ggml_n_dims(t) >= 2) {
                printf("  %-48s %-6s -> %-6s  (%s)\n", name,
                       ggml_type_name(t->type), ggml_type_name(t->type), skip);
            }
            continue;
        }

        const int64_t n_per_row = t->ne[0];
        const int64_t nrows     = ggml_nrows(t);

        f32.resize((size_t) nrows * n_per_row);
        to_f32(t, f32.data());

        payloads.emplace_back(ggml_row_size(want, n_per_row) * nrows);
        quantize_rows(want, f32.data(), payloads.back().data(), nrows, n_per_row, nthread);

        gguf_set_tensor_type(gguf_out, name, want);
        gguf_set_tensor_data(gguf_out, name, payloads.back().data());

        bytes_out += payloads.back().size();
        n_converted++;

        if (verify) {
            const double err = relative_rms_error(want, f32.data(), payloads.back().data(),
                                                  nrows, n_per_row, verify_scratch);
            err_sum += err;
            err_n++;
            if (err > err_worst) { err_worst = err; err_worst_name = name; }
            printf("  %-48s %-6s -> %-6s  rel.rms err %.4f%%\n", name,
                   ggml_type_name(t->type), ggml_type_name(want), 100.0 * err);
        }
    }

    char b1[32], b2[32];
    printf("\nconverted %" PRId64 " tensors; left alone %" PRId64 " 1-D, %" PRId64 " by name, %" PRId64 " by shape\n",
           n_converted, n_skipped_dims, n_skipped_name, n_skipped_shape);
    printf("tensor bytes: %s -> %s (%.1f%%)\n",
           size_str(bytes_in, b1, sizeof(b1)), size_str(bytes_out, b2, sizeof(b2)),
           100.0 * bytes_out / bytes_in);
    if (verify && err_n > 0) {
        printf("relative rms error: mean %.4f%%, worst %.4f%% (%s)\n",
               100.0 * err_sum / err_n, 100.0 * err_worst, err_worst_name);
    }

    printf("writing %s ...\n", fname_out);
    if (!gguf_write_to_file(gguf_out, fname_out, /*only_meta =*/ false)) {
        fprintf(stderr, "error: failed to write %s\n", fname_out);
        return 1;
    }

    gguf_free(gguf_out);
    gguf_free(gguf_in);
    ggml_free(ctx_data);
    ggml_quantize_free();

    printf("done\n");
    return 0;
}
