# ggml notes

Things found in ggml itself while working on this fork — upstream bugs, missing
kernels, and defaults worth knowing. Kept separately from `known-issues.md`
because none of it lives in this repository's own sources: it is either a patch
carried in our ggml fork, something to send upstream, or context for the next
person who wonders why a backend behaves oddly.

Measured against **ggml 0.20.2** (commit `8c63e70`) on a GTX 1660 SUPER
(Turing TU116, no tensor cores) and an RX 6800 XT (RADV, gfx1030).

---

## Where our ggml patches live

The `ggml` submodule points at **`Olegd5g56/ggml`, branch `qwen3-tts`**, not at
`ggml-org/ggml`. The branch is `8c63e70` (v0.20.2) plus one commit per fix, each
written to stand alone so it can be cherry-picked into an upstream PR later
without dragging the others along.

| commit | what |
|---|---|
| `f0aeec2` | check `ggml_gallocr_reserve_n` return value — `known-issues.md` #17 |

Inside the submodule, `origin` is the fork and `upstream` is `ggml-org/ggml`
with pushing disabled, so a stray `git push upstream` cannot reach the real
project. To take a newer ggml: `git fetch upstream && git merge upstream/master`
on `qwen3-tts`, then move the submodule pointer in the parent repo.

Nothing here has been sent upstream yet; that is a deliberate hold, not an
oversight. See `known-issues.md` #17 for the reasoning.

---

## Worth reporting upstream

### multi-add fusion is pathological on RADV / RDNA2

Chained `ggml_add` over large tensors gets fused, and on RADV the fused path
collapses. This fork's vocoder sums 15 residual codebook projections
(192000 x 96, F32) as an add chain:

| build | RTF, long clip, RX 6800 XT |
|---|---|
| ggml 0.20.2 | 2.203 |
| + `GGML_VK_DISABLE_FUSION=1` | 0.674 |
| + `GGML_VK_DISABLE_MULTI_ADD=1` | 0.649 |

3.3x slowdown from one fusion rule, and disabling only that rule is enough to
beat the pre-fusion ggml. Not reproducible on NVIDIA's Vulkan driver
(6963 ms vs 6901 ms — noise), so it looks RADV-specific.

A report should include the add-chain shape (15 adds, `[192000, 96, 1]` F32)
and note that the same graph is fine on the CUDA/ROCm path.

### `argsort.cu` CCCL version gate is wrong for CUDA 13.3

```c
#if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
#    define STRIDED_ITERATOR_AVAILABLE
```

On CUDA 13.3 this is true while `cuda::make_strided_iterator` is not actually
available, so the CUDA backend fails to compile:

```
error: namespace "cuda" has no member "make_strided_iterator"
```

Present in ggml 0.9.11; fixed by 0.20.2. Recorded because anyone pinning an
older ggml with a current toolkit will hit it.

---

## `ggml_conv_1d` has no direct kernel, and keeps its im2col in F16

`ggml_conv_1d` is not an op. It is a graph-level composition — `ggml_im2col`
followed by `ggml_mul_mat` (`ggml/src/ggml.c:4521`) — so there is no
`GGML_OP_CONV_1D` node to look for and no backend kernel to be missing. Upstream
has `ggml_conv_2d_direct` and `ggml_conv_3d_direct`; a `conv_1d_direct` still
does not exist as of ggml 0.20.2. Two consequences for audio models:

- **im2col materialises very large intermediates.** For this vocoder,
  `[672, 287445]` F16 is ~386 MB per call, written and read back for nothing a
  fused direct conv would need. On a 6 GB card this is the binding constraint,
  not arithmetic: decoding ~150 frames in one graph asks for ~1.5 GiB and simply
  fails to allocate. Chunked decode is not an optimisation here, it is what makes
  the GPU path fit at all.
- **The im2col is F16 unless the weights are BF16**, hardcoded at that call site.
  That makes the matmul's `src0` F16, which is what makes the CUDA/HIP backends
  choose a half-precision accumulator — see `known-issues.md` #14, where it cost
  30 dB. Upstream PR ggml-org/llama.cpp#25323 ("use kernel type for conv_1d
  im2col to support f32") is the fix for this and was still open in August 2026.

**Correction (2026-08-24).** This section previously claimed the convolutions
reached only ~12% of the card's FP32 peak, and cited "the vocoder takes the same
time on both cards, despite 4x the compute" as proof that ggml's conv was the
wall. Both observations were real; the explanation was wrong. The vocoder was
running **on the CPU** — see `known-issues.md` #13 — so of course the two cards
agreed. Any conclusion drawn from those numbers, including "a direct conv1d
kernel is the only remaining target", was drawn from a profile of the wrong
device. Re-profile before planning against this stage again.

A direct conv1d kernel is still a plausible upstream contribution, and the
im2col traffic above is a real argument for it. It is no longer supported by
*this* project's measurements, because this project no longer has a measurement
that isolates it.

---

## `CONV_TRANSPOSE_1D` — the real target, and it is not reaching the GPU

**Correction, 2026-08-25.** The kernel analysis below is sound but it is not
what has been running. `GGML_SCHED_DEBUG=2` shows seven `CONV_TRANS` splits per
graph executing on the **CPU**, on the CUDA build and the HIP build alike:
`supports_op` in `ggml-cuda.cu` accepts this op only when both inputs are F32,
and the vocoder stores its conv weights as F16. So the timings below are CPU
convolutions with a GPU round trip either side, filed under a "CUDA" op because
the profiler times the gap between scheduler callbacks and never asks which
device did the work.

The first fix is therefore not a better kernel — it is making the op eligible at
all, by converting the vocoder's conv weights to F32 at load or by teaching the
kernel to take F16. Only then does the analysis below start to apply. See
`optimization.md`.

Re-profiling after the vocoder reached the GPU (`known-issues.md` #13) put
**52% of decode in `CONV_TRANSPOSE_1D`**, against 1.7% in `IM2COL`. The biggest
single entry costs **71 ms per call** for a `[520, 768]` output — 400k elements,
so this is roughly three orders of magnitude off what the hardware can do.

`ggml/src/ggml-cuda/conv-transpose-1d.cu` shows why. One thread per output
element; each thread loops over every input channel and the entire kernel width,
reading weights and inputs straight from global memory:

```c
for (int c = 0; c < src0_ne2; c++) {
    ...
    for (int k = 0; k < src0_ne0; k++) {
        int input_numer = out_t + p0 - k*d0;
        if (input_numer < 0 || input_numer % s0 != 0) {
            continue;                      // most iterations end here
        }
        ...
        accumulator += src0[kernel_offset + k] * src1[input_offset + input_t];
    }
}
```

No tiling, no shared memory, no reuse of the weights every thread in a block
needs — and the `% s0` test discards roughly `s0-1` of every `s0` iterations, so
the loop trip count is a multiple of the useful work. Neighbouring output
elements read almost the same inputs and the exact same weights, which is what a
tiled kernel exists to exploit.

Unlike the conv1d gap below, this is ordinary optimisation of an existing kernel
rather than adding a missing op, and it benefits every audio decoder on ggml.

---

## Defaults and behaviours worth knowing

- **`GGML_CUDA_GRAPHS` defaults to OFF** in standalone ggml (it is ON in
  llama.cpp): `option(GGML_CUDA_GRAPHS "ggml: use CUDA graphs (llama.cpp only)"
  ${GGML_CUDA_GRAPHS_DEFAULT})` with the default set to OFF. Easy to miss when
  porting build flags from llama.cpp. `GGML_HIP_GRAPHS`, by contrast, is ON.
- **An eval callback disables fusion.** Setting
  `ggml_backend_sched_set_eval_callback` (as this fork's `op_profiler` does)
  makes the scheduler run node by node, so any per-op profile you collect is of
  the *unfused* graph. Useful to know when profile totals disagree with wall
  time — that gap is what fusion was buying.
- **Fusion needs the pattern contiguous in the graph.** The CUDA backend has a
  snake-activation rule (`MUL, SIN, SQR, MUL, ADD`), but it only matches when
  those five nodes are consecutive. Emitting the per-channel `exp`/`scale` for
  the beta term between `SQR` and `MUL` silently prevents the match. Frontends
  should materialise all broadcast operands before starting an activation
  chain.
- **ROCm on deprecated targets is fine.** gfx1030 is no longer officially
  supported by AMD, but Arch's rocBLAS still ships Tensile kernels for it (88
  library files), so GEMM does not fall back to a generic path. HIP performance
  is competitive with Vulkan there, and ahead once you use two streams.

## The weight type silently picks the activation type, and F16 has a ceiling

`ggml_mul_mat` never feeds `src1` to the kernel as it stands. Each `src0` type
declares a `vec_dot_type` (`ggml/src/ggml-cpu/ggml-cpu.c`, the type traits
table) and `src1` is converted to it first:

| `src0` (weights) | `vec_dot_type` | what happens to the activations |
|---|---|---|
| F32 | F32 | nothing |
| F16 | **F16** | **hard cast — anything above 65504 becomes inf** |
| Q8_0 | Q8_0 | block-scaled int8; the per-block scale absorbs any magnitude |

The surprising part is the ordering of robustness: **Q8_0 weights are safer
than F16 weights** on a model with large activations, even though Q8_0 carries
less information per value. Quantised types cannot overflow, because their
scale is derived from the data; F16 has a fixed exponent range and cannot
adapt.

This is not a bug, but it is a trap, and it is invisible: nothing warns, the
matmul just returns inf. Any model with activations above 65504 — which
includes the "massive activations" that modern LLM/TTS transformers are known
for — is unusable in plain F16 while working fine in Q8_0. Qwen3-TTS's code
predictor hits 185587 and does exactly this; see `known-issues.md` #16.

BF16 sidesteps it entirely (F32's exponent range in 2 bytes) and is a
first-class type on both the CPU and CUDA backends.

## `block_q8_1` keeps its block sum in F16, and that is a second hidden ceiling

Separate from the weight-type trap above, and easy to miss because it depends on
the backend.

When `src0` is Q4_0, Q4_1 or Q5_1, the CUDA/HIP matmul packs the activations as
`block_q8_1` (`ggml-common.h:258`):

```c
typedef struct {
    ggml_half d;       // delta
    ggml_half s;       // d * sum(qs[i])   <-- F16
    int8_t qs[QK8_1];  // 32 quants
} block_q8_1;
```

`s` exists so the kernel can correct for those types' dequant offset. It is a
**sum over 32 values**, so the effective ceiling on a single activation is
`65504 / 32 = 2047`, not 65504 — two orders of magnitude tighter than the
weight-type trap, and nothing warns.

Which types are affected is decided by `mmq_get_q8_1_ds_layout`
(`ggml-cuda/mmq.cuh:60`): the `DS4` and `D2S6` layouts carry `s`, `D4` does not.

| carries `s` | does not |
|---|---|
| Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, Q2_K, IQ1_S | Q8_0, Q5_0, Q3_K, Q6_K, most IQ, MXFP4 |

Two consequences worth remembering:

- **Q8_0 is immune and Q4_0 is not**, for a reason that has nothing to do with
  bit width.
- **The CPU is immune** — its `vec_dot_type` for Q4_0 is `block_q8_0`, which has
  no sum field. So a model can quantise fine, run fine on CPU, and return inf on
  the GPU. That is not a backend bug; it is two correct implementations with
  different intermediate formats.

## The Python `gguf` package cannot write K-quants

It implements them for dequantisation only. `gguf.quants.quantize` raises a
message-less `NotImplementedError` for every K- and IQ-type; what it can write
is `F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 MXFP4 TQ1_0 TQ2_0`.

This is by design upstream — `convert_hf_to_gguf.py` writes F16/BF16/Q8_0 and
the C++ `llama-quantize` does the rest, because the K-quant search lives in
`ggml-quants.c`. It only bites projects like this one that convert in a single
Python step.

`ggml_quantize_chunk()` (`ggml.h:2861`) is public C API and is the way out for
anyone who needs K-quants without llama.cpp's model loader.
