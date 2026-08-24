# ggml notes

Things found in ggml itself while working on this fork — upstream bugs, missing
kernels, and defaults worth knowing. Kept separately from `known-issues.md`
because none of it can be fixed here: it is either a patch to send upstream or
context for the next person who wonders why a backend behaves oddly.

Measured against **ggml 0.20.2** (commit `8c63e70`) on a GTX 1660 SUPER
(Turing TU116, no tensor cores) and an RX 6800 XT (RADV, gfx1030).

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
