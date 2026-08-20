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

## Missing kernel: direct 1D convolution

`ggml_conv_1d` is still im2col + `mul_mat`, with no direct kernel (unlike
`ggml_conv_2d_direct`). For audio decoders this is the dominant cost and it
leaves most of the GPU unused:

- On the 1660 SUPER the vocoder's convolutions reach roughly **12% of the
  card's FP32 peak**. A `mul_mat` of 192x64000x1344 takes ~57 ms where the
  arithmetic alone should be ~10 ms.
- More telling: the vocoder takes **the same time on both cards** — 19.1 s on
  the RX 6800 XT vs 18.1 s on the 1660 SUPER, despite roughly 4x the compute
  and 1.5x the bandwidth. Whatever the limit is, it is not the GPU.
- im2col also materialises very large intermediates (`[672, 192000]` F16 =
  ~258 MB per call) that a fused direct conv would never write.

This is the single largest unexploited win seen in this project, and it would
benefit every audio/TTS model on ggml, not just this one. It is a ggml-side
project, not something that can be worked around in the caller.

**It is now the *only* target left on this path** (measured 2026-08-20, one run
of `ward.txt`, 1.7B, CUDA, `QWEN3_TTS_PIPELINE=0` so the two stages are timed
separately):

```
generate  13 025 ms   25.1 ms/frame   (talker 10.1 + code predictor 13.1)
decode    27 703 ms   53.4 ms/frame
```

The vocoder costs **twice the whole talker + code predictor**. With the
pipeline on, generation hides entirely behind it — 7.5 s of the "generate"
wall time is the talker blocked on the decode queue. So the code predictor,
long treated as the remaining floor, is already free: making it instantaneous
would take generation from 13.0 s to 6.2 s and change the total by nothing.
Everything below the vocoder's 27.7 s is invisible until that number moves.

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
