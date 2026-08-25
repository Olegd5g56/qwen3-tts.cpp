# Quantisation

**This document's job:** which weight types this model can actually use, what
each costs, and why two of them break. It exists because the answer is not the
usual llama.cpp answer — this model's activations are extreme enough that the
choice of weight type decides whether it works at all, not just how well.

For the bug-by-bug history see `known-issues.md` #16, #18, #21, #22. For where
the time goes see `optimization.md`.

## The one fact everything follows from

The talker's activations are enormous, and the two model sizes put the peak in
different places:

| model | largest SwiGLU | where |
|---|---|---|
| 0.6B | **185587** | `cp_prefill.blk.2` — the code predictor |
| 1.7B | **9495** | `talker.blk.2` — the talker |

F16 tops out at **65504**. Everything below is a consequence of that collision.

Measure it with `QWEN3_TTS_PROBE_NUM=1` (see `src/op_profiler.h`), and measure
it **on both model sizes** — a magnitude from one says nothing about the other.

## What to use

| `--type` | 0.6B size | works | notes |
|---|---|---|---|
| **`q8_0`** | 1.34 GB | everywhere | **the default.** Fastest on both GPUs. |
| `bf16` | 1.84 GB | everywhere with VRAM | bit-identical to the checkpoint |
| `q4_0` | 1.08 GB | everywhere | fastest, smallest, **audibly worse** |
| `f32` | 3.66 GB | everywhere | 2x bytes for nothing; no reason left to use it |
| `f16` | 1.84 GB | **nowhere** | accepted so #16 stays reproducible; warns |

Speed, ms/frame, median of 3, one sentence, seed 42:

| | 0.6B CUDA | 0.6B ROCm | 0.6B CPU 8t | 1.7B CUDA |
|---|---|---|---|---|
| q8_0 | 25.0 | 24.2 | 45.6 | 33.0 |
| q4_0 | **21.3** | — | **27.4** | **28.8** |
| bf16 | 34.4 | 27.0 | — | *OOM on 6 GB* |
| f16 | 37.0¹ | 24.6¹ | — | — |
| f32 | 38.0 | — | — | — |

¹ a proxy file with 477 of 478 tensors F16 — a real one fails on frame 0.

## Why F16 is not a safe default

In ggml the **weight type silently picks the activation type**. `vec_dot_type`
for `GGML_TYPE_F16` is F16 (`ggml-cpu.c:224`), so F16 weights drag the
activations through F16 too — and 185587 becomes inf. Quantised types cannot do
this: their scale is derived from the data per block, so they have no fixed
ceiling. **Q8_0 is safer than F16 here despite carrying less information.**

bf16 sidesteps it entirely: same 2 bytes, F32's exponent range. And the
checkpoints are bf16 on disk, so `--type bf16` copies bits rather than
converting them — verified bit-exact against the safetensors.

## Why 4-bit needs a scale, and where it lives

On CUDA and HIP a Q4_0/Q4_1/Q5_1 matmul packs its activations as `block_q8_1`,
which carries the block's **sum in F16** (`ggml-common.h:263`) so the kernel can
correct for those types' dequant offset. A block covers 32 values, so the real
ceiling on an activation is **65504/32 = 2047**, not 65504.

Q8_0 uses the D4 layout, which has no sum field (`ggml-cuda/mmq.cuh:60`) — that
is the whole reason it is immune. The CPU is immune too: its `vec_dot_type` for
Q4_0 is `block_q8_0`, also sumless.

The fix is in `tts_transformer.cpp`: `ffn_down`'s input is divided by 128 and
the result multiplied back, gated on the weight type by
`needs_q8_1_activation_sum()`. The matmul is linear so the answer is unchanged;
128 is the smallest power of two that clears the worst case.

**It is not free.** `d` is F16 too, so scaling down pushes small blocks toward
subnormals. Frame 0 stays bit-identical; from frame 1 logits drift ~0.5% with
token ranking preserved — the same order as reduction-order noise. Against inf
on frame 0 that is not a real cost, which is why it is gated rather than
unconditional.

## Quality: the part that is not measured

**Q4_0 coarsens fine acoustic detail and Oleg could hear it** — established in
August on the code predictor alone, and never retracted. Everything in the
speed table above is signal statistics (RMS, zero-crossing rate), which only
proves the output is speech-shaped, not that it sounds right.

So: q4_0 is a real speed and size win and a real quality loss, and the loss has
only ever been judged by ear, once, on part of the model. **Anyone proposing to
ship 4-bit has to listen first**, particularly to how a cloned voice holds up
over a long line.

## Adding K-quants, if that is ever wanted

The motivation is exactly the gap above: Q4_K is better quality than Q4_0 at
about the same size, which is what 4-bit here is short of.

**Why `--type q4_k` cannot simply be re-added.** The Python `gguf` package
implements K-quants for *dequantisation only*. It can write:

```
F32 F16 BF16  Q4_0 Q4_1 Q5_0 Q5_1 Q8_0  MXFP4 TQ1_0 TQ2_0
```

and nothing else. In llama.cpp the split is deliberate: `convert_hf_to_gguf.py`
writes F16/BF16/Q8_0 and the C++ `llama-quantize` does the rest, because the
K-quant search lives in `ggml-quants.c`.

**`llama-quantize` will not work on our files** — it goes through llama.cpp's
model loader, which needs a known architecture, and ours is a custom
`qwen3-tts`.

**The path that would work:** `ggml_quantize_chunk()` is public ggml API
(`ggml.h:2861`) and ggml is our submodule, so a small C++ tool can walk our GGUF
tensor by tensor. That is also how per-tensor type mixing would be done — which
is all an "Unsloth Dynamic" UD-Q4_K_XL is: a normal GGUF whose tensors carry
different types, chosen with an importance matrix from a calibration run and
per-tensor overrides. Not a new format. Our `_should_quantize()` is the same
idea by hand, without calibration.

**Watch out:** K-quants hit the same ceiling as #21 and need the same gate.

| type | `q8_1` layout | needs the scale? |
|---|---|---|
| Q4_K, Q5_K, IQ1_S | DS4 | **yes** |
| Q2_K | D2S6 | **yes** |
| Q3_K, Q6_K, IQ2/3/4_* | D4 | no |

`needs_q8_1_activation_sum()` currently lists only Q4_0/Q4_1/Q5_1. Adding a
K-quant without extending it returns inf on frame 0.
