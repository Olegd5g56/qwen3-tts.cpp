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

| `--type` | works | notes |
|---|---|---|
| **`q8_0`** | everywhere | **the default.** Cheapest to be sure about. |
| `q4_k` | everywhere | same bytes as `q4_0`, less error - see below |
| `q5_k` `q6_k` | everywhere | the middle of the ladder |
| `bf16` | everywhere with VRAM | bit-identical to the checkpoint |
| `q4_0` | everywhere | **superseded by `q4_k`**, kept only for old files |
| `f32` | everywhere | 2x bytes for nothing; no reason left to use it |
| `f16` | **nowhere** | accepted so #16 stays reproducible; warns |

K-quants need `qwen3-tts-quantize` (see below); the Python converter cannot
write them.

## How much each one throws away

`qwen3-tts-quantize --verify` reports `rms(dequantised - original) / rms(original)`
per tensor and averaged. It is scale-free, so it compares across tensors, types
and model sizes - and it is the **only** ranking of weight types available on
this model, for the reason in the next section.

| type | bits/weight | rel. rms error | 0.6B | 1.7B |
|---|---|---|---|---|
| `q8_0` | 8.5 | **0.55%** | 1281 MiB | 2345 MiB |
| `q6_k` | 6.56 | 1.81% | 1159 MiB | 1999 MiB |
| `q5_k` | 5.5 | 3.68% | 1093 MiB | 1809 MiB |
| `q4_k` | 4.5 | 7.26% | 1030 MiB | 1630 MiB |
| `q4_0` | 4.5 | 8.82% | 1030 MiB | 1630 MiB |

Measured 2026-08-25 against the bf16 files. The two model sizes agree to the
third decimal, so this is a property of the format, not of the model.

Two things fall out of it:

**`q4_0` has no reason to exist any more.** Q4_K is the same 4.5 bits per
weight - the two files match to the byte - for 17% less error, and it is no
slower anywhere and faster on the CPU.

**A better quant is nearly free in bytes here.** 43% of the file is embeddings
and heads that are never quantised: of the 0.6B's 1745 MiB only 1002 MiB is
eligible. So `q4_k` -> `q6_k` costs 129 MiB (+12%) and cuts the error fourfold.
The usual llama.cpp instinct - squeeze to 4 bits, the file is mostly weights -
pays much worse on this model than it does on an LLM.

## Why generated audio cannot rank any of this

It is tempting to quantise two ways, synthesise the same sentence, and listen.
That ranks nothing. The talker is autoregressive, so any numerical difference
at all changes one sampled token and every token after it differs. Under greedy
decoding, one sentence on the 0.6B came out:

| bf16 | q8_0 | q6_k | q5_k | q4km | q4_k | q4_0 |
|---|---|---|---|---|---|---|
| 107 | 106 | **189** | **189** | 109 | 100 | **52** frames |

`q6_k` is a far better quant than `q4_0` and ran nearly four times longer.
That spread is chaos, not quality. The same applies with sampling on: a
listening test compares one draw from each type, not the types.

So: `--verify` for ranking, ears for whether the winner is good enough, and
never a frame count or a waveform diff.

## Speed, and why the answer depends on the backend

Generation only, ms/frame, median of 3, same sentence and seed. Every build
`GGML_NATIVE=ON` — a build directory configured after `063470c` defaults to OFF,
and comparing two directories that disagree measures the flag, not the backend.

| 0.6B | CUDA (1660S) | ROCm (6800 XT) | CPU 8t |
|---|---|---|---|
| `q8_0` | 23.1 | 18.2 | 39.0 |
| `q6_k` | 21.8 | 18.5 | 29.4 |
| `q5_k` | 21.1 | 19.6 | 24.9 |
| `q4_k` | **20.3** | 19.3 | **20.2** |
| `q4_0` | 20.3 | **18.5** | 21.3 |
| `bf16` | 33.3 | 24.0 | 75.8 |

| 1.7B | CUDA (1660S) | ROCm (6800 XT) | CPU 8t |
|---|---|---|---|
| `q8_0` | 27.1 | 21.0 | 65.0 |
| `q6_k` | 25.7 | 20.7 | 50.8 |
| `q5_k` | 23.7 | 21.5 | 42.0 |
| `q4_k` | **23.0** | 20.6 | **34.1** |
| `q4_0` | 23.3 | **20.2** | 35.9 |
| `bf16` | 39.5 | 28.9 | 124.8 |

**On the CPU the weight type is the single biggest lever there is** — `q8_0` to
`q4_k` is nearly 2x on the 1.7B. On NVIDIA it is worth ~15%. On AMD it is worth
nothing at all: every quantised type lands within 1.4 ms of every other. Do not
carry a number from one backend to another.

`bf16` is the outlier everywhere and by a lot — 1.5x the cost of `q8_0` on the
GPUs and nearly 2x on the CPU. It is for checking numerics against the
checkpoint, not for running.

### These are not whole-request numbers

Generation is the part the weight type governs, which is why the table measures
it. It is not what a request costs, and it is **not comparable across backends
in general**: the vocoder is overlapped into generation on CUDA, ROCm and Metal
but not on Vulkan or CPU (`pipeline_supported()` in `qwen3_tts.cpp`), so the
same column means different things on either side of that line.

For whole-request figures see `optimization.md`. The short version, warm server,
1.7B Q8_0, a 100-word line: ROCm 6800 XT 11.7 s, Vulkan 6800 XT 14.5 s, CUDA
1660 SUPER 17.5 s, Vulkan 1660 SUPER 26.9 s.

## Making one

The Python converter writes bf16; `qwen3-tts-quantize` does the rest.

```
scripts/convert_tts_to_gguf.py --type bf16 <hf-dir> model-bf16.gguf
qwen3-tts-quantize model-bf16.gguf model-q4_k.gguf q4_k --verify
```

Any input type works - each tensor is dequantised to F32 and requantised - but
feed it bf16 when you care, since that is a lossless start.

`--tensor-type SUBSTR=TYPE` overrides by name, which is the whole mechanism
behind an "Unsloth Dynamic" UD-Q4_K_XL: a normal GGUF whose tensors carry
different types. Theirs are chosen with an importance matrix from a calibration
run; ours by hand.

```
qwen3-tts-quantize in.gguf out.gguf q4_k --tensor-type ffn_down=q6_k --tensor-type attn_v=q6_k
```

It leaves the same tensors alone the converter does (embeddings, norms, biases,
heads), plus any row that is not a whole number of blocks - 38 speaker-encoder
convolutions, which K-quants skip far more often than Q8_0 because they need
rows divisible by 256 rather than 32.

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

On CUDA and HIP a matmul whose weights use the DS4 or D2S6 layout packs its
activations as `block_q8_1`, which carries the block's **sum in F16**
(`ggml-common.h:263`) so the kernel can correct for those types' dequant offset.
The sum is over 32 values, so in the worst case an activation of 2047 already
overflows it — though what actually trips it here is simpler: the 0.6B's 185587
clears 65504 on its own, no summing needed.

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

## Which types need the scale

`needs_q8_1_activation_sum()` in `tts_transformer.cpp` is a copy of
`mmq_get_q8_1_ds_layout()` in `ggml-cuda/mmq.cuh`. Keep it that way.

| type | `q8_1` layout | needs the scale? |
|---|---|---|
| Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, IQ1_S | DS4 | **yes** |
| Q2_K | D2S6 | **yes** |
| Q8_0, Q5_0, Q3_K, Q6_K, IQ2/3/4_*, MXFP4 | D4 | no |

**Q4_K survives without the scale today, and the gate stays on anyway.** Tested
2026-08-25 by disabling it: Q4_0 fails on frame 0 as expected, Q4_K does not, on
either model size and on a 157-token prefill. The reason is that CUDA has two
kernels and only one of them reads the field — `vec_dot_q4_K_q8_1` (the vector
path, `vecdotq.cuh:946`) takes `d` alone and rebuilds the sum in F32, while
`vec_dot_q4_K_q8_1_impl_mmq` (`mmq.cuh:552`) reads the F16 `s`. Which one runs
depends on batch size, and `ggml_cuda_mul_mat` tries MMVQ before MMQ
(`ggml-cuda.cu:1856`), so the one matmul that carries 185587 — the code
predictor's, at `ne11 = 2` — happens to take the safe path.

That is a property of today's ggml, not of the model. A submodule bump, a
different batch shape, or a wider code-predictor prefill flips it, and the
symptom is inf on frame 0 with nothing else to go on. 0.5% logit drift is the
right price for not having that mine in the tree.

## Quality

`--verify` ranks the types (see the error table above). Whether the winner is
*good enough* is an ear question, and the ear test has a trap in it: every type
produces a different token sequence, so a side-by-side comparison is comparing
one draw against another. Judge across several seeds, or the seed is what you
are hearing.

### Voice cloning is not an axis to choose on

Measured 2026-08-25 the objective way: synthesise with a cloned voice, run the
output back through the speaker encoder, and cosine it against the reference
voice's own embedding. Score every candidate with the **bf16** model or the
quant grades its own homework. The speaker encoder is bf16 in every file anyway
— all 38 of its convolutions have rows that are not a whole number of blocks,
at any block size — so that part of the pipeline is identical throughout.

Two voices, 1.7B, same sentence, 6-14 seeds each:

| type | `bolshshalskiy` | `ostro` |
|---|---|---|
| bf16 | 0.9938 | 0.9914 |
| q8_0 | 0.9934 | 0.9909 |
| q6_k | 0.9931 | **0.9916** |
| q5_k | 0.9924 | 0.9906 |
| q4_k | 0.9918 | 0.9905 |
| q4_0 | 0.9924 | 0.9909 |

The first voice gives a clean ladder that matches the weight-error table. The
second does not reproduce it — `q6_k` scores above `bf16` and `q4_0` ties
`q8_0`. Read one voice alone and you would ship a conclusion that is not there.

What settles it is the scale, which is easy to forget to measure:

| | cosine |
|---|---|
| two **different people** (`bolshshalskiy` vs `ostro` references) | **0.9446** |
| any correctly cloned voice | 0.990 - 0.994 |
| two seeds of the *same* type, to each other | 0.9956 |

So the usable range is 0.055 wide and every weight type lands inside the top
0.001 of it — 2%, and less than the seed-to-seed spread within one type. The
t-test still calls `q4_k` significantly below `bf16`; it is a significant 0.1%.

**Conclusion: the weight type does not change whose voice it is.** Whatever
4-bit costs, it is not speaker identity, and a speaker-similarity number will
not find it - the embedding is invariant to prosody, articulation and noise
floor, which is where quantisation damage would live. Choose on size, on speed
for the backend that will run it, and on ears.

### Recorded ear verdicts, most recent first

- **2026-08-25, 1.7B, cloned Russian voice, one seed each.** Oleg's read:
  heavily seed-dependent; on that draw `q4_k` came out best of the six, with
  possibly weaker cloning fidelity than `q8_0`/`bf16` but correct stress
  placement throughout. The cloning half of that was then measured and is below
  the resolution of the metric (above).
- **August 2026, code predictor only.** `q4_0` coarsens fine acoustic detail
  and Oleg could hear it. Never retracted, and `q4_0` has the worst weight
  error of any type here.
