# Qwen3-TTS GGML Optimization Report

Performance characterization of this fork. Last updated **2026-08-26**.

> Everything measured before **2026-08-24** described a build whose vocoder was
> silently running on the CPU (`known-issues.md` #13). Sections are ordered so
> current numbers come first; the August-20 sweep is kept at the end as history
> because the relative wins it records are still real.
>
> The newest layer is *The conv_transpose experiment* (2026-08-26), which closes
> that target: rewritten as `mul_mat` + `col2im_1d`, vocoder decode is 2.3-4.5x
> faster on every backend and generation is now the whole cost of a request.

## Current profile (where the time goes)

`ward.txt`, 1.7B, `QWEN3_TTS_PIPELINE=0` so the stages time apart, warm process,
fixed seed. Frame counts differ per backend because sampling does, so compare
ms/frame, not wall time:

Decode covers the 150-frame ICL reference warm-up as well as the generated
frames, in these numbers and in the older ones they are compared against; per
*decoded* frame the vocoder is 10.1 ms on CUDA.

| stage | CUDA, 1660 SUPER | ROCm, RX 6800 XT |
|---|---|---|
| talker forward | 10.1 ms/frame | 7.1 ms/frame |
| code predictor | **13.2 ms/frame** | **10.9 ms/frame** |
| generate (total) | 25.1 ms/frame | 18.9 ms/frame |
| vocoder decode | ~~12.9 ms/frame~~ | ~~8.9 ms/frame~~ |

**The vocoder row is superseded.** `conv_transpose` moved to a GEMM +
`col2im_1d` decomposition on 2026-08-26 and decode is now 4.29 ms/frame on
CUDA/1660S and 1.49 on HIP/6800 XT — see *The conv_transpose experiment*. Those
were measured on `bench_ru.txt` rather than the `ward.txt` this table uses, so
they are not swapped into it; the generate rows are unaffected and still
current. **Generation is now 5-7x the vocoder, not 2x.**
| wall, pipeline off | 20 493 ms (539 fr) | 13 715 ms (494 fr) |
| wall, pipeline on | 18 977 ms | 11 846 ms |

**The ordering has flipped.** The vocoder was 53.4 ms/frame and hid everything;
it is now 12.9 and *generation is roughly twice it*. Prefill is 793 ms for the
1.7B (210 tokens, 150 of them ICL reference frames).

The **code predictor is now the largest single line** — 13.2 ms/frame, more than
the talker. It costs the same on the 0.6B and the 1.7B, because it is 15
sequential 5-layer passes per frame over weights that do not change size with
the talker. It was retired as a target in the previous revision of this file;
that verdict was conditional on a 53 ms/frame vocoder hiding it, and that
condition is gone.

### Inside the vocoder

Per-op, CUDA, `QWEN3_TTS_PROFILE_OPS=1`. The profiler disables fusion, but for
this graph it costs only 3.6% (decode 2472 ms → 2561 ms), so the shares are
trustworthy here — unlike the generate profile, which it inflates by 64%.

Measured twice on different workloads, which is why the shares move a little:

| op | 1.7B | 0.6B, `ward.txt` |
|---|---|---|
| `CONV_TRANSPOSE_1D` | **52%** | **44.8%** |
| `MUL_MAT` (the conv towers) | 11% | 17.9% |
| `ADD` | 2% | 6.7% |
| `IM2COL` | **1.7%** | **4.1%** |

Note what this kills. The whole previous roadmap was built on giving ggml a
direct `conv_1d` kernel to avoid im2col — and im2col is a few per cent of
decode either way. That project is not worth doing for this model.

The new target is `CONV_TRANSPOSE_1D`, and it is a far better-shaped one,
because unlike `conv_1d` it is a real op with a real kernel to improve. The
biggest single entry costs **71 ms per call** for a `[520, 768]` output, which
is absurd for 400k elements. `ggml/src/ggml-cuda/conv-transpose-1d.cu` explains
it: one thread per output element, each looping over every input channel and
the whole kernel width, reading weights and inputs straight from global memory
with no tiling or reuse — and `continue`-ing out of roughly `s0-1` of every
`s0` iterations, so most of the loop is thrown away. See `ggml-notes.md`.

## Against the reference PyTorch pipeline

Measured 2026-08-24. The point of this section is not the headline ratio — it
is that the ratio is the average of a large win and a large loss, and that the
loss has a name.

> **Stale as of 2026-08-27, on the vocoder half.** Two days after this was
> measured the vocoder moved onto the GPU and got 2.3-4.5x faster (`74ff512`,
> `f775bbf`), so the 3.6x loss to cuDNN below is from before that and is
> probably smaller or gone. `tts_transformer.cpp` changed three times since as
> well, so the generate row is not guaranteed either. **The method below is
> what survives** — per frame, bf16 on both sides, warm-up discarded — and it
> is the answer to whether this comparison can be made honestly at all. Re-run
> before quoting a number from the table.

Setup, identical on both sides: 0.6B Base, `ward.txt` (726 chars, Russian),
ICL cloning from the model card's `clone.wav` (8.08 s, 97-101 reference
frames), `QWEN3_TTS_PIPELINE=0`, one warm-up then the median of three runs.
Reference side is `qwen-tts` 0.1.1 + transformers 4.57.3, bf16, `sdpa`.
Frame counts differ between engines because sampling does, so everything is
per frame. Scripts: `scripts/bench_torch_vs_ggml.py` and
`scripts/bench_ggml_cli.py`, which print the same summary.

| | PyTorch, CUDA (1660S) | **ggml, CUDA (1660S)** | PyTorch, ROCm (6800 XT) | **ggml, HIP (6800 XT)** |
|---|---|---|---|---|
| talker | — | 6.4 ms/frame | — | 4.5 ms/frame |
| code predictor | — | 13.0 ms/frame | — | 10.8 ms/frame |
| **generate** | 104.5 ms/frame | **20.4 ms/frame** | 233.9 ms/frame | **15.9 ms/frame** |
| **vocoder decode** | **2.67 ms/frame** | 9.60 ms/frame | 60.1 ms/frame | **6.25 ms/frame** |
| reference encode | 53 ms | 85 ms | 35 ms | 164 ms |
| wall | 60.1 s | **19.1 s** | 175.7 s | **13.6 s** |
| host RSS peak | 2.94 GB | **1.53 GB** | 3.98 GB | — |
| VRAM peak (nvidia-smi) | 5.17 GB | **3.40 GB** | — | — |

### What the numbers say

**We are 3.1x faster end-to-end on CUDA and 13x on ROCm** — but on CUDA that
average hides a 5.1x win on generation and a **3.6x loss on the vocoder**.

**The vocoder loss is cuDNN, not PyTorch.** The same PyTorch code on ROCm
decodes at 60.1 ms/frame — **9.6x slower than our path** on the same card, with
MIOpen falling back to untuned GEMM solvers and warning about it. NVIDIA ships a
decade of hand-tuned transposed-convolution kernels; AMD does not, for these
shapes on gfx1030. So `CONV_TRANSPOSE_1D` remains the right target and 2.67
ms/frame is a **demonstrated ceiling on a 1660 SUPER** rather than a hope.

Note that "our path" here is not ggml's CUDA kernel — see the next section. It
is a CPU convolution with a GPU round trip either side, which makes beating
MIOpen by 9.6x a comment on MIOpen rather than a compliment to us.

**PyTorch also decodes all 653 frames in one shot inside 5.2 GB where we must
chunk.** That looked like a second pointer at the same op — memory-hungry and
slow sharing one cause. It is not; see *Chunk size is not a speed lever* below.
The memory difference is real, the speed inference was wrong.

**The generation win is mostly engine, not quantisation.** The control is now
**bf16**, which is an exact match rather than a handicap: the same 2 bytes as
PyTorch's weights and bit-identical to them (see known-issues #16), so nothing
has to be discounted. It gives 30.3 ms/frame against PyTorch's 104.5. Of our
5.1x: **3.45x is the engine, 1.48x is Q8_0.**

| talker weights | generate | talker | code predictor |
|---|---|---|---|
| Q8_0 | 20.4 ms/frame | 6.4 | 13.0 |
| **bf16** (matched control) | **30.3 ms/frame** | 9.2 | 20.5 |
| F32 | 34.1 ms/frame | 10.2 | 23.3 |
| F16 | *unusable — see known-issues #16* | | |

The earlier F32 control read 3.1x / 1.7x and had to be hedged, because F32
carries twice the weight bytes of the thing it was compared against. bf16
removes the hedge: the engine share is larger than that reading suggested and
the quantisation share smaller.

Decode is unaffected by talker precision (9.60 / 9.49 / 9.44 ms/frame across
the three), which is expected — the vocoder is the same F16 file in all of them
— and is a useful check that the vocoder comparison above is not an artefact.

### Caveats that the table would otherwise hide

- **Neither card has bf16 in hardware** (Turing sm_75, RDNA2 gfx1030), so the
  reference runs emulated. It is still the only usable mode: **fp16 makes the
  model produce NaN logits in the code predictor** and `torch.multinomial`
  dies with a device-side assert. So this is the reference pipeline at its
  practical best on this hardware, not a strawman — but on a bf16-native card
  its generate numbers would improve and ours would not.
- **The vocoder gap is measured in the same bf16**, so it is if anything
  understated by that handicap.
- Two ways to get nonsense out of the reference, both hit while setting this
  up: a reference clip much longer than its transcript makes ICL degenerate
  (generation ran to the 8192-token cap, 655 s of audio), and
  `non_streaming_mode=True` puts the whole text in the prefill and lets the
  model stop after a few frames. Our CLI feeds text one token per frame, so
  the comparable setting is the default `False`.

## Chunk size is not a speed lever — measured, negative

The vocoder never decodes a whole utterance. Every production path — CLI,
server, ICL warm-up, live streaming — goes through `stream_decode` in chunks;
one-shot `decode()` survives only in tests. The premise worth testing was
that chunking is a compromise costing us throughput, and that fixing the
memory blowup would buy speed back. **It does not.**

Medians of three, `ward.txt`, `QWEN3_TTS_PIPELINE=0`, otherwise-idle cards:

| chunk | CUDA (1660S) | VRAM | HIP (6800 XT) |
|---|---|---|---|
| 25 | 13.27 ms/frame | 2254 MiB | — |
| **100** (default) | **9.32** | 3396 MiB | **5.87 ms/frame** |
| 200 | 9.06 | 4902 MiB | 5.93 |
| 400 | out of memory | needs 4089 MiB for the compute buffer alone | 5.76 |
| 713 (one-shot) | out of memory | — | 5.95 |

Above ~100 frames the differences are inside the run-to-run spread — single
runs in both groups came in at 12.3-12.9 ms/frame while their neighbours sat at
9.1. Below it there is a real cliff: 25 frames costs 42%. **So the default of
100 already sits on the plateau, and one-shot decode is worth nothing.** Do not
spend effort making it fit.

The memory figure is nonetheless real: VRAM grows ~15 MiB per frame, so a
713-frame one-shot needs ~12.6 GB, confirming the ~14 GB folklore. It buys
headroom on a small card, not throughput.

Two things did come out of the measurement:

- **known-issues #17** — running out of memory here segfaults rather than
  failing, because `ggml_backend_sched_alloc_splits` discards the return value
  of `ggml_gallocr_reserve_n`. One-line upstream fix, verified.
- `QWEN3_TTS_DECODE_BATCH` was documented as *the* vocoder batch knob but only
  reached the overlapped path and the ICL warm-up; the sequential path had 100
  hardcoded. Both now honour it, keeping their different defaults (16 and 100).

## 0.6B vs 1.7B — not a speed dial

The two models do not produce the same amount of audio for the same text, so
RTF cannot compare them. Per frame, measured 2026-08-20 (CUDA, `ostro`, 5 runs each, means):

| | prefill | talker | code predictor | frames | audio | VRAM |
|---|---|---|---|---|---|---|
| short line, 1.7B | 437 ms | 9.30 ms/frame | 13.62 ms/frame | 54 | 4.34 s | 2476 MiB |
| short line, 0.6B | 193 ms | 5.68 ms/frame | 13.54 ms/frame | 51 | 4.05 s | 1414 MiB |
| `ward.txt`, 1.7B | 832 ms | 10.46 ms/frame | 13.58 ms/frame | 506 | 40.45 s | 2620 MiB |
| `ward.txt`, 0.6B | 352 ms | 6.74 ms/frame | 13.36 ms/frame | 554 | 44.35 s | 1558 MiB |

The 0.6B makes the talker **36–39% cheaper** and halves prefill — but the code
predictor is byte-for-byte the same 5-layer stack running 15 sequential passes
per frame in both models, and the vocoder is literally the same GGUF. Two of the
three stages do not shrink at all.

**Re-measured 2026-08-24**, once the vocoder stopped running on the CPU and
stopped dwarfing everything (`ward.txt`, pipeline off for the stage numbers):

| | talker | code predictor | generate | vocoder | **per frame** |
|---|---|---|---|---|---|
| CUDA, 1.7B | 10.1 | 13.2 | 25.1 | 12.9 | **38.0 ms** |
| CUDA, 0.6B | 6.5 | 13.0 | 20.3 | 12.9 | **33.2 ms** |
| ROCm, 1.7B | 7.1 | 10.9 | 18.9 | 8.9 | **27.8 ms** |
| ROCm, 0.6B | 4.5 | 10.8 | 16.1 | 8.6 | **24.7 ms** |

So the 0.6B is now worth **11–13% per frame**, up from ~6% when a 53 ms/frame
vocoder sat on top of everything. Real, but nowhere near proportional to a model
2.8x smaller — and the reason is arithmetic: of the 38 ms a frame costs on the
1.7B, only the talker's 10.1 shrinks, and only by a third.

**RTF still cannot compare the two.** They produce different amounts of audio
for the same text, and `ward.txt` on ROCm shows exactly why: wall clock came out
identical (11.85 s vs 11.84 s) while RTF read 0.300 for the 1.7B and 0.273 for
the 0.6B — the entire difference is that the 0.6B spoke 43.4 s where the 1.7B
spoke 39.5 s. Compare ms/frame.

**Still choose the 0.6B for memory, not for speed:** ~1.06 GB less VRAM,
consistently, which is the difference between fitting alongside something else
on a 6 GB card and not. The quality cost is audible and has not changed.

One forward-looking note: the 0.6B's advantage is capped by the size-invariant
stages, so **it grows if the code predictor gets faster** — the two items are
related, not independent.

## The conv_transpose experiment

Measured **2026-08-26**. This section replaces the older "two ways in, neither
tried" plan, which assumed getting the op onto the GPU was the win and the
kernel's quality a later argument. It is the other way round.

### What was wrong

`CONV_TRANSPOSE_1D` is 45–52% of vocoder decode and **had never run on a GPU**.
`supports_op` accepts the op only when both inputs are F32, and the vocoder
ships its six conv_transpose weights as F16:

- `ggml-cuda.cu:5142`
- `ggml-vulkan.cpp:11549` — **so Vulkan was affected too**, which the previous
  revision of this file recorded as a CUDA/HIP-only problem.

Six nodes: `tok_dec.upsample.{0,1}.conv.weight` (stride 2) and
`tok_dec.dec.{1,2,3,4}.conv_t.weight` (strides 8, 5, 4, 3). Each was a CPU
convolution with a GPU round trip either side.

> **A retraction that belongs here.** This paragraph used to continue: *"which
> is also why `GGML_NATIVE` is worth 2.25x end to end — it governs exactly the
> CPU kernels these six land in."* Both halves are wrong. Re-measured
> 2026-08-26 against purpose-built `NATIVE=OFF` build directories, the flag has
> **no measurable effect** on either backend, with the GEMM vocoder *or* with
> the CPU fallback forced back on (four rows, all within noise — in `speed-v1.tsv`, the pre-2026-08-27 series).
> And `NATIVE=OFF` does not strip AVX2 or FMA as was claimed — it only drops
> `-march=native`; ggml still builds its explicit ISA variants. What the
> original 26.67-vs-11.86 s measurement actually captured is unknown; it was
> taken on the same day two build directories were found to disagree, so a
> second uncontrolled difference is likely. Keep grepping `CMakeCache.txt`
> before comparing build dirs — but do not attribute a difference to this flag
> without measuring it.

### The fix, and what it cost

Widening those six weights to F32 at load makes the op eligible. **+51 MB**, not
the ~36 MB the old entry estimated from `dec.1` alone. `QWEN3_TTS_CONV_T_F32`.

Isolated vocoder decode, `bench_ru.txt`, 1.7B Q8_0, `QWEN3_TTS_PIPELINE=0`,
fixed seed so the frame count is identical across the pair, three runs:

| backend | card | CPU fallback | on the GPU | |
|---|---|---|---|---|
| CUDA | 1660S | 10.0 ms/frame | **25.5** | 2.5x worse |
| HIP | 6800XT | 6.8 ms/frame | **8.8** | 1.3x worse |
| Vulkan | 6800XT | 7.0 ms/frame | **3.8** | 1.8x better |
| Vulkan | 1660S | 11.0 ms/frame | **8.5** | 1.3x better |

Whole request, the `bench_speed.sh` protocol as it stood then (mean of four
unpinned runs), rows in `benchmarks/speed-v1.tsv`:

| config | control | conv_transpose on GPU |
|---|---|---|
| VK 6800XT | 13.16 s | **11.51 s** |
| VK 1660S | 22.61 s | **20.32 s** |
| CUDA 1660S | **16.71 s** | 25.82 s |
| ROCm 6800XT | **10.76 s** | 14.41 s |

So it defaults on for Vulkan and off everywhere else.

### What that says

**The CUDA kernel, not the placement, is the bottleneck.** Two controls make
this hard to argue with. On the *same* GTX 1660 SUPER, ggml's Vulkan shader
does the op at 8.5 ms/frame where its CUDA kernel takes 25.5 — **3x** — and the
threaded AVX2 CPU fallback takes 10.0, so the CUDA kernel loses to the CPU by
2.5x. Per-op totals over the same workload, profiler on, everything else in the
graph within 1% between the two runs:

| | CONV_TRANSPOSE_1D total |
|---|---|
| CPU fallback | 3365 ms |
| CUDA kernel | 14203 ms |

`ggml/src/ggml-cuda/conv-transpose-1d.cu` explains it: one thread per output
element, each looping over every input channel and the whole kernel width,
reading straight from global memory with no tiling and no reuse, and
`continue`-ing out of roughly `s0-1` of every `s0` iterations. The Vulkan shader
parallelises over output channels instead. See `ggml-notes.md`.

**The ceiling is still cuDNN's 2.67 ms/frame** on the 1660 SUPER (*Against the
reference PyTorch pipeline*). Vulkan on the 6800 XT now reaches 3.8, so the gap
that remains is a kernel gap on one backend, not a structural one.

### The third route, and it wins on every backend

Same day, and it makes the whole argument above a historical note.

ggml already has `GGML_OP_COL2IM_1D` — "scatter-add GEMM columns back to 1D
signal". A transposed convolution *is* a GEMM producing `K*OC` columns per input
step followed by a scatter-add of those columns onto the signal, so the op can
be written as `mul_mat` + `col2im_1d` and never touch `conv_transpose_1d` at
all. That routes the work through cuBLAS/rocBLAS/the Vulkan matmul, keeps the
weights F16, and needs no ggml patch.

The six weights are repacked once at load, from ggml's `[K, OC, IC]` to the
`[IC, K*OC]` a matmul contracts over. The op emits the same uncropped length as
before, so every downstream crop and streaming overlap is untouched.
`QWEN3_TTS_CONV_T_GEMM`, on by default.

Isolated vocoder decode, `bench_ru.txt`, 1.7B Q8_0, `QWEN3_TTS_PIPELINE=0`,
fixed seed, three runs — against the best each backend could previously do:

| backend | card | best before | GEMM + col2im | |
|---|---|---|---|---|
| CUDA | 1660S | 10.0 (CPU fallback) | **4.29** | 2.3x |
| HIP | 6800XT | 6.65 (CPU fallback) | **1.49** | 4.5x |
| Vulkan | 6800XT | 3.82 (shader) | **1.05** | 3.6x |
| Vulkan | 1660S | 8.44 (shader) | **5.04** | 1.7x |

Whole request, `bench_speed.sh` protocol, against the original baseline:

| config | baseline | GEMM + col2im | |
|---|---|---|---|
| CUDA 1660S | 16.71 s | **15.96 s** | −4.5% |
| ROCm 6800XT | 10.76 s | **10.51 s** | −2.3% |
| VK 6800XT | 13.16 s | **10.01 s** | **−24%** |
| VK 1660S | 22.61 s | **19.01 s** | **−16%** |

**Why the end-to-end gain is so uneven:** CUDA and ROCm overlap decode into
generation, so most of the vocoder was already hidden behind the talker and
shrinking it recovers only the exposed part. Vulkan cannot overlap
(`pipeline_supported()`), so its vocoder time is fully on the critical path and
the whole win shows up. **On the 6800 XT this flips the ranking: Vulkan at
10.01 s now beats ROCm at 10.51 s.**

Correctness: output matches ggml's own op at **67 dB SNR**, correlation
0.9999999 — F16 rounding, not a difference in the maths. Streaming parity passes
on CUDA, HIP and Vulkan. Cost is **+34 MB** at the shipping 16-frame chunk
(+174 MB at the 100-frame sequential chunk), for the `[K*OC, T_in]` F32
intermediate.

Against the reference: PyTorch+cuDNN does this decode at 2.67 ms/frame on the
1660 SUPER where we were at 9.60. At 4.29 the gap is 1.6x, down from 3.6x.

**Still worth reporting upstream**, and now with a sharper point: ggml's
`conv_transpose_1d` CUDA kernel is beaten 6x by a decomposition built out of
ggml's own ops, and 2.5x by the CPU. Either the kernel should be rewritten or
the frontend should lower `conv_transpose_1d` to `mul_mat` + `col2im_1d` where
a backend has no good kernel. Hold it with the other two ggml findings — see
`ggml-notes.md`.

## Per-voice prefill reuse

The idea was "cache the voice's KV and stop re-deriving the reference on every
request". What it is actually worth turns entirely on **where the reference
sits in the prompt**, and that had never been checked. The layout, logged by
`build_prefill_graph` (1.7B, the benchmark voice, a 73-character line):

```
ICL layout: prefix=9 ref_text=37 new_text=27 eos=1 codec_bos=1 ref_frames=112 total=187
```

Read it in order: role tokens and codec overlay, the **reference transcript**,
the **line being spoken**, then the **112 reference frames**. The frames are
last. Attention is causal, so from the second layer up their K and V mix in the
target text — a different text every request. **The 112 frames, 60% of the
prompt and the whole reason the idea looked big, are not reusable at all.**

What is reusable is the head: `prefix + ref_text`, **46 of 187 tokens**. That
is what shipped, together with a second cache that has nothing to do with KV.

### The two caches

**The reference block** (`voice_prefix_entry`). Assembling the prompt's
reference section sums 16 codebook rows per frame — 1792 single-row device
reads for a 112-frame voice — and projects the reference transcript. Both
depend on the voice alone. Cached whole.

**The prompt head's KV** (`prefix_kv_entry`). The talker's K/V tensors are
`[head_dim, n_kv_heads, n_ctx]` with the token index on the slowest axis, so a
prefix is one contiguous byte range per layer and survives a change of `n_ctx`
untouched — snapshot with one `tensor_get` per layer, restore with one
`tensor_set`.

Both are capped in voices by `QWEN3_TTS_PREFIX_CACHE` (default 8) and evicted
least-recently-used, ~7 MB of host memory per voice on the 1.7B.

### Measured

Fixed seed, so the request is bit-identical run to run and the only variance
left is timing noise. Medians of 10 (short) and 5 (long), 1.7B Q8_0, warm
server, the benchmark voice. Three line lengths: a 26-character clause, then
`bench_ru_short.txt` (73 chars) and `bench_ru.txt` (570 chars).

| | line | prefill before | prefill after | decode loop before | after |
|---|---|---|---|---|---|
| CUDA 1660S | 26 ch | 310 ms | **201 ms** | 25.90 ms/frame | 26.04 |
| CUDA 1660S | 73 ch | 310 ms | **249 ms** | 27.16 ms/frame | 27.04 |
| CUDA 1660S | 570 ch | 617 ms | **533 ms** | 29.23 ms/frame | 29.48 |
| ROCm 6800XT | 73 ch | 81 ms | **30 ms** | 19.95 ms/frame | 19.73 |
| ROCm 6800XT | 570 ch | 112 ms | **60 ms** | 20.11 ms/frame | 20.05 |

The decode-loop column is the control: generation is untouched and does not
move.

**Whole-request wall time usually cannot resolve this change**, because a
changed last bit sends the sampler down a different path and the two runs
generate a different number of frames — 503 vs 533 on the long line, and no
comparison survives that. The 26-character line is the exception worth having:
it came out at **24.0 frames on both sides**, so its totals are directly
comparable — **986 → 880 ms, −10.7%**. That is the best case, and it is the
one a TTS server hits most often.

The split of the saving differs by card, which is the interesting part:

| | build (reference block) | forward (KV) |
|---|---|---|
| CUDA 1660S | 24 → 1 ms | 288 → 245 ms |
| ROCm 6800XT | 47 → 1 ms | 38 → 33 ms |

On the 6800 XT prefill *forward* is only 38 ms, and the 1792 device reads cost
more than the entire transformer pass. The cache that matters there is the
cheap one. On the 1660 SUPER the two contribute about equally.

In request terms: **60-110 ms off a request on CUDA, ~50 ms on ROCm**, near
enough fixed — it is the same 46 tokens and the same reference block whatever
the line. As a share that is **10.7% of a 26-character clause, 3% of a
one-sentence line and 0.5% of a 42-second paragraph.** The original 16% was
never there.

### In the deployed stack

The numbers above are the benchmark voice, whose reference transcript is 93
characters. A real one from the 198-voice library is bigger and the reuse
scales with it:

```
ICL layout: prefix=9 ref_text=66 new_text=24 eos=1 codec_bos=1 ref_frames=150 total=251
```

`ostro`, 196 characters of transcript, and its reference hits the 150-frame
cap. The reusable head is **75 of 251 tokens** against 46 of 187 for the
benchmark voice, and the reference block is 2400 device reads instead of 1792.
Three identical requests through the stack's proxy, 1.7B Q8_0 on the 1660 SUPER:

```
[1635] ok 3.44s audio in 2472ms (prefill=479ms ...)   first request, cache miss
[1636] ok 3.60s audio in 1497ms (prefill=291ms ...)   hit
[1637] ok 3.68s audio in 1524ms (prefill=291ms ...)   hit
```

**479 → 291 ms, −39%** — the reuse is worth more here than on the benchmark
voice because the head is longer. The totals are not comparable across those
three lines (3.44 / 3.60 / 3.68 s of audio is a different frame count each
time, and the first also pays warm-up).

The cache holds 8 voices; the library has 198. That is deliberate and it costs
nothing to be wrong about, because the startup preload never touches this
cache — it only encodes each voice's embedding and reference codes, which is a
different thing entirely. Only voices that are actually spoken take a slot. A
workload that cycles through more than 8 speakers before repeating will thrash
it; raise `QWEN3_TTS_PREFIX_CACHE` there, at about 7 MB of host memory per
voice.

### Why the prefill is split even on a cache miss

The head is prefilled as its own chunk whether or not it was cached. A head
computed in a batch of 46 and the same head computed inside a batch of 187 are
the same mathematics but not the same last bits, and without the split the
first request for a voice would sound different from every request after it.
With it, **hit and miss are byte-identical** — verified by md5 of the returned
WAV. The cost is one extra graph launch on the first request per voice
(forward 305 → 343 ms, once).

The reference transcript is projected on its own for the same reason: batched
together with the line, its rows would shift with the line's length and the KV
cache would miss on bytes that are mathematically identical.

### Correctness

Everything below is byte-exact comparison of the returned WAV, fixed seed:

* cache on == cache off (`QWEN3_TTS_PREFIX_CACHE=0`), short and long.
* first request for a voice (miss) == every request after it (hit).
* Three voices built to be adversarial — same audio with a different
  transcript, and the same transcript with different audio — interleaved
  `a,b,c,a,b,c,c,a` on one server, each matching the output of a server that
  only ever saw that one voice. Repeated with `QWEN3_TTS_PREFIX_CACHE=1`, so
  every voice switch evicts.
* The non-ICL path (no reference at all) does not use either cache.

## Remaining ideas (descending value)

1. **`CONV_TRANSPOSE_1D` — done.** Rewritten as `mul_mat` + `col2im_1d`; decode
   is 2.3-4.5x faster on every backend and the vocoder is no longer a
   meaningful share of a request. See *The conv_transpose experiment* above.
   What is left here is an upstream report, not more local work.
   **The vocoder is closed as a target; everything below is generation.**
2. **The code predictor: only one lever left, and it is not a code change.**
   At 13.2 ms/frame it is the largest single line and no longer hidden, but it
   is ~70% weight traffic (see the ruled-out entry). Quantising it works and is
   worth 6.6% end-to-end, rejected on audible quality. Fusing the passes bought
   3% and broke Vulkan. Everything that targets dispatch overhead is capped at
   the remaining ~30% of the stage. **Cutting the number of sequential passes is
   the only large win, and that is a model-architecture question.**

   The one free scrap: `code_pred.lm_head.*` ships F16 while the rest of the
   talker is Q8_0. Near-lossless to quantise, 30 MB of file, ~2% of the stage.
3. **Overlap pays less than it used to.** With the vocoder on the CPU the
   pipeline overlapped two different pieces of hardware; now both stages want
   the same GPU and it recovers only part of the decode (CUDA 20 493 → 18 977
   ms, ROCm 13 715 → 11 846). Still worth keeping on, no longer a big lever.
4. **Per-voice prefill reuse — done, and three times smaller than this entry
   used to claim.** The old text said prefill was "dominated by the 150 ICL
   reference frames — and it is recomputed on every request, though for a fixed
   voice that prefix is byte-identical every time", and priced that at 16% of a
   request. **The premise was wrong.** The reference frames are byte-identical
   as *input embeddings*, but they sit at the **end** of the prompt, after the
   line being spoken, so their KV is different for every request and can never
   be reused. Only the head is reusable. See *Per-voice prefill reuse* below for
   what that is worth and what shipped.
5. **The streaming vocoder's host round-trip — mostly bounded, never timed
   directly.** `stream_decode` pulls the conv tails and KV device→host after
   every chunk and pushes them back before the next, a fixed cost per chunk.
   The chunk sweep caps it indirectly: halving the number of chunks at the
   default (100 → 200 frames, 7 round-trips → 4) changes nothing measurable,
   so whatever the copies cost, at the operating point we ship it is under the
   noise floor. What the sweep cannot do is separate the copies from graph
   efficiency, and the 25-frame cliff is large enough that *something* per
   chunk is expensive. One timing line around the copy block in
   `audio_tokenizer_decoder.cpp` still settles it, and it is cheap.

## Two cards, and what that is actually good for

Both stages want the same GPU now, which is why the overlap fell from 25-40% to
~7% (idea 3). This machine has two cards, so splitting the two models across
them is the obvious thought. Two things to know before trying it.

**CUDA and ROCm cannot be linked into one binary.** The HIP backend compiles the
*same* `ggml-cuda/*.cu` sources with a different compiler
(`ggml/src/ggml-hip/CMakeLists.txt` globs them), so a static build with both
collides on every symbol. The only route is `GGML_BACKEND_DL=ON`, which builds
each backend as its own shared object loaded at runtime — untested here, and it
requires `BUILD_SHARED_LIBS`. **CUDA + Vulkan in one binary is fine**: separate
sources, no conflict. So the cheap experiment is talker on CUDA, vocoder on
Vulkan pointed at the other card, which needs `QWEN3_TTS_VOCODER` extended from
`cpu`/`gpu` to a device selector. Whether it wins is unmeasured — the ceiling is
`max(talker, vocoder)` per frame instead of the sum, minus whatever the
cross-device transfer costs.

**For batch work, do not do any of that.** Splitting one request across two
cards is a latency optimisation and a hard one. Audiobook-style work wants
throughput, and there the answer is free: **run two independent servers, one per
card, and split the text between them.** Separate processes, separate binaries,
no symbol problem, no code. On 2026-08-25 numbers (10.76 s and 16.16 s for the
same work) that is about +65% throughput — more than any single optimisation
listed above.

## Tried and ruled out — do not redo

- **Fused code predictor** (one graph for all 15 codebooks, `argmax` +
  `get_rows` kept on device, zero logits readbacks). Worth only ~3% on CUDA —
  proving the cost is per-kernel latency inside the chain, not per-dispatch
  overhead — and catastrophic on Vulkan (a 4 s line did not finish in 10 min).
  Removed. Revisit only together with a way to cut the *number* of sequential
  kernels, e.g. truncating the codebook chain.
- **Truncating the RVQ codebook chain** (`--codebooks N`, kept as a flag). Cost
  scales exactly as hoped — code predictor 14.6 → 7.1 ms/frame at 8 codebooks —
  but the codebook embeddings are summed back into the talker's step embedding,
  so zeroing the tail pushes the talker out of distribution and the *text*
  degrades: at 8 codebooks the ASR reads "беседно исчез 3-звучая иностранный
  пацан ик", and the 0.6B model ran to the frame cap (491 s of audio for a 43 s
  script). Clean down to 12, where the saving is only ~3% RTF. See
  `docs/known-issues.md` #7.
- **F32 vocoder weights** (hypothesis: TU116 lacks tensor cores, so F16 GEMM
  might lose to SGEMM). Converted the tokenizer GGUF to F32: 11% *slower*
  (24.8 s vs 22.4 s). The doubled weight traffic costs more than the compute
  path saves. F16 stays. Measured with the vocoder on the CPU, and a separate
  question from the F32 *accumulator* the vocoder graph now asks for, which is
  both more accurate and free — see `known-issues.md` #14.
- **CUDA graphs** are enabled (`GGML_CUDA_GRAPHS=ON`, off by default in ggml)
  and do capture, but the generation loop is not launch-bound once the KV
  window is fixed, so they are not where the remaining time is.
- **Vocoder decode batch size.** Swept 8→200 on `ward.txt` with a fixed seed
  (identical 519 frames every run): 33.4 / 30.7 / 29.9 / 29.8 / 30.3 / 31.0 /
  33.6 s for 8 / 16 / 32 / 64 / 100 / 128 / 200. A shallow optimum at 32–64
  worth 2.6%, bounded on one side by the talker blocking on small inefficient
  vocoder calls and on the other by the tail that has to be decoded after
  generation ends. On a short line the ordering reverses — a batch bigger than
  the utterance never overlaps at all (56-frame line: 9.41 s at 16, 9.64 s at
  32, 10.28 s at 64) — so the default stays 16. The *sequential* path defaults
  to 100 instead, which is a different question, answered in *Chunk size is not
  a speed lever*. Caveat on this entry: it is one run per configuration, and a
  2.6% optimum is below the noise floor this box turned out to have — the
  direction (small batches overlap sooner, big ones never overlap) is sound, the
  2.6% is not worth quoting.
- **Quantising the code predictor.** It is **bandwidth-bound, not compute-bound**
  — established by running it *both* directions on `ward.txt`, 1.7B, CUDA, with
  the talker as an untouched control (10.1 → 10.3 ms/frame throughout):

  | code predictor stack | bytes read per pass | ms/frame |
  |---|---|---|
  | F16 (dequantised) | 158 MB | 21.0 |
  | Q8_0 (shipped) | 86 MB | 13.2 |
  | Q4_0 + Q8_0 heads | 47 MB | 10.2 |

  Solving the two points: ~9.3 of the 13.2 ms/frame is weight traffic at
  ~139 GB/s effective, and ~3.9 ms is everything else. So the stage is ~70%
  memory.

  Q4_0 works and is worth **6.6% end-to-end** (RTF 0.440 → 0.411, 2.27x → 2.43x
  realtime, pipeline on). Rejected anyway: it coarsens by 4x exactly the part of
  the model that produces fine acoustic detail, and Oleg could hear a difference.
  Q5 would halve the gain for half the risk, which is not a better trade.

  What this rules out for the future: chasing arithmetic or better kernels in
  this stage is pointless — the card is idle waiting on memory. CUDA graphs,
  fusing the passes, and dispatch overhead all address the 3.9 ms, so they are
  capped at ~30% of the stage and part of that is already taken. **The only
  large lever left is reducing the number of sequential passes**, which is a
  model-architecture question, not a code one.

  Noted in passing: the 15 `code_pred.lm_head.*` tensors ship as F16 while the
  rest of the talker is Q8_0. Quantising only those is near-lossless and saves
  30 MB of file, but it is 2% of the stage's traffic — measurable, not useful.

- Everything in the June list still holds: persistent attention mask, decode
  batch 200/400/800, embedding lookup via host scratch, one-shot vocoder
  `decode()` for long clips.

## History: the August 2026 sweep

Numbers in this section and the next are from 2026-08-20, with the vocoder
on the CPU. The relative wins (KV windowing, prefill, backend ranking) still
hold; the absolute end-to-end and RTF figures do not.

**Test configuration:** NVIDIA GTX 1660 SUPER (Turing TU116, 6 GB, **no tensor
cores**), Ryzen 7 5700X, Q8_0 1.7B Base talker + F16 vocoder, ~42 s Russian
clip, ICL cloned voice, default sampling (`temperature 0.9`, `top_k 50`).

**Read the numbers in this table as warm-process numbers** — a server that has
already answered one request for this voice. A cold process pays **~2.2 s more**
for the ICL warm-up, where the vocoder decodes the reference frames before the
first synthesis (6.2 s before the vocoder reached the GPU; re-measured
2026-08-24 as the gap between tokenize and prefill, 1.9 s on ROCm;
it is cached per process, so the CLI pays it on every invocation and the server
only once per voice). Comparing a CLI run against this table without accounting
for that is how the 22.9 s below turns into an apparent 29.8 s.

| Metric | Before | After |
|--------|--------|-------|
| End-to-end, long clip | 33.7 s | **22.9 s** |
| RTF, long clip (lower is better) | 0.815 | **0.53** (1.9x realtime) |
| RTF, short line, warm server, 1.7B | 0.70 | **0.60** |
| Talker step, 500-frame context | 29.5 ms | **18.0 ms** |
| Prefill (210 tokens) | 1450 ms | **415 ms** |

Backend guidance changed on **both** cards. The vendor backends (CUDA, ROCm)
win, because they are the ones that can overlap the vocoder with generation:

| GPU | backend | RTF |
|---|---|---|
| RX 6800 XT | ROCm/HIP + overlap | **0.530** |
| RX 6800 XT | Vulkan (`GGML_VK_DISABLE_MULTI_ADD=1`, needed on the mesa of the day) | 0.649 |
| RX 6800 XT | Vulkan, ggml 0.9.11 — what the fork shipped | 0.672 |
| GTX 1660 SUPER | CUDA + overlap | **0.530** |
| GTX 1660 SUPER | Vulkan | 0.793 |

Both cards used to rank the other way round; the causes are gone and written up
in `known-issues.md` #9 (ROCm) and #10 (CUDA).

### What was done

1. **ggml 0.9.11 → 0.20.2.** The April revision does not compile against
   CUDA 13.3 (`cuda::make_strided_iterator` in `argsort.cu`), which is what
   blocked the CUDA backend in the first place.

2. **KV windowing in the talker** (`build_step_graph`, `build_prefill_forward_graph`).
   Both graphs previously viewed the *entire* KV cache — 6413 rows, sized from
   `MAX_AUDIO_TOKENS` — so every step ran attention over 6413 positions when
   only ~260 were populated, and flash-attn alone was 31% of generation. Step
   graphs now view `GGML_PAD(n_past + 1, QWEN3_TTS_KV_STEP)` rows (64-row
   granularity keeps the graph shape stable for long stretches), prefill views
   `n_past + n_tokens`. Talker step time fell 29.5 → 18.0 ms/frame, prefill
   1450 → 415 ms. Output is unchanged in content — the discarded tail was
   zeroed cache masked to `-inf` — though flash-attn's reduction order differs,
   so greedy decoding can pick a different (equally valid) path.

3. **Generate/decode overlap** (`decode_pipeline` in `qwen3_tts.cpp`). The
   vocoder now runs on a worker thread while the talker keeps generating,
   instead of being called synchronously from the frame callback. The talker's
   per-step graphs are tiny and leave most of the GPU idle; the vocoder's conv
   tower saturates it. This is the single largest win: 31.5 s → 22.4 s on the
   long clip. PCM is still delivered to the caller from the calling thread, in
   order, so the `on_pcm` contract is unchanged. Verified against the
   sequential path at corr 0.9999999, identical RMS.

   **Backend-gated**: enabled on CUDA and Metal, disabled on Vulkan. Two ggml
   Vulkan backend instances on one device do not make progress concurrently —
   they serialise so badly that a 4-second line did not finish in 10 minutes.
   Override with `QWEN3_TTS_PIPELINE=1/0`.

4. **Snake activation ordering** (`apply_snake`). The alpha/beta `exp`/`scale`
   nodes were emitted between `sqr` and `mul`, breaking the consecutive
   `mul→sin→sqr→mul→add` run that backends fuse into one kernel. They are now
   materialised before the chain. No measurable win on this card (the fusion
   still does not trigger), but the graph is no longer structurally hostile.

5. **Shorter ICL reference audio** — not the throughput win it was first
   reported to be; the original claim came from an RTF comparison, which is
   invalid here (see *Measurement notes*: the reference sets the speaking pace,
   so it changes how much audio gets produced).

   Measured 2026-08-20 (1.7B Base, CUDA, `ostro` 11.9 s vs the same voice
   cut to 3.2 s, 5 runs each, means):

   | | prefill | ms/frame | frames | audio | generate | RTF |
   |---|---|---|---|---|---|---|
   | short line, 11.9 s ref | 446 ms | 23.04 | 53.2 | 4.26 s | 1.67 s | 0.393 |
   | short line, 3.2 s ref | 172 ms | 22.26 | 61.0 | 4.88 s | 1.53 s | 0.313 |
   | `ward.txt`, 11.9 s ref | 836 ms | 23.94 | 517 | 41.4 s | 13.21 s | 0.319 |
   | `ward.txt`, 3.2 s ref | 547 ms | 23.58 | 559 | 44.7 s | 13.73 s | 0.307 |

   What is real: the reference is prepended to the prompt, so trimming it takes
   **a fixed ~290 ms off prompt processing** per call, and roughly halves the
   first call for a voice (the ICL vocoder warm-up decodes the reference
   frames). Per-frame cost barely moves — 2–3%, because the attention window
   grows past the reference length within a few hundred frames either way.

   What is not real: a throughput win. The 3.2 s cut makes the model produce
   **more audio for the same text** — +15% on a short line, +8% on `ward.txt`.
   Those extra frames cost more than the 290 ms saved as soon as the line is
   long enough. End to end on a warm server, a short line came out at
   3.001 s ±0.141 with the long reference and 2.921 s ±0.179 with the short
   one — a 2.7% difference inside the run-to-run scatter — while `ward.txt`
   was **3.9% slower** with the short reference.

   So: pick a reference for **prosody and for cold-start cost**, not for
   throughput. Quality across lengths is not monotonic either — the 3.2 s and
   the full 11.9 s cut both sound better than the 6.8 s and 10.6 s ones. The
   short cut contains two questions with rising intonation that match an
   interrogative target line; the mid-length cuts add flat narrative speech and
   blur the delivery. Match the register of the lines you will generate. The
   transcript in `sample.txt` must be cut to match the audio, so none of this
   is safe to automate from the audio alone.

6. **Opt-in per-op profiler** (`src/op_profiler.{h,cpp}`, `QWEN3_TTS_PROFILE_OPS=1`).
   Hooks the scheduler's eval callback and reports per (op, shape) totals for
   `generate` and `vocoder`. This is what located the KV-window problem. Note
   that it disables kernel fusion and adds sync per node, so treat its absolute
   numbers as upper bounds and its proportions as the signal.

## The server added 200 ms to every request — the "3% hiccup" (2026-08-27)

The benchmark history described a periodic hiccup: roughly every fourth request
~300 ms slower, "entirely inside the generation loop", on a period of about
38 s. Both halves of that were wrong. It was not periodic and it was not in the
generation loop — it was the HTTP handler, and it was on **every** request.

The non-streaming handler starts a side thread that polls
`req.is_connection_closed()` every 200 ms so a client that hangs up stops the
generation. The thread was a `sleep_for(200ms)` loop, and the handler joins it
on the way out. httplib does not write the response until the handler returns,
so the join sat on the caller's clock, waiting out whatever was left of the
current sleep. Whole-request wall time was therefore **synthesis rounded UP to
a multiple of 200 ms**.

Twenty requests, one warm server, `bench_ru_short.txt`, 1.7B Q8_0, CUDA 1660S,
comparing what the server logged for itself against what curl measured:

| | server's own total | client wall | difference |
|---|---|---|---|
| sleep loop | 2015 – 2033 ms | **2207 ms, all 20 runs** | 174 – 192 ms |
| condition variable | 2006 – 2025 ms | 2013 – 2031 ms | 6 – 7 ms |

The server's own number was stable to 18 ms while the wire time was pinned to
the grid — which is the whole diagnosis in one table. The "every fourth request"
pattern was synthesis time drifting across a tick boundary: a run whose real
cost is 2205 ms is delivered at 2400, one at 2195 ms at 2200.

The fix (`src/server.cpp`) is a `condition_variable::wait_for` with the same
200 ms cadence, notified when the handler stops it, so the join returns at once.
Disconnect detection is unchanged — a client killed mid-request still gets 499
within one tick (measured 3009 ms into a request it abandoned at 3000 ms).

Worth, by request length — cost is a flat 0–200 ms, so the share is whatever
that is against the request:

| text | before | after | |
|---|---|---|---|
| `bench_ru_short.txt` (136 B, 5.44 s audio) | 2.21 s | **2.05 s** | −7% |
| `bench_ru.txt` (1032 B, 42.16 s audio) | 15.61 s | **15.51 s** | −0.6% |

The short row is the A/B above — one server configuration, one model file, only
the binary changed — because the `speed.tsv` row that predates the fix was taken
on the *previous* Q8_0 file and produces 5.60 s of audio, so its seconds are not
comparable. The paragraph row is a straight `speed.tsv` comparison against
2026-08-27T13:08Z: same commit lineage, same file, same 42.16 s of audio.

Two things this also fixes. The benchmark could not resolve anything finer than
its 200 ms grid: the long-text row used to read
`15.41,15.41,15.41,15.41,15.61,15.61,15.61,15.61,15.61` — two discrete values —
and now reads a continuous 15.42 – 15.59. And the live-streaming path never had
this thread at all, so `stream_format=sse|audio` never paid it; only the
one-shot endpoint did, which is the one every benchmark row measures.

The lesson worth keeping: **a number that only ever lands on a round grid is
measuring the harness, not the work.** The grid was visible in every row of
`benchmarks/speed.tsv` from the day the file was created — every wall time
ending in .01, .21, .41, .61 or .81 — and it was read as machine noise for a
day and a half.

## Tracking speed across commits

`scripts/bench_speed.sh` runs one configuration and appends a row to
`benchmarks/speed.tsv`, which is committed. The point is to be able to look back
and say *this backend was faster a week ago, so we broke something* — or the
reverse — instead of arguing from memory.

```
export TTS_MODEL=... TTS_VOCODER=... TTS_VOICES_DIR=...
scripts/bench_speed.sh --build build-cuda --label "CUDA 1660S" --env CUDA_VISIBLE_DEVICES=0
```

Every row carries the commit, the build's `GGML_NATIVE` and `QWEN3_TTS_TIMING`,
the model, the text with its byte count, and the **voice**. **A comparison is
only honest between rows where all of those match except the one being
studied.**

The voice belongs in that list because it changes the answer: different voices
speak at different rates, so the same text becomes a different number of frames
and therefore a different number of seconds.

**The default voice ships with the repo** (`benchmarks/voice/`), and the script
stages it into a temporary directory as the server's entire library. Nothing
outside the repo is touched, because a voice library is a working directory that
gains and loses voices — a benchmark cannot rest on one. It is 8.9 s this model
generated from `benchmarks/voice/sample.txt`, so the transcript matches the
audio exactly. That property is the requirement, not a detail: a reference
paired with a transcript it does not match drives generation to the token cap,
which is what `examples/readme_clone_input.wav` did before it was deleted
(`known-issues.md` #23).

Running with no voice at all is not an option either: unreferenced, the model is
unconstrained, and the same text came out 717 frames one run and 496 the next —
a 45% spread, against ~6% with the benchmark voice. You cannot see a 10%
regression through that.

`--voice NAME --voices-dir DIR` still takes a library voice when a specific one
is the point. Seeded rows named `ostro` and `femaleuniquenocturnal` are from
before the built-in voice existed. That is
not pedantry: on 2026-08-25 a full day of cross-backend work turned out to be
measuring `GGML_NATIVE` differing between two build directories, because a cache
made before `063470c` keeps `ON` and anything configured after it gets `OFF`.

What the protocol does and why each part is load-bearing:

- **A warm server, not the CLI.** The CLI pays model load, kernel and shader
  compilation, and the ICL warm-up on every invocation; the deployment pays them
  once. The CLI also defaults to one-shot decode where the server streams.
- **One warm-up request, discarded** — it costs about double (79.9 vs ~32
  ms/frame on CUDA, 2026-08-25).
- **A fixed seed** (`--seed`, default 1234, recorded in the row). This is the
  one thing that makes the headline number usable, and until 2026-08-27 it was
  missing. Whole-request wall time is dominated by how much audio the model
  chose to produce, and with an unpinned seed that is a fresh draw every run.
  ROCm 6800XT, `bench_ru.txt`, 12 runs each:

  | seed | audio produced | wall time |
  |---|---|---|
  | random | 38.72 – 44.88 s (**14.7%**) | 9.81 – 11.01 s (**11.5%**) |
  | fixed | 39.44 s, bit-identical every run | 9.61 – 10.01 s (**4.2%**) |

  The spread in seconds *was* the spread in length, almost exactly. On the
  short text the pinned protocol lands on 1.41 s nine times out of nine.
- **Nine timed requests, and the MEDIAN** — every run kept in the row so a bad
  spread stays visible. This is what made the 200 ms quantisation above
  survivable before it was found, and it is still the right protocol: a mean
  hands a single slow run a ninth of the weight. Every row written before
  the fix carries that quantisation and reads in 0.2 s steps — the first rows
  without it are the two noted "watcher join no longer waits out its 200ms
  sleep tick". Across that line two rows differ by 0–200 ms of harness, so
  compare them only if the gap is larger than that.
- **The produced duration is recorded and checked.** All runs in a row must
  have generated the same audio; if they did not, the script warns and marks
  the row `VARIES`, because a median over different amounts of work compares
  nothing.
- **Whole request wall time, end to end**, as the headline — what a caller
  actually waits for. `ms_per_frame` is the same wall time divided by the
  frames that came out of it, for the cases where seconds cannot be compared
  (see below). Neither is the server's own generation ms/frame: never compare
  backends by *that*, because the vocoder is overlapped into generation on
  CUDA, ROCm and Metal and not on Vulkan or CPU (`pipeline_supported()` in
  `qwen3_tts.cpp`), so it means different things on either side of that line.
- **One voice in the library directory.** The server preloads the whole library
  in the background, and 198 voices compete with what is being measured.

**Which column to compare:**

- Same backend, same weights, a change that does not touch numerics — compare
  `median_s` directly, after checking that `audio_s` matches. If it does not,
  the change moved the token stream and seconds are timing different work.
- Different backend, different weight type, or anything else that changes the
  tokens — the lengths differ by construction, so compare `ms_per_frame`.

The baseline written under this protocol shows why that rule is not pedantry.
Same machine, same GPU, same text, same seed:

| | median_s | audio_s | ms_per_frame |
|---|---|---|---|
| ROCm 6800XT | **9.61** | 39.44 | 20.31 |
| VK 6800XT | 10.21 | 43.28 | **19.66** |

Vulkan is 6% slower by the clock and 3% faster per frame, and both are true:
it generated 10% more audio for the same input, because a different backend's
arithmetic samples a different token stream. Under the old protocol only the
first column existed and this looked like a clean Vulkan regression.

That Vulkan number was checked rather than trusted, because a backend can also
look fast by producing less: both files transcribe to the source text word for
word and sit at the same level (-23.4 dB against ROCm's -23.7).

The full baseline, 2026-08-27, 1.7B `q8_0`, medians of 9 at seed 1234:

| | short text | | long text | |
|---|---|---|---|---|
| | median_s | ms/frame | median_s | ms/frame |
| ROCm 6800XT | 1.41 | 21.92 | 9.61 | 20.31 |
| VK 6800XT | 1.41 | 24.08 | 10.21 | **19.66** |
| CUDA 1660S | 2.21 | 32.89 | 16.41 | 30.80 |
| VK 1660S | 2.81 | 38.51 | 19.21 | 37.54 |

Prefill is where the two cards differ most: 59 ms on the 6800 XT against
528 ms on the 1660 SUPER for the long text, which is four PCIe lanes as much
as it is the GPU.

`--docker IMAGE` benchmarks a built image instead of a build directory, and it
is not the same configuration: the image is built portable (`QWEN3_TTS_NATIVE`
defaults to OFF) and carries its own userspace GPU driver. `qwen3-tts:vulkan`
on the 6800 XT is 20.42 ms/frame against the host build's 19.66 — 4% for
portability, on Debian's mesa 25.0.7 instead of the host's 26.2.1. Benchmark
the image when the question is about what gets deployed.

`benchmarks/speed-v1.tsv` is the history from before this protocol: means of
four unpinned runs. Its rows are kept because they are the record of what was
measured, but they are **not comparable** to rows in `speed.tsv` — different
statistic, different seed regime. The script refuses to append to a file whose
header is not its own, so the two series cannot silently merge.

`benchmarks/bench_ru.txt` is the default input and should stay unchanged; a new
text starts a new series. Rows with `text=ward.txt` predate it and came from
Oleg's own `test_tts.sh` on a file outside this repo.

## Measurement notes

- Wall time is the headline, but only with the seed pinned — otherwise the
  frame count varies run to run with sampling and takes the seconds with it,
  and **ms/frame** is the only thing left worth comparing. Greedy
  (`--temperature 0`) is unstable on this model regardless — it can run to the
  6144-frame cap and produce near-silence. Two of this sweep's early
  "regressions" were that, not code.
- **Do not compare configurations by RTF when they change how much audio the
  model produces.** RTF is time per second of output, so anything that makes
  the model speak longer for the same text looks faster. This is how the
  reference-length item above got written up backwards: RTF improved by 20%
  while the request got no faster. Voice, reference sample,
  instructions and temperature all move the produced duration — compare wall
  time per request, and report the produced duration alongside it.
- Verify intelligibility, not just timing: transcribing the output with an ASR
  model catches degradations that RMS and duration checks miss.
- **Three runs and a median, minimum** — nine in `bench_speed.sh`. (The "~3%
  periodic hiccup" this used to cite was the server's own 200 ms rounding, not
  the machine — see above; it is fixed.) This machine throws occasional 30%
  outliers on an unchanged configuration — 12.3-12.9 ms/frame next to 9.1 in
  the same group during the chunk sweep. Two apparent wins on 2026-08-24 (7% on
  CUDA, 11% on HIP) turned out to be single-run noise. If the spread within a
  group is comparable to the gap between groups, write "no effect", not a
  cautious win. `scripts/bench_torch_vs_ggml.py` and `scripts/bench_ggml_cli.py`
  default to three runs and print medians for this reason.
- Both benchmark scripts split the same three stages — reference encode,
  generate, vocoder decode — and print the same summary, so a PyTorch run and a
  CLI run can go straight into one table. `bench_ggml_cli.py` re-rolls the seed
  on a run-away rather than letting it poison the median.
