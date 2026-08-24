# Qwen3-TTS GGML Optimization Report

Performance characterization of this fork. Last updated **2026-08-24**.

## Summary

> **The tables in this section are from the August 20 sweep and describe a
> build whose vocoder was silently running on the CPU** (`known-issues.md` #13).
> They are kept because the *relative* wins they record — KV windowing, prefill,
> backend ranking — are real and still hold. The absolute end-to-end and RTF
> figures are not current. For numbers that describe today's build, go to
> **Current profile** below.
>
> What moved on 2026-08-24: vocoder decode 53.4 → 12.9 ms/frame (CUDA, 1.7B),
> ICL warm-up 6.2 s → 2.2 s, voice cloning 40 s → 0.65 s (`known-issues.md`
> #13-#15).

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
| RX 6800 XT | Vulkan (needs `GGML_VK_DISABLE_MULTI_ADD=1`) | 0.649 |
| RX 6800 XT | Vulkan, ggml 0.9.11 — what the fork shipped | 0.672 |
| GTX 1660 SUPER | CUDA + overlap | **0.530** |
| GTX 1660 SUPER | Vulkan | 0.793 |

Both cards used to rank the other way round; the causes are gone and written up
in `known-issues.md` #9 (ROCm) and #10 (CUDA).

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
per frame in both models, and the vocoder is literally the same GGUF. So
per-frame cost falls only ~16% (22.9 → 19.2 ms), and the models differ by
±10% in how much audio they produce for the same text, which is enough to
swallow that. Wall-clock per request came out 5% *faster* on the short line and
5% *slower* on `ward.txt`.

**Choose the 0.6B for memory, not for speed:** ~1.06 GB less VRAM, consistently,
which is the difference between fitting alongside something else on a 6 GB card
and not. The quality cost is audible.

## What was done (August 2026)

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

## Current profile (where the time goes)

**Re-measured 2026-08-24, after the vocoder was found to have been running on
the CPU (`known-issues.md` #13). Every profile older than that date described
the wrong device — do not mix the two.**

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
| vocoder decode | 12.9 ms/frame | 8.9 ms/frame |
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

| op | share of decode |
|---|---|
| `CONV_TRANSPOSE_1D` | **52%** |
| `MUL_MAT` (the conv towers) | 11% |
| `ADD` | 2% |
| `IM2COL` | **1.7%** |

Note what this kills. The whole previous roadmap was built on giving ggml a
direct `conv_1d` kernel to avoid im2col — and im2col is now **1.7%** of decode.
That project is not worth doing for this model.

The new target is `CONV_TRANSPOSE_1D`, and it is a far better-shaped one,
because unlike `conv_1d` it is a real op with a real kernel to improve. The
biggest single entry costs **71 ms per call** for a `[520, 768]` output, which
is absurd for 400k elements. `ggml/src/ggml-cuda/conv-transpose-1d.cu` explains
it: one thread per output element, each looping over every input channel and
the whole kernel width, reading weights and inputs straight from global memory
with no tiling or reuse — and `continue`-ing out of roughly `s0-1` of every
`s0` iterations, so most of the loop is thrown away. See `ggml-notes.md`.

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
  32, 10.28 s at 64) — so the default stays 16. Note the one-shot path uses a
  separate hardcoded 100.
- Everything in the June list still holds: persistent attention mask, decode
  batch 200/400/800, embedding lookup via host scratch, one-shot vocoder
  `decode()` for long clips.

## Remaining ideas (descending value)

1. **`CONV_TRANSPOSE_1D` kernel in ggml** — 52% of decode, and the current CUDA
   kernel is naive enough that this is ordinary optimisation work rather than
   research: tile it, stage weights in shared memory, and stop iterating over
   the stride positions that are discarded. Upstream project; benefits every
   audio decoder on ggml.
2. **The code predictor is a target again** at 13.2 ms/frame — now the single
   largest line, no longer hidden. Note what was already ruled out: fusing the
   15 passes into one graph bought 3% and broke Vulkan, and only ~10% of its
   time is framework overhead, so the win has to come from cutting the *number*
   of sequential passes, not from dispatching them faster.
3. **Overlap pays less than it used to.** With the vocoder on the CPU the
   pipeline overlapped two different pieces of hardware; now both stages want
   the same GPU and it recovers only part of the decode (CUDA 20 493 → 18 977
   ms, ROCm 13 715 → 11 846). Still worth keeping on, no longer a big lever.
4. **Prefill** (793 ms, 1.7B) is dominated by the 150 ICL reference frames. A
   shorter reference shrinks it — but pick a reference for prosody, not speed;
   see the ruled-out note on reference length.

## Measurement notes

- Compare **ms/frame**, not wall time. Frame count varies run to run with
  sampling, and greedy (`--temperature 0`) is unstable on this model — it can
  run to the 6144-frame cap and produce near-silence. Two of this sweep's
  early "regressions" were that, not code.
- **Do not compare configurations by RTF when they change how much audio the
  model produces.** RTF is time per second of output, so anything that makes
  the model speak longer for the same text looks faster. This is how the
  reference-length item above got written up backwards: RTF improved by 20%
  while the request got no faster. Voice, reference sample,
  instructions and temperature all move the produced duration — compare wall
  time per request, and report the produced duration alongside it.
- Verify intelligibility, not just timing: transcribing the output with an ASR
  model catches degradations that RMS and duration checks miss.
