# Qwen3-TTS GGML Optimization Report

Performance characterization of this fork. Last updated **2026-08-19** after the
August sweep (CUDA backend, KV windowing, generate/decode overlap).

Historical note: earlier revisions of this document described a CPU-only
baseline and then a Vulkan-on-RX-6800-XT baseline. Both are superseded — the
numbers below were measured on the card this fork now targets.

## Summary

**Test configuration:** NVIDIA GTX 1660 SUPER (Turing TU116, 6 GB, **no tensor
cores**), Ryzen 7 5700X, Q8_0 1.7B Base talker + F16 vocoder, ~42 s Russian
clip, ICL cloned voice, default sampling (`temperature 0.9`, `top_k 50`).

**Read the numbers in this table as warm-process numbers** — a server that has
already answered one request for this voice. A cold process pays **~6.2 s more**
for the ICL warm-up, where the vocoder decodes the reference frames before the
first synthesis (measured 2026-08-20 as the gap between tokenize and prefill;
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

Both old impressions were real measurements with since-removed causes. CUDA
measured slower (45.6 vs 38.3 ms/frame generate) because the KV-window bug hit
its flash-attention path harder than Vulkan's; fixing the window took CUDA to
31.5 ms/frame and reversed the ranking. HIP measured slower on a ROCm/ggml pair
that is now years out of date — with ROCm 7.2.4 and ggml 0.20.2, gfx1030 is a
perfectly good target, and Arch's rocBLAS still ships Tensile kernels for it.
(Separately, the April ggml no longer builds against CUDA 13.3, so the old
revision cannot be re-measured without downgrading the toolkit.)

## 0.6B vs 1.7B — not a speed dial

An earlier revision of this table listed the 0.6B at RTF 0.52 against the
1.7B's 0.60 and implied it was the faster model. That was the same RTF mistake
described under item 5 below: the two models do not produce the same amount of
audio for the same text.

Measured 2026-08-20 (CUDA, `ostro`, 5 runs each, means):

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

5. **Shorter ICL reference audio** — originally reported here as the largest
   remaining win, on the strength of an RTF comparison. **That was wrong**, and
   the correction matters more than the original claim: RTF is time per second
   of *produced* audio, so it silently rewards a configuration that makes the
   model talk longer. The reference sets the speaking pace, and a shorter cut
   changed it, so RTF was not comparing like with like.

   Re-measured 2026-08-20 (1.7B Base, CUDA, `ostro` 11.9 s vs the same voice
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

Short line (~4 s audio, 54 frames), warm server, 1.7B, CUDA:

- **prefill ~415 ms** — 210 tokens (150 of them ICL reference frames) through
  28 layers. Close to this card's GEMM throughput; the way to shrink it is a
  shorter reference sample, not a faster kernel.
- **talker ~9 ms/frame**, **code predictor ~13 ms/frame**, embed lookups ~0.2.
- **vocoder ~53 ms/frame** — twice the talker and code predictor combined, and
  the thing everything else now hides behind, not the other way round.

The code predictor is the standout: it costs the **same 13 ms/frame on the
0.6B and the 1.7B model**, because it is 15 sequential 5-layer passes per frame
— ~1100 small kernels whose launch/teardown latency dominates the arithmetic.

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
  path saves. F16 stays.
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

1. **Vocoder convolutions — the only target left.** 53.4 ms/frame against
   25.1 ms/frame for all of generation (`ward.txt`, 1.7B, CUDA, pipeline off so
   the stages are timed apart: generate 13 025 ms, decode 27 703 ms). With the
   pipeline on, 7.5 s of the generate wall is the talker blocked on the decode
   queue. `im2col`+`mul_mat` reaches only ~12% of this card's FP32 peak; a
   direct conv1d kernel would be a ggml-side project. See `docs/ggml-notes.md`.

   The consequence for everything else: **the code predictor is already free.**
   At 13.1 ms/frame it looked like the floor, but it sits inside the 25.1 ms
   that the vocoder's 53.4 ms already covers. Making it instantaneous would
   move generation from 13.0 s to 6.2 s and the total by nothing. Do not spend
   effort there until the vocoder number moves.

## Measurement notes

- Compare **ms/frame**, not wall time. Frame count varies run to run with
  sampling, and greedy (`--temperature 0`) is unstable on this model — it can
  run to the 6144-frame cap and produce near-silence. Two of this sweep's
  early "regressions" were that, not code.
- **Do not compare configurations by RTF when they change how much audio the
  model produces.** RTF is time per second of output, so anything that makes
  the model speak longer for the same text looks faster. This is how the
  reference-length recommendation above got written up backwards: RTF improved
  by 20% while the actual request got no faster. Voice, reference sample,
  instructions and temperature all move the produced duration — compare wall
  time per request, and report the produced duration alongside it.
- Verify intelligibility, not just timing: transcribing the output with an ASR
  model catches degradations that RMS and duration checks miss.
