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
| `q4_k` | everywhere | same bytes as `q4_0`, less *weight* error - but see *One case where the rms ranking inverts* before choosing it over `q4_0` |
| `q5_k` `q6_k` | everywhere | the middle of the ladder |
| `bf16` | everywhere with VRAM | bit-identical to the checkpoint |
| `q4_0` | everywhere | more weight error than `q4_k`, and on the one line where the two were compared by transcription it was the one that stayed correct. Not superseded; not obviously worse either |
| `f32` | everywhere | 2x bytes for nothing; no reason left to use it |
| `f16` | **nowhere** | accepted so #16 stays reproducible; warns |

K-quants need `qwen3-tts-quantize` (see below); the Python converter cannot
write them.

Every one of these quantises `talker.text_embd` — the Qwen vocabulary table,
36% of the file — and keeps the *other* embedding tables (`codec_embd`,
`codebook`) at the source type. That split is measured, not inherited: see
*The embedding tables* below.

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

**What does work is many draws, scored by something that is not an ear.** One
draw is a lottery ticket; twenty draws is a rate, and a rate has a confidence
interval. Give the same line to every type with seeds 1..20, transcribe all of
it with `whisper` — the stack already runs one — and score whether the words
came back. That turns "it sounded wrong" into 3/20 against 20/20, which is a
finding. It is how *One case where the rms ranking inverts* below was measured,
and it is the only ears-free test here that has caught something `--verify`
could not see.

## The embedding tables are 36% of the file, and one of them is now quantised

`_should_quantize()` in the Python converter, mirrored by `should_quantize_name()`
in `qwen3-tts-quantize`, used to keep every tensor whose name contains `_embd`,
`codebook`, `_norm`, `lm_head`, `codec_head` or ends in `.bias` at the source
type. The comment under it said only "Tensors to keep in F16 for quality".
No measurement was ever recorded for that. It was measured on 2026-08-26 and
2026-08-27, and `talker.text_embd` now quantises with everything else; the rest
of the list stands.

It cost more than it looked. Broken down by tensor type, 1.7B, **under the old
skip list**:

| | q8_0 | q4_k |
|---|---|---|
| quantised | 1518 MiB | **804 MiB** |
| BF16, kept by name | 820 MiB | **820 MiB** |
| F32 (norms) | 0.6 MiB | 0.6 MiB |
| **total** | **2339 MiB** | **1625 MiB** |

The quantised part shrinks by exactly 1.888x, which is 8.5/4.5 bits — the
arithmetic is doing what it should. The kept block is a fixed 820 MiB floor,
and it is why `q4_k` is only 1.44x smaller than `q8_0` instead of the ~1.9x
the bit widths promise.

**593.5 MiB of that 820 is a single tensor**: `talker.text_embd.weight`,
2048 x 151936 — the Qwen vocabulary. It alone is 36% of the file.
`code_pred.codec_embd.*` (fifteen tables) is 180 MiB more, `spk_enc` 23 MiB.

That one tensor is the whole change made on 2026-08-27. What files look like
now, both sizes, from the same bf16 sources:

| | old default | now | |
|---|---|---|---|
| 1.7B `q8_0` | 2345 MiB | **2067 MiB** | −12% |
| 1.7B `q4_0` | 1630 MiB | **1204 MiB** | −26% |
| 0.6B `q8_0` | 1281 MiB | **1003 MiB** | −22% |
| 0.6B `q4_0` | 1030 MiB | **604 MiB** | −41% |

The smaller model gains more: the vocabulary is the same 151936 rows either way.

### What quantising it actually costs

Measured 2026-08-26. The autoregressive rollout diverges on a changed last bit,
so the last place two builds can be compared number to number is the prefill:
`QWEN3_TTS_DUMP_PREFILL=<path>` writes the last token's hidden state and its
logits as raw f32. Prefill is deterministic, so these are exact values, not
samples. Every row below is against the **bf16** model on the same text, same
voice, same backend (HIP, 6800 XT), with the voice's embedding and reference
codes frozen in `cache.bin` so only the talker's weights differ.

1.7B, four texts (short Russian, a 570-character paragraph, English, one full
of digits):

| | rel. rms of hidden | logit SNR | file |
|---|---|---|---|
| `q8_0`, standard skips | 3.4% | 36.5 dB | 2345 MiB |
| `q4_k`, standard skips | 22.1 – 27.3% | 16.4 – 19.1 dB | 1631 MiB |
| `q4_k` + `text_embd` at `q4_k` | 22.4 – 27.8% | 16.9 – 19.5 dB | **1204 MiB** |

The two `q4_k` rows are the same model to within the spread across texts, and
the logit SNR is *higher* with the embedding quantised in all four. **427 MiB
for nothing.**

Why: quantise `text_embd` **alone**, leaving everything else bf16, and its own
contribution is **0.61 – 1.29%** rel. rms, 42.8 – 50.9 dB. Errors from
independent sources add in quadrature, so 25% and 1% combine to 25.02%. It is
not that the embedding is noise-free; it is that it is twenty-five times
quieter than the body of the model it sits next to.

Both model sizes agree, and the smaller one gains more because the vocabulary
is the same 151936 rows either way:

| 0.6B | rel. rms | logit SNR | file |
|---|---|---|---|
| `q4_k`, standard skips | 22.5 – 23.0% | 18.5 – 19.0 dB | 1025 MiB |
| `q4_k` + `text_embd` at `q4_k` | 22.0 – 23.4% | 18.2 – 19.0 dB | **598 MiB** |

**−26% on the 1.7B, −42% on the 0.6B.**

For reference, llama.cpp quantises `token_embd` in all of its standard quant
mixes. Keeping the whole table at source precision is the unusual choice, not
the safe one.

### What was NOT settled

* **Nobody has listened, and that turned out to matter.** This measures the
  model's computation up to the first frame. The section after this one shows
  a case the measurement is blind to: two files it cannot tell apart differ
  3/20 against 8/20 on reading a number back correctly. Treat the prefill
  metric as a way to rule a change out, not to clear it.
* **The other kept tables are a separate question.** `code_pred.codec_embd.*`
  isolated the same way gives 1.3 – 2.6% / 39.8 – 45.0 dB — larger than
  `text_embd` but still an order of magnitude under the body, and worth
  another 152 MiB. But their main use is inside the code predictor's fifteen
  sequential passes, which the prefill dump does not reach at all, and the
  code predictor is the one place where 4-bit was *heard* (see below). Do not
  move those on this evidence.
* **The default was not changed on this evidence.** It took the transcription
  test below. `--tensor-type` also beats the skip list now, so any tensor can
  still be forced either way per file:

```
qwen3-tts-quantize 1.7B-bf16.gguf 1.7B-fat.gguf q4_k \
    --tensor-type talker.text_embd=bf16
```

### What cleared it: 1460 clips, and no difference anywhere

Measured 2026-08-27. The rule the section below sets is that the prefill metric
can rule a change out but not clear it, and that clearing it needs the 20-seed
transcription test **on a construct the base type does not already break**.
That is what this is, six times over, at three times the seed count.

Files built from `1.7B-bf16.gguf` and `0.6B-bf16.gguf` with one build of
`qwen3-tts-quantize`; `-e` means `--tensor-type talker.text_embd=<base type>`.
Same voice, same frozen `cache.bin`, HIP. Transcribed by `whisper large-v3-turbo`.

| model / line | what is scored | base | +embd | Fisher p |
|---|---|---|---|---|
| 1.7B `В 2026 году … 17,5 процента …` | `17,5` came back | `q8_0` 59/60 | `q8_0e` 58/60 | 1.000 |
| | | `q4_0` 118/120 | `q4_0e` 113/120 | 0.171 |
| 1.7B English pangram | every word | `q8_0` 20/20 | `q8_0e` 20/20 | 1.000 |
| | | `q4_0` 20/20 | `q4_0e` 20/20 | 1.000 |
| 1.7B `Он прочитал Шопенгауэра …` | `Шопенгауэра` | `q8_0` 52/60 | `q8_0e` 55/60 | 0.558 |
| | | `q4_0` 50/60 | `q4_0e` 45/60 | 0.369 |
| 0.6B `Он прочитал Шопенгауэра …` | `Шопенгауэра` | `q8_0` 45/60 | `q8_0e` 45/60 | 1.000 |
| | | `q4_0` 41/60 | `q4_0e` **53/60** | **0.014** |

Pooled: `q8_0` 176/200 vs `q8_0e` 178/200 (p=0.876); `q4_0` 229/260 vs `q4_0e`
231/260 (p=0.891). **Nothing moves.**

Two things in that table need saying out loud rather than reading off:

* **The one significant cell points the wrong way for the cautious answer.** On
  the 0.6B, 4-bit with the vocabulary quantised reads the name *better* — 53/60
  against 41/60, and level with bf16's 52/60, in a file 41% smaller. It does not
  replicate on the 1.7B, where the same pair goes 45 against 50. Across sixteen
  comparisons one cell at p=0.014 is what noise looks like. Read it as "no
  penalty", never as "quantising helps".
* **`bf16` drops the word too.** On the `17,5` line at full precision it says
  "семьдесят пять процентов" once in 60. `q8_0` and `q4_0` do it 0/60 and
  0/120, `q8_0e` and `q4_0e` 2/60 and 2/120 — pooled 1/240 against 4/180,
  p=0.169. The failure mode that condemned `q4_k` (15 times in 20) exists in
  this model at every precision at a ~2% rate. A file is only condemned by it
  at `q4_k`'s magnitude, not at this one.

**The 0.6B could not be tested on the `17,5` line at all.** At bf16 it returns
that number correctly 7 times in 60 and drops the word 14 times; shortened to
`Цена выросла на 17,5 процента.` it stops producing Russian words at all
("на щенсетить 5%", "на сочесенится 5%"). The whole `17,5` collapse is native
to the small model. Screening candidate lines on bf16 *before* comparing types
is not optional — a construct at the floor cannot rank anything.

What is still not settled is the other tables. `code_pred.codec_embd.*` is
another 152 MiB and has never been through this test; the code predictor is the
one place where 4-bit was *heard*. Do not move those without running it.

## One case where the rms ranking inverts — and what that costs the metric above

Measured 2026-08-26, after an ear on a q4_k clip reported that "числа
запоролись". The question was whether that was the seed or the type. It was the
type, and the answer is more specific and more awkward than expected.

All seven files quantised from the same `1.7B-bf16.gguf` with the same build of
`qwen3-tts-quantize`. Same voice, same line, **20 seeds each**, output
transcribed by `whisper large-v3-turbo` and scored on whether each number came
back. Whisper is a noisy judge, but it is the same judge for every row.

Line: `В 2026 году цена выросла на 17,5 процента, до 3480 рублей за штуку.`

| weights | MiB | `2026` | `17,5` | 95% CI on `17,5` | `3480` |
|---|---|---|---|---|---|
| bf16 | 3679 | 20/20 | **20/20** | 84 – 100% | 14/20 |
| `q8_0` | 2344 | 20/20 | 19/20 | 76 – 99% | 13/20 |
| `q6_k` | 1998 | 20/20 | 19/20 | 76 – 99% | 14/20 |
| `q5_k` | 1809 | 20/20 | **20/20** | 84 – 100% | 12/20 |
| `q4_k` | 1630 | 20/20 | **3/20** | **5 – 36%** | 15/20 |
| `q4_k` + `text_embd` | 1203 | 20/20 | 8/20 | 22 – 61% | 5/20 |
| `q4_0` | 1630 | 20/20 | **20/20** | 84 – 100% | 14/20 |

`q4_k` says "семьдесят пять процентов" in 15 of 20 draws. It does not mangle a
digit — it **drops a whole word**. `q4_0`, the same 4.5 bits per weight and a
*worse* relative rms error (8.82% against 7.26%), never does it once.

### What this does and does not license

**It is not a seed lottery.** 3/20 against 20/20 across the same twenty seeds
settles that much.

**It does not generalise to "q4_k breaks numbers."** A second line —
`Заказ номер 482 весит 6,3 килограмма и стоит 1250 рублей.` — was run the same
way, and there every type is bad and the confidence intervals sit on top of one
another: bf16 8/20 all-three, `q5_k` 10/20, `q4_k` 4/20, `q4_0` 3/20. bf16
itself returns "4082" for "482" in three draws of twenty. **Numbers are
unreliable in this model at any precision**, which is the larger fact, and the
`17,5` collapse is one construct on top of it. Note that the second line's
`6,3` — also a decimal with a comma — survives `q4_k` 19/20, so it is not
decimals as a class either.

**It does invalidate a recommendation.** *What to use* above said `q4_k`
supersedes `q4_0` because it has less relative rms error at the same size. That
ranking is real but it is not a ranking of behaviour: here is a line where the
type with less weight error is the one that loses a word, 85% of the time,
against a type with more. Both are listed at 4-bit now with this caveat.

**It puts a ceiling on the prefill metric.** The section above establishes that
quantising `text_embd` is invisible in the prefill's hidden state and logits.
That measurement is still correct — and it is blind to this. `q4_k` and
`q4_k` + `text_embd` were indistinguishable there (22–28% rel. rms, both) and
are 3/20 against 8/20 here. **The prefill metric measures how far the
computation is perturbed, not how the model then behaves.** Use it to rule out
a change that should be numerically neutral; do not use it to clear a change
for release.

### Consequences

* Publish `q8_0`, `q6_k`, `q5_k` without qualification — all three are 19/20 or
  20/20 on the construct that separates them.
* Do not present `q4_k` as strictly better than `q4_0`. At 4 bits, whatever is
  chosen, say that numeric content is where this model is least reliable.
* **The `text_embd` default was not changed on the strength of the prefill
  measurement alone.** That measurement says it is numerically free; this
  section says numerical freedom is not the whole question. What it took was
  this same test against constructs the base type does not already break —
  1460 clips of it, above, which is what actually moved the default.

Checked since (2026-08-27): the `17,5` collapse **is** native to the 0.6B,
which returns the number correctly 7 times in 60 at bf16. And `bf16` on the
1.7B drops the word once in 60 on this line — so this failure mode is not
peculiar to `q4_k`, only its 75% rate is. One line is still one line; the
`Шопенгауэра` line above is the second.

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

A tensor named explicitly by `--tensor-type` is quantised even if the skip
list would otherwise keep it: the list is a blanket policy, naming a tensor is
the operator overriding it. That is how the `text_embd` measurement above was
made, and without it `--keep-all` was the only escape and it takes every other
kept tensor with it.

Otherwise it leaves the same tensors alone the converter does (`codec_embd`
and `codebook` — but not `talker.text_embd`, see above — norms, biases, heads),
plus any row that is not a whole number of blocks - 38 speaker-encoder
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

Where the thing being judged can be written down — a number, a name, a foreign
word — twenty seeds through a transcriber beats any number of careful listens,
and it caught something `--verify` ranked backwards. See *One case where the
rms ranking inverts*.

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
