# Known issues

Running log of bugs and rough edges found while working on this fork. Entries
stay here until fixed or deliberately dismissed — the point is that nothing
found in passing gets lost.

Status: **open** (not fixed) / **fixed** / **mitigated** (symptom addressed,
root cause still there) / **wontfix**.

| # | Issue | Status |
|---|---|---|
| [1](#1) | Streaming decoder is not fully causal — `test_streaming_parity` fails | closed 2026-08-20 |
| [2](#2) | Voice cache is not invalidated across model variants | fixed 2026-08-20 |
| [3](#3) | `--temperature 0` degenerates | fixed 2026-08-20 (budget catches it, both front ends warn) |
| [4](#4) | Vendored ggml could not build against CUDA 13 | fixed (ggml 0.9.11 → 0.20.2) |
| [5](#5) | Two ggml Vulkan backend instances serialise on one device | worked around (pipeline gated to CUDA/Metal) |
| [6](#6) | Stale architecture numbers in `AGENTS.md` | fixed 2026-08-20 |
| [7](#7) | Codebook truncation breaks generation, not just fidelity | wontfix 2026-08-20 |
| [8](#8) | ggml Vulkan multi-add fusion is pathological on RADV / RDNA2 | worked around (env var), upstream bug |
| [9](#9) | ROCm is no longer the slow option on gfx1030 | resolved |
| [10](#10) | Why CUDA used to measure slower than Vulkan | explained; fixed by the KV window |
| [11](#11) | Generation does not stop — runs to the 6144-frame cap on long text | mitigated (guard + retry); root cause is upstream and unfixed |
| [12](#12) | Frame budget counted a digit as a letter | fixed 2026-08-20 |
| [13](#13) | The vocoder never ran on the GPU | fixed 2026-08-24 |
| [14](#14) | GPU vocoder accumulated the whole conv tower in F16 | fixed 2026-08-24 |
| [15](#15) | Voice cloning spent 40 s in a hand-rolled O(n²) DFT | fixed 2026-08-24 |
| [16](#16) | An F16 talker runs away every time; Q8_0 and F32 do not | fixed 2026-08-25 (bf16 added) |
| [17](#17) | A vocoder OOM segfaults instead of failing — upstream ggml | fixed locally 2026-08-24; upstreaming deliberately deferred |
| [18](#18) | The converter's default output type is the one that does not work | fixed 2026-08-25 |
| [19](#19) | Nothing checks the vocoder's output for non-finite samples | fixed 2026-08-25 |
| [—](#rough-edges) | Open rough edges (sampler duplication, env sprawl, no 429, C ABI lag, …) | open |

---

<a id="1"></a>
## 1. Streaming decoder is not fully causal — `test_streaming_parity` fails

**Status:** closed 2026-08-20 — not a bug; the test asked the wrong question
**Severity:** none (was: correctness)

`decode(32)` did not reproduce the first 32 frames of `decode(64)` — max-abs
6.096e-03 against a 1e-4 tolerance — which looked like tail/overlap state
depending on future frames. It is not.

- **Repeatability**: same input twice, `0.000e+00`. Nothing is jitter.
- **Decisive**: decode 64 frames, then 64 again with only `codes[32:]` changed
  — the prefix is **bit-identical**, at every length from 64 to 512.
- **Length sweep**: `decode(N/2)` vs `decode(N)` is zero everywhere *except*
  `N/2` in [32, 64]. The step is between 63 and 64 frames exactly.
- **Kernel swap**: replacing `ggml_flash_attn_ext` with explicit
  mul_mat/soft_max/mul_mat moves the failing lengths to 8/32/66/96 and changes
  the magnitudes. A structural leak would not care which kernel runs.

**Cause:** attention sums over the keys in an order set by how the kernel
blocks the row, so the sequence length changes the last bits; four
transposed-conv stages with snake activations amplify that to ~1e-3.

**The tell, reusable as a diagnostic:** the difference always starts at sample
1365 — exactly one frame. Frame 0 is bit-exact because its softmax has a single
term, and one term sums the same way in any order. Frame 1 is the first with
two. No state leak produces that boundary.

**Magnitude:** chunked vs one-shot is a steady **-55 dB** below the signal on
random codes (off-distribution, amplifies more); real speech is ~-67 dB. A
floor, not a drift.

The test now gates on the future-change comparison (must be bit-exact), judges
streaming against one-shot on error energy relative to the signal (-45 dB) plus
a peak-relative click gate, and reports repeatability. Reproduce:

```bash
QWEN3_TTS_TEST_VOCODER=/path/to/tokenizer.gguf ctest --test-dir build -R streaming
```

---

<a id="2"></a>
## 2. Voice cache is not invalidated across model variants

**Status:** fixed 2026-08-20
**Severity:** usability

Speaker embeddings are as wide as the talker's `hidden_size` (0.6B 1024, 1.7B
2048), but `cache.bin` was validated only against mtimes and the format
version — not the model that wrote it. Pointing the other variant at the
library failed every synth, with no way out but deleting every `cache.bin` by
hand.

`hdr.embedding_n` was already in the header; nothing read it. The old note
claimed the check needed the model loaded — wrong: the server already
snapshots `has_speaker_encoder` at load, so `VoiceStore` takes the width the
same way and `read_cache` compares a plain integer.

A mismatch is reported as *staleness*, so it falls into the existing re-encode
path. Every cache rejection now logs its reason.

**Cost of a switch:** the whole library re-encodes, ~10 s per voice (14 voices,
CUDA: 150 s). That is the audio tokenizer's conv encoder — the same conv1d wall
as the vocoder.

Per-width filenames (`cache-1024.bin` / `cache-2048.bin`) would make switching
free. Not done: switching is rare and one file per voice is simpler.

---

<a id="3"></a>
## 3. `--temperature 0` degenerates

**Status:** fixed 2026-08-20 (budget catches it, both front ends warn)
**Severity:** usability / trap

Greedy decoding is unstable on this model: it frequently runs to the cap and
emits near-silence. Independent of backend, reproduced on the pre-sweep binary.

Why it is worse than on a text LLM: the chosen codebook embeddings are summed
back into the talker's next step embedding, so the model listens to itself.
Without sampling noise, a state whose best continuation is "what I just did"
is mathematically inescapable — and near-silence is exactly such a state,
because the most predictable continuation of silence is more silence. Combined
with a weak EOS (#11), it never stops.

**Fixed as a trap, not as a mode.** The frame budget bounds the damage when it
degenerates, and both front ends warn on `temperature 0`, pointing at a low
temperature with a fixed seed for repeatable output. It does not fail every
greedy run — a short line often completes normally (verified: 36 frames, 200).
The failure is input- and seed-dependent, which is what made it confusing.

---

<a id="4"></a>
## 4. Vendored ggml could not build against CUDA 13

**Status:** fixed (ggml 0.9.11 → 0.20.2)
**Found:** 2026-08-19

`ggml/src/ggml-cuda/argsort.cu` gates `cuda::make_strided_iterator` on
`CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1`, which is true on CUDA
13.3 while the symbol is not actually available:

```
error: namespace "cuda" has no member "make_strided_iterator"
```

The CUDA backend therefore cannot be built against a current CUDA toolkit.
Resolved by moving the submodule to 0.20.2.

Note this is a *today* fact, not the history: this revision builds fine with
CUDA 12.x, which is what was installed when the fork's original CUDA
measurements were taken. The reason CUDA measured slower back then is issue
#10, not a broken build.

---

<a id="5"></a>
## 5. Two ggml Vulkan backend instances serialise on one device

**Status:** worked around (pipeline gated to CUDA/Metal)
**Found:** 2026-08-19
**Severity:** upstream behaviour, not a bug in this repo

Running the transformer and the vocoder from two threads, each owning its own
`ggml_backend_t` on the same Vulkan device, does not overlap — it collapses.
A 4-second line that takes ~2.5 s sequentially did not finish in 10 minutes.
The same arrangement on CUDA gives a 1.4x end-to-end win.

Worked around by enabling `decode_pipeline` only on CUDA/Metal
(`QWEN3_TTS_PIPELINE=1/0` overrides). Worth revisiting if ggml-vulkan grows
proper multi-queue support — the AMD card would benefit most.

**Coexistence is fine; only concurrency collapses.** Since 2026-08-24 the
decoder always takes its own backend instance (#13's follow-up), so two Vulkan
instances now exist on every run. That is harmless because the pipeline gate
keeps them from running at the same time — verified end to end on both RADV and
Nvidia's Vulkan driver, plus `streaming_parity_test`, which passes at −62.4 dB.

---

<a id="6"></a>
## 6. Stale architecture numbers in `AGENTS.md`

**Status:** fixed 2026-08-20
**Found:** 2026-08-19
**Severity:** docs

The "Model Architecture" section described the talker as "28-layer Qwen2 (1024
hidden ...)" — true only for the 0.6B. The 1.7B is `hidden_size=2048`,
`n_layers=28`, FFN 6144.

Worse, the code predictor was described as "same attention config" as the
talker. It is not: `code_predictor.embedding_length` is **1024 on both
variants**. That is the structural reason the 0.6B is not proportionally faster
end to end — the code predictor and the vocoder cost the same on both, only the
talker shrinks (see `optimization.md`).

Numbers now read off the ggufs directly, with the metadata keys listed so the
next person checks rather than trusts.

The prefill-embedding and special-token sections below it turned out to be
**correct** — verified against `build_prefill_graph()` and `tts_transformer.h`.
Added the two prefill variations they omitted (`nothink` path, ICL assembly)
and the missing language ids.

---

<a id="7"></a>
## 7. Codebook truncation breaks generation, not just fidelity

**Status:** wontfix 2026-08-20 — the dial works, the payoff does not justify it
**Severity:** design finding

`--codebooks N` truncates the RVQ chain and does cut the code predictor
proportionally (14.6 → 11.0 → 7.1 → 3.2 ms/frame at 16/12/8/4). The problem is
what it does to the *text*:

| codebooks | ASR of the output | audio length (43 s expected) |
|---|---|---|
| 16 | clean | 43.0 s |
| 12 | clean | 41.5 s |
| 8 | "беседно исчез 3-звучая иностранный пацан ик…" | 58.2 s |
| 4 | "из частной психиатрической клиники — personal Aloysius Sab!!!" | 48.6 s |

**Why**: every predicted codebook's embedding is summed back into the talker's
next step embedding. Zeroing the tail feeds the talker an input it never saw in
training, it drifts out of distribution, and the *text* degrades — a language
model failure, not a codec one.

**Measured ceiling** (0.6B, CUDA, pipeline on, 461-char line, fixed seed), per
frame because the two produce different amounts of audio:

| codebooks | code predictor | frames | total | per frame |
|---|---|---|---|---|
| 16 | 13.2 ms/frame | 313 | 20 459 ms | 65.4 ms |
| 12 | 9.7 ms/frame | 338 | 21 569 ms | 63.8 ms |

The predictor gets 3.5 ms/frame cheaper but **2 ms never reaches the clock** —
the generate/decode overlap was already hiding it. Net **2.4%** at the last
usable setting.

Making it work below 12 would need the talker fed something in-distribution for
the dropped codebooks (a learned or averaged placeholder, not zero).
Unvalidated, worth maybe 5%, and the vocoder is the wall anyway. Flag stays as
a research knob.

---

<a id="8"></a>
## 8. ggml Vulkan multi-add fusion is pathological on RADV / RDNA2

**Status:** worked around (env var), upstream bug
**Found:** 2026-08-19
**Severity:** major performance regression on AMD

Moving to ggml 0.20.2 made the Vulkan backend **3.3x slower on an RX 6800 XT**
(RADV, gfx1030) — and only there. Long clip, same input:

| build | RTF |
|---|---|
| ggml 0.9.11 (old) | 0.672 |
| ggml 0.20.2 | **2.203** |
| ggml 0.20.2 + `GGML_VK_DISABLE_FUSION=1` | 0.674 |
| ggml 0.20.2 + `GGML_VK_DISABLE_MULTI_ADD=1` | **0.649** |

So it is specifically the multi-add fusion, and disabling just that is enough —
it even edges out the old version. The vocoder sums 15 residual codebook
projections as a chain of `ggml_add` over very large tensors
(192000 x 96), which is exactly the pattern that fusion targets.

Not reproducible on NVIDIA's Vulkan driver (GTX 1660 SUPER: 6963 ms vs 6901 ms,
i.e. noise), so this is a RADV/RDNA2 path.

**Workaround:** `GGML_VK_DISABLE_MULTI_ADD=1` whenever the Vulkan backend runs
on an AMD card. Set in `Dockerfile.vulkan`. Harmless on NVIDIA.

**Worth reporting upstream** with the shape/dtype of the add chain.

---

<a id="9"></a>
## 9. ROCm is no longer the slow option on gfx1030

**Status:** resolved — recommendation changed
**Found:** 2026-08-19

The old measurement (HIP 1:43 vs Vulkan 0:45 on the same utterance) is stale.
With ROCm 7.2.4 and ggml 0.20.2 on the RX 6800 XT:

| backend | RTF |
|---|---|
| Vulkan, old ggml (what the fork shipped) | 0.672 |
| Vulkan, new ggml + `DISABLE_MULTI_ADD` | 0.649 |
| ROCm/HIP | 0.673 |
| **ROCm/HIP + generate/decode overlap** | **0.530** |

HIP has caught up with Vulkan on raw throughput, and unlike Vulkan it supports
overlapping the vocoder with generation (it is the CUDA code path, hipified, so
each backend instance gets its own stream). That overlap is what puts it ahead.
Arch's rocBLAS ships Tensile kernels for gfx1030, so GEMM is not on a generic
fallback despite AMD having dropped the card from official support.

**Recommendation for AMD: build HIP, not Vulkan.**

---

<a id="10"></a>
## 10. Why CUDA used to measure slower than Vulkan

**Status:** explained; fixed by the KV window
**Found:** 2026-08-19

The fork carried the belief that CUDA was much slower than Vulkan on the
1660 SUPER. The measurement was real; the cause was the KV-window bug (#see
`optimization.md`), which hurt CUDA more than Vulkan:

| build | generate |
|---|---|
| Vulkan, ggml 0.9.11 — the original baseline | 38.3 ms/frame |
| CUDA, before the KV window fix | 45.6 ms/frame |
| CUDA, after the KV window fix | 31.5 ms/frame |

Running attention over the whole 6413-row cache instead of the ~260 populated
rows degraded CUDA's flash-attention path (huge n_kv, single query) more than
Vulkan's. Fixing the window reversed the ranking. The backend was never broken
or unexercised — it was doing 25x the necessary attention work and handling
that waste worse than the other backend did.


---

<a id="11"></a>
## 11. Generation does not stop — runs to the 6144-frame cap on long text

**Status:** mitigated (guard + retry); root cause is upstream and unfixed
**Found:** 2026-08-19
**Severity:** availability (a request occupies the model for ~5 minutes)

The decode loop has exactly one stop condition (`tts_transformer.cpp:3083`):

```cpp
if (next_token == cfg.codec_eos_id) break;
```

There is no fallback. If the sampler never draws EOS, the loop runs to
`max_len` = `MAX_AUDIO_TOKENS` = 6144 frames = **491 s of audio**, which is
~5 min of GPU time on the 1660 SUPER and ~8 min on slower paths.

**Reproducer** (**stale — see below**): `ward_max.txt` (4067 chars) + 1.7B Base +
voice `ostro`, `--seed 6`. Rate over a seed sweep:

| text | chars | seeds | runaways |
|---|---|---|---|
| `ward.txt` | 727 | 40 | **0** |
| `ward_max.txt` | 4067 | 15 | **1** (seed 6) |

Length is the trigger, not the wording — the failure needs a long generation
to occur at all.

**The seed-6 reproducer no longer reproduces (checked 2026-08-25).** It now
ends normally at 2851 frames, inside the healthy 2701–3088 band. Two things
moved under it, both after this entry was written:

- `f569cb0` (*attend only to the populated KV window*, 19 Aug) changed the
  attention reduction order. Its own message says greedy decoding "can take a
  different (equally valid) path" — so a seed no longer lands on the same
  token trajectory it did when this was filed.
- The `ostro` voice cache (`cache.bin`) was regenerated on 24 Aug, which
  changes the speaker embedding the generation starts from.

The bug is not fixed — its reproducer is. Do not read a clean seed 6 as
evidence that #11 is gone; find a fresh runaway by sweeping seeds instead. And
treat *any* recorded seed here as valid only against the commit that recorded
it: sampling follows the reduction order, so a perf change to attention
reshuffles which seeds fail.

**Fresh reproducer** (valid against `ba68c83`, and only against it — see the
warning above): same text, model and voice, **`--seed 20`**. Sweep of seeds
7–21 on 2026-08-25:

| | frames |
|---|---|
| 14 healthy seeds | 2684–3132 |
| **seed 20** | **5820 = the budget (runaway)** |

One runaway in 15 seeds — the same 1-in-15 rate this entry recorded on 19 Aug,
so nothing about the failure's frequency has changed, only which seed hits it.

**It is not a numerical blowup (settled 2026-08-25).** The non-finite logit
guard from `06cd5fc` checked the talker's and the code predictor's logits on
every one of seed 20's 5820 frames and never fired. Combined with the F16 KV
cache having a 460x margin (K/V peak at 141.7 against a 65504 ceiling), the
overflow mechanism behind #16 is ruled out here.

Note what that does and does not say: it rules out inf/NaN — the overflow class
— not gradual precision loss that stays finite. But it does close the specific
question of whether #11 is #16 wearing a different hat. It is not, and the two
never looked alike: #16 is noise from the first second, #11 speaks the whole
text correctly and only then wanders.

**What the failure actually looks like** (not what was first assumed):

- The model reads the whole text correctly. ASR at 200 s is clean and matches
  the script; normal runs for this text end at 2701–3088 frames (216–247 s),
  and the runaway's speech ends in the same place.
- After that, output drops ~6 dB and stays there for the remaining 260 s.
  ASR returns Whisper's silence hallucination ("Субтитры создавал ..."), i.e.
  it is not speech.
- It is **not** a repeating loop. Autocorrelation of the 100 ms RMS envelope
  over 300–480 s peaks at 0.09 — noise. A phrase-repetition detector would
  not catch this.
- Character of the garbage vs. real speech: RMS -25.9 vs -19.6 dB, crest
  22.1 vs 15.2 dB, zero crossings 3776 vs 2810 /s. Sparse clicks and hiss,
  same peak level, much lower energy.

**Useful empirical constant:** frame count scales tightly with input length,
~**0.70 frames per character** of Russian text (727 chars → 508 frames;
4067 chars → ~2890 frames average). Good enough to derive a per-request
budget instead of the global 6144.

**Ruled out: the repetition penalty.** Re-running the same 15 seeds with
`--repetition-penalty 1.0` gave the same 1-in-15 rate (a different seed failed,
as expected from a different sampling stream). Separately worth knowing: the
penalty at `tts_transformer.cpp:3033` is applied over *every* CB0 token
generated so far, where HuggingFace uses a sliding window — on a 3000-frame
generation that flattens a large part of the codec vocabulary. A correctness
nit, not this bug.

**Ruled out: mixed script.** Upstream #318 blames Latin/digits inside another
script. 60 runs here across plain Russian / Latin proper nouns / digits / all
mixed, 12 seeds each: zero runaways. See #12 — that hunt found a bug in our
guard instead.

**Upstream:** this is a known defect of the model, not of this fork.
[QwenLM/Qwen3-TTS#118](https://github.com/QwenLM/Qwen3-TTS/issues/118) reports
the same symptom on 1.7B-Base and 0.6B-Base at default sampling, estimates
~0.5% of calls (1 in 200), finds no pattern in the triggering inputs, and was
closed as "not planned".
[Discussion #211](https://github.com/QwenLM/Qwen3-TTS/discussions/211) adds the
conditions that make it more likely — long input, reference audio over 20–30 s,
no explicit token cap — and describes the same truncated-ending symptom.
[#318](https://github.com/QwenLM/Qwen3-TTS/issues/318) is the extreme case:
mixed-script input (Thai with stray Latin/Cyrillic) hung 27 of 30 samples. The
recommendation everywhere is the same: cap the token count from a plausible
speech duration. So a guard on our side is the only available fix.

**What was done:**

1. `audio_token_budget()` (`qwen3_tts.h`) derives a per-request frame budget
   from the input: 1.4 frames per character, 8 per codepoint that is spoken as
   a whole word (digits, Han, kana, hangul — see #12), plus 128 frames of
   slack, clamped to `MAX_AUDIO_TOKENS`. Roughly 1.8x the
   worst case measured, because a false positive on a legitimately slow voice
   would be worse than a rare runaway running longer.
   `QWEN3_TTS_FRAME_BUDGET=0` disables it.
2. `tts_result::hit_token_budget` reports a generation that stopped on the
   budget rather than on EOS. That audio is not usable — the tail is noise and
   the end of the text is missing — so it is never reported as success.
3. The server retries once with a fresh sampling stream, then returns 500 with
   an explicit message. Live streaming cannot retry (bytes are already on the
   wire) and only logs a warning.
4. The CLI reports and exits non-zero; it deliberately does not retry, so the
   failure rate stays visible.

Measured effect on a 70-char line: budget 226 frames instead of 6144, so a
runaway costs ~9 s instead of ~5 min. On a 4067-char line the budget is 5821
frames — barely under the global cap, so there the retry is the whole fix.

**Side benefit:** the KV cache is sized from the budget
(`n_ctx = prefill + budget + 8`), so a short request now allocates n_ctx 494
instead of 6412 — most of a gigabyte of VRAM that a co-resident game gets to
keep.

**Still open:** the guard is length-based, so it fires late on long input. A
level-based detector would catch the collapse ~250 s in instead of ~466 s (the
runaway drops ~6 dB and stays there), but it only works where the vocoder runs
concurrently with generation — CUDA and ROCm, not Vulkan. Not implemented.

---

<a id="12"></a>
## 12. Frame budget counted a digit as a letter

**Status:** fixed 2026-08-20
**Severity:** correctness (rejected valid input)

The #11 guard budgeted 1.4 frames per letter, 8 per CJK codepoint. Digits fell
in the letter bucket. They should not: `25` is two characters spoken "двадцать
пять", the same way one han character is spoken as a whole word.

Measured on Russian NPC lines, 1.7B, cloned voice, 12 seeds each:

| line | chars | digits | frames | budget used |
|---|---|---|---|---|
| plain Russian | 145 | 0 | 103 | 31% |
| + Latin proper nouns | 161 | 0 | 114 | 32% |
| + a few numbers | 140 | 7 | 112 | 35% |
| Latin and numbers | 163 | 6 | 136 | 38% |
| inventory list | 159 | 27 | 224 | **64%** |

A letter costs 0.71 frames, a digit 2.5–4.8 — under-counted about fourfold.

**Not theoretical.** `Опись склада: 12, 47, 8, 56, ...` (171 chars, 78 digits)
got a 367-frame budget and needed 370–394: two of three seeds were cut off
mid-list and returned a 500. Fixed by moving digits (ASCII and Arabic-Indic)
into the CJK class; the line now budgets 882 and completes on all three seeds.
**Text without digits is unaffected by a single frame.**

Found while checking upstream #318's claim that mixed script raises the runaway
risk. It does not — zero runaways in 60 runs. The bug was ours.

---

---

<a id="13"></a>
## 13. The vocoder never ran on the GPU

**Status:** fixed 2026-08-24
**Found:** 2026-08-24
**Severity:** performance (3x on the dominant stage), and it invalidated the
profile everything else was planned around

`AudioTokenizerDecoder::load_model` asked for its weight buffer with
`GGML_BACKEND_DEVICE_TYPE_IGPU` alone (`audio_tokenizer_decoder.cpp:289`,
introduced upstream in `87a169b`). `load_tensor_data_from_file` had no ladder:
if the exact device class was missing it fell straight to the CPU. There is no
integrated GPU on a desktop with a discrete card, so the vocoder weights landed
in a CPU buffer — and `ggml_backend_sched` pins a node to the buffer its weights
live in, so **the entire vocoder graph ran on the CPU** while the log cheerfully
reported `AudioTokenizerDecoder backend: CUDA0`.

The tell was there all along and was misread as evidence of a GPU limit: the
vocoder took the same time on cards that differ 4x in compute, and reached
"12% of FP32 peak". Neither card was doing the work.

**Reproducer** (before the fix), CUDA build, 61 frames:

| `TTS_THREADS` | decode |
|---|---|
| 1 | 28 301 ms |
| 8 | 5 619 ms |

A 5x swing from CPU thread count is not something a GPU does.
`GGML_SCHED_DEBUG=2` confirmed it: `IM2COL`, `SIN`, `CONCAT`,
`CONV_TRANSPOSE_1D` and every `MUL_MAT` of the vocoder graph assigned to CPU.

**Fixed** by giving `load_tensor_data_from_file` the same IGPU → GPU → ACCEL →
CPU ladder `init_preferred_backend` already used, and honouring
`QWEN3_TTS_FORCE_CPU` there.

Two things surfaced only once the graph was actually on the GPU:

- **ICL warm-up had to be chunked.** It fed all ~150 reference frames to
  `stream_decode` in one call; that graph's im2col intermediates ask for
  ~1.5 GiB, which will not allocate next to the talker. It now uses the same
  batch the live path does. `stream_decode` is incremental and appends, so
  only the working set changed.
- **F16 accumulation** — see #14.

**Numbers** (RX 6800 XT, same seed, 68 frames both runs, `QWEN3_TTS_PIPELINE=0`):

| vocoder | decode |
|---|---|
| CPU (before) | 6 754 ms |
| GPU (after) | 1 979 ms |

`docs/ggml-notes.md` was rewritten: its conv1d section was built on the
mis-attributed profile above.

---

<a id="14"></a>
## 14. GPU vocoder accumulated the whole conv tower in F16

**Status:** fixed 2026-08-24
**Severity:** correctness (30 dB of accuracy, failed streaming parity)

Fixing #13 made `streaming_parity_test` fail. Every matmul in the vocoder has an
F16 left-hand side — `ggml_conv_1d` lowers to im2col + `mul_mat` and keeps the
im2col in F16, and the pre-transformer's weights are F16 in the GGUF. With an
F16 `src0` the CUDA/HIP backends select `CUBLAS_COMPUTE_16F`
(`ggml-cuda.cu:1394`), so a 25-layer convolution tower accumulates in half
precision.

This is not streaming-specific: the *one-shot* GPU decode already diverges from
the CPU reference. Same codes, dumped and compared outside the process
(`test_streaming_parity --dump`):

| backend | one-shot vs CPU | streaming vs one-shot |
|---|---|---|
| CPU | — | −55.5 dB (pass) |
| CUDA, F16 accum | 32.3 dB SNR | −30.0 dB (**fail**) |
| ROCm, F16 accum | 22.4 dB SNR | −24.8 dB (**fail**) |
| CUDA, F32 accum | 51.6 dB SNR | −50.0 dB (pass) |
| ROCm, F32 accum | 53.0 dB SNR | −61.3 dB (pass) |

Worst single sample before the fix: CPU `0.2278` vs ROCm `0.0048`. Not rounding.

**Fixed** by walking the finished vocoder graph and marking every `MUL_MAT`
`GGML_PREC_F32` (`force_f32_matmuls`, `audio_tokenizer_decoder.cpp`). Scoped to
this graph on purpose — F16 accumulation is the right default for the talker.
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` reproduces the same effect process-wide and
was how the cause was isolated.

**It is also faster.** On a 1660 SUPER (Turing TU116, no tensor cores) F16 GEMM
buys nothing and the F32 path is better optimised: decode 4 089 ms → 2 729 ms.
Do not assume this trade costs speed without measuring it on the target card.

**Ruled out:** the individual kernels. `test-backend-ops` passes for `IM2COL`,
`CONV_TRANSPOSE_1D`, `SIN`, `CONCAT`, `SQR` and `EXP` on ROCm — the ops are
correct, the accumulator type was not.

---

<a id="15"></a>
## 15. Voice cloning spent 40 s in a hand-rolled O(n²) DFT

**Status:** fixed 2026-08-24
**Severity:** performance (61x on the encode stage)

`compute_mel_spectrogram` in the speaker encoder called `compute_dft`, a naive
O(n²) transform with `cosf`/`sinf` in the inner loop, single-threaded. With
`n_fft = 1024`, hop 256 and a 60 s reference that is ~5600 frames x ~1M
iterations x 2 transcendentals: **~12 billion trig calls**.

It was mistaken for a backend problem at first. It is not: the giveaway is that
it did **not** scale with `TTS_THREADS` (41 s at 1 thread, 40 s at 8), which no
ggml CPU graph does. It was plain scalar C++ that no backend would ever touch.

Replaced with an iterative radix-2 Cooley-Tukey FFT (`compute_fft_radix2`);
`compute_spectrum` still falls back to the naive path for a non-power-of-two
`n_fft`. 60 s reference, 1660 SUPER host:

| | encode |
|---|---|
| naive DFT | 39 913 ms |
| radix-2 FFT | **657 ms** |

**Accuracy.** The FFT is the more accurate of the two — the naive version sums
n floats per bin and forms `k*t/n` in float. Magnitudes agree to 3.5e-5
relative; the resulting speaker embedding agrees to 3e-6 relative in L2 norm
and ~1.5e-5 per component. Comparing generated *audio* proves nothing here:
generation is autoregressive, so any perturbation changes the first sampled
token and the whole sequence diverges. Compare the embedding.

**The encoder had #13 and #14 too, and was fixed the same way.** Its weights
were CPU-resident (no device type passed, so the default won), and ECAPA-TDNN
is a conv tower over F16 weights, so the GPU path hit the same half-precision
accumulator. Measured against a CPU reference on a 60 s clip, embeddings of
1024 floats:

| speaker encoder | encode | cosine vs CPU | mean component error |
|---|---|---|---|
| CPU (weights in RAM) | 650 ms | — | — |
| GPU, F16 accumulator | 481 ms | 0.999999046 | 2.8e-4 |
| GPU, F32 accumulator | **478 ms** | **1.000000238** | **2.2e-5** |

F32 accumulation is ~12x more accurate here and costs nothing in time. The
weights are 16.9 MB, so the VRAM added is noise.

Note what the right metric is. Comparing generated *audio* proves nothing:
generation is autoregressive, so any perturbation changes the first sampled
token and the sequence diverges — a correct change and a broken one both look
like "completely different audio". For a speaker embedding the meaningful
question is direction, so compare cosine similarity, and compare the embedding
rather than anything downstream of it.

`force_f32_matmuls` moved to `gguf_loader.{h,cpp}` so both models call the same
helper. Talker graphs deliberately do not use it: F16 accumulation is the right
default for LLM-shaped matmuls, and only these deep conv towers compound it.

The Mimi codec encoder is still CPU-resident, and has no trig and no deep conv
tower — not investigated further.

---

<a id="16"></a>
## 16. An F16 talker runs away every time; Q8_0 and F32 do not

**Status:** fixed 2026-08-25 — `--type bf16` added, `--type f16` kept but warns
**Found:** 2026-08-24
**Severity:** correctness — `--type f16` produces an unusable talker

`scripts/convert_tts_to_gguf.py --type f16` produces a GGUF whose talker never
emits an end-of-speech token. Every generation runs to the frame budget, so
every request degenerates into the #11 failure — except that here it is not
rare, it is total.

**Measured** (0.6B Base converted from `Qwen/Qwen3-TTS-12Hz-0.6B-Base`,
`ward.txt`, ICL with the model card's 8.08 s `clone.wav`, CUDA on the 1660
SUPER):

| talker weights | seeds tried | runaways |
|---|---|---|
| Q8_0 (shipped file) | 10 | **0** |
| Q8_0 (converted here) | 2 | 0 |
| **F16** | **10** | **10** |
| F32 | 2 | 0 |

On the first sentence of `ward.txt` the contrast is exact and repeatable: Q8_0
stops at 128 and 135 frames on seeds 42/43, F32 at 125 and 155, and F16 hits
the 353-frame budget on both. #11 records 40 clean seeds on `ward.txt` with
Q8_0, so this is not the same phenomenon at a higher rate — it is a different
failure.

**Root cause (2026-08-25): one activation tensor does not fit in F16.**

It is not the weights — it is what the weights' *type* does to the activations.
In ggml the weight type picks the type the activation side is converted to
before the dot product (`ggml/src/ggml-cpu/ggml-cpu.c:224`: `vec_dot_type` of
`GGML_TYPE_F16` is `GGML_TYPE_F16`). So:

| weights | activation side becomes | ceiling |
|---|---|---|
| F16 | F16 | **65504** |
| Q8_0 | Q8_0, block-scaled int8 | none — the block scale absorbs any magnitude |
| F32 | F32 | none |

F16 is the only one of the three with a ceiling, which is exactly why only F16
fails, and why it fails identically on CPU and on CUDA.

**The measurement.** `QWEN3_TTS_PROBE_NUM=1` scans every graph node's output for
`max|x|` and non-finite values. On the working Q8_0 file, over the whole
generation, **exactly one node in the entire pipeline exceeds 65504**:

```
max|x|=185587.2  nonfinite=0  n=129  MUL  cp_prefill.blk.2.ffn_swiglu [3072,2,1]
```

That is the SwiGLU output (`silu(gate) * up`) of **code-predictor layer 2**,
2.83x above the F16 ceiling, hit once per frame on every frame. Its consumer is
`code_pred.blk.2.ffn_down`. On the F16 file the probe names that consumer as
the first non-finite node in the graph, on the very first frame:

```
[probe] FIRST non-finite output: MUL_MAT node_127 [1024,2]  (1024 of 2048 values)
```

Same node, Q8_0: 63417.7, finite. So the code predictor emits inf on frame 1,
every frame, and the sampler never draws EOS.

**The output was never speech.** The CLI refuses to save a runaway
(`qwen3_tts.cpp`), which is why this looked like "speech that will not stop"
for a day — nobody had heard it. With `QWEN3_TTS_KEEP_RUNAWAY=1`:

| | RMS | zero-crossing rate |
|---|---|---|
| Q8_0 | −24.6 dB | 0.130 |
| F16 | −34.5 dB | 0.332 |

Noise from the first second, not speech. This is **not** #11 at a higher rate,
and the earlier "it degenerates into #11" reading was wrong.

**Confirmed by flipping it.** Converting `--type f16` with
`code_pred.blk.2.ffn_down` alone forced to F32 — 1 tensor of 478 — ends the
runaway completely: 10/10 seeds terminate (124–142 frames, against Q8_0's
128/135), and the audio matches Q8_0's character (RMS −24.9, ZCR 0.133).

**Corroboration from upstream.** The reference PyTorch pipeline has the same
failure in the same place: it runs bf16 only, because in fp16 *its* code
predictor emits NaN logits and `torch.multinomial` dies. This is a property of
the model, not of our port.

**What the earlier notes got wrong:** "Not F16 range — every 2D+ tensor was
scanned" checked the *weights*. The weights are all in range. The activations
are not, and only F16 weights drag the activations through F16.

**The fix: `--type bf16`.** bf16 spends its 16 bits differently — 8 mantissa
bits instead of F16's 11, but F32's full exponent range, so there is no ceiling
to hit. 185587 lands on 185344 instead of inf.

The per-tensor alternative (keep the offending weights out of F16) was rejected:
the residual stream measures 63554, **97% of the F16 ceiling**, so a different
voice, text or model size can push another node over. Only the 0.6B on one
sentence was ever probed.

**It is not a conversion at all.** Every one of the checkpoint's 478 tensors is
already bf16, so `--type bf16` copies the bits rather than reinterpreting them;
spot-checked bit-exact against the safetensors for talker, code-predictor and
attention weights. We were the only step in the chain converting anything, which
is why we were the only step that broke.

**Verified**: 10/10 seeds terminate normally (119–149 frames, against Q8_0's
128/135), audio matches Q8_0's character (RMS −25.6 / ZCR 0.142 against −24.6 /
0.130), no guard trips on either backend.

**Speed** (0.6B, seed 42, median of 3, ms/frame — the f16 row is a proxy file
with 477 of 478 tensors F16, since a real F16 file now fails at frame 0):

| type | CUDA (1660 SUPER) | ROCm (6800 XT) |
|---|---|---|
| **q8_0** | **25.0** | **24.2** |
| bf16 | 34.4 | 27.0 |
| f16 (broken) | 37.0 | 24.6 |
| f32 | 38.0 | — |

Q8_0 stays the default and stays fastest on both cards. Against F16 the two
backends disagree — bf16 is 7% faster on CUDA (the #14 effect: dodging F16 pays
on a card with no tensor cores) and 10% slower on ROCm. Either way bf16 is the
only half-precision type here that is correct, and it is the one to use as the
precision control in `optimization.md` instead of F32.

**Practical effect today:** none — every shipped GGUF is Q8_0. It matters the
moment anyone converts with `--type f16`, and it blocks using F16 as the
precision control when attributing speed wins to quantisation (see
`optimization.md`). F32 works and can stand in, at 2x the weight bytes.

<a id="17"></a>
## 17. A vocoder OOM segfaults instead of failing — upstream ggml

**Status:** fixed in our ggml fork, branch `qwen3-tts`; **deliberately not
reported upstream** — Oleg's call, 2026-08-24, revisit alongside the kernel work
**Found:** 2026-08-24
**Severity:** availability — an out-of-memory kills the process, server included

`QWEN3_TTS_DECODE_BATCH=400` on the 1660 SUPER needs a 4089 MiB compute buffer,
which does not fit. Instead of the clean error our code is ready to return, the
process died with SIGSEGV:

```
#0  ggml_gallocr_alloc_graph ()
#1  ggml_backend_sched_alloc_graph ()
#2  qwen3_tts::AudioTokenizerDecoder::stream_decode(...)
```

**Root cause**, `ggml/src/ggml-backend.cpp` in
`ggml_backend_sched_alloc_splits()`:

```c
ggml_gallocr_reserve_n(sched->galloc, &sched->graph, ...);   // result discarded
if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
```

`ggml_gallocr_reserve_n` has already rewritten `galloc->node_allocs` to point
into buffers by the time the allocation fails, and on failure it leaves those
buffers NULL. Discarding its return value means `ggml_gallocr_alloc_graph` then
walks the stale assignments and dereferences NULL. Our own check on
`ggml_backend_sched_alloc_graph` (`audio_tokenizer_decoder.cpp:1167`) is
correct; it simply never gets to run.

**Fix** — check the return value:

```c
if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph,
                            sched->node_backend_ids, sched->leaf_backend_ids)) {
    GGML_LOG_ERROR("%s: failed to reserve graph\n", __func__);
    return false;
}
```

Verified: the same run now exits 1 with `Failed to decode speech codes: Failed
to allocate streaming graph`.

Reproduce with any `QWEN3_TTS_DECODE_BATCH` large enough to exhaust the card.

The patch lives in the fork as commit `f0aeec2` on `qwen3-tts`, so a plain
`git submodule update` now restores it rather than removing it. It used to be
an uncommitted working-tree edit, which is why the diff is written out in full
above; that copy is now redundant and kept only because it reads well here.

Worth noting for the eventual upstream report: `ggml_backend_sched_reserve()`
in the same file already checks this exact call. The patch makes the two paths
agree rather than introducing a new convention.

Reporting it upstream was considered and deferred on 2026-08-24: worth doing
together with the `CONV_TRANSPOSE_1D` work, which touches the same project, so
there is one conversation rather than two.

---

<a id="rough-edges"></a>
## Open rough edges

Not bugs, and none of them bite today. Kept so they are not rediscovered from
scratch. Each was verified against the code on 2026-08-24.

- **Sampling logic exists twice.** `predict_codes_autoregressive`'s
  `sample_or_argmax` lambda (`tts_transformer.cpp`) against the unrolled loop in
  `generate()`. They have already diverged: the suppression window and the
  repetition penalty are only in `generate()`. A change to sampling rules has to
  be made in both places and can silently be made in one.
- **17 `QWEN3_TTS_*` env vars read by `getenv()` from 8 files**, six documented.
  Several (`DUMP_CODES`, `DUMP_LOGITS`, `DUMP_STAGES`, `DUMP_FEATURES`) are
  diagnostics branched inline in hot loops. Tolerable; worth a config struct
  before the next handful arrives.
- **No fast-fail when busy.** Synthesis is serialized and a second request just
  waits. `synth_mutex.try_lock()` → 429 with `Retry-After` would stop clients
  queueing. `/health` already reports `busy`.
- **The C ABI (`qwen3tts_c_api`) lags the core**: no ICL/`ref_codes`, no
  streaming, and a different `max_audio_tokens` default. `top_p` is present but
  deliberately unused (kept for ABI stability, documented at the declaration —
  that part is fine). Either catch it up or mark the target experimental.
- **`examples/` holds two wavs, no example source, and one of the wavs is a
  trap.** `readme_clone_input.wav` is **60 s** long while `reference_text.txt`
  next to it transcribes about eight seconds of speech. Nothing in the README
  references either file any more. Feeding that pair to the reference PyTorch
  pipeline as an ICL prompt makes generation run to the token cap (655 s of
  audio) — it cost real time on 2026-08-24 before the mismatch was spotted.
  Replace it with the model card's `clone.wav`, which is what the transcript
  actually matches, or delete both and the stale transcript with them.
- **`docs/model_inspection.txt`** is a raw metadata dump with no header saying
  which GGUF it came from or when.
- **Two ggml findings are sitting unreported.** #8 (RADV multi-add fusion) is
  written up and reproducible; #17 (OOM segfault) is a one-line fix with a
  verified repro. Both are upstream ggml, and #17 is deliberately held until
  the `CONV_TRANSPOSE_1D` work goes the same way.

### Dismissed after checking

- **"Repetition penalty diverges from HuggingFace — it should use a sliding
  window."** It should not. HF's `RepetitionPenaltyLogitsProcessor` applies to
  the whole sequence, which is what `tts_transformer.cpp` already does; the
  sliding window (`repeat_last_n`) is llama.cpp's. Adding one would move *away*
  from the reference. #11 also rules the penalty out as a runaway cause by
  measurement. Raised by an external review, recorded here because the same
  suggestion looks plausible on a quick read.

<a id="18"></a>
## 18. The converter's default output type is the one that does not work

**Status:** open
**Found:** 2026-08-25
**Severity:** correctness — the no-argument path produces an unusable model

`scripts/convert_tts_to_gguf.py` declares `--type` with `default="f16"`, and
its own usage example in the module docstring passes `--type f16`. Per #16 an
F16 talker never emits end-of-speech and generates noise from the first frame.

So the obvious way to run the converter — no `--type` at all, or copying the
example — silently produces a broken GGUF. Everything shipped is Q8_0 only
because whoever converted those files happened to pass `--type q8_0`.

This is the answer to "where would a bad GGUF even come from": from here, by
default.

**Fixed 2026-08-25.** `default="q8_0"` — what every shipped file uses and the
fastest option on both cards. The docstring example follows it, `--type f16` now
prints a warning naming #16 before it converts anything, and the help text marks
f16 as broken. f16 is kept rather than removed so that anyone reproducing #16
still can.

<a id="19"></a>
## 19. Nothing checks the vocoder's output for non-finite samples

**Status:** fixed 2026-08-25
**Found:** 2026-08-25
**Severity:** robustness — a numerically broken vocoder fails silently

`06cd5fc` guards the talker and the code predictor: their logits are checked
for non-finite values on every frame. The vocoder has no such check anywhere in
`audio_tokenizer_decoder.cpp` — and #14 was precisely a numerical failure of
the vocoder, caught only because someone measured SNR by hand.

If the conv tower goes non-finite again the samples reach the encoder as-is,
and the user gets silence or noise with a clean log.

**Fixed.** Both sample-output sites in `audio_tokenizer_decoder.cpp` now scan
the decoded block and fail with the stage named, matching the logit guard's
reporting style. One scan is ~675k floats for a 28 s clip against ~495 ms of
decode.

**Verified by injecting an inf** into the decoded samples: the streaming path —
the one production actually uses — reports it and aborts the request. Note what
that showed: the failure surfaced during the *vocoder warm-up on the reference
codes*, before any generated audio, which is the earliest point it could.

`decode()`, the one-shot path, is called only from `tests/`; production always
goes through `stream_decode` (chunked decode is mandatory, ~15 MiB/frame). Its
guard is the same three lines but was not exercised: `test_decoder` cannot run
here at all, because it needs `reference/speech_codes.bin` and
`reference/decoded_audio.bin`, which are not in the repository. That gap
predates this change and is worth its own entry.
