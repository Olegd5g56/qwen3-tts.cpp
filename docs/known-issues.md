# Known issues

Running log of bugs and rough edges found while working on this fork. Entries
stay here until fixed or deliberately dismissed — the point is that nothing
found in passing gets lost.

Status: **open** (not fixed) / **fixed** / **mitigated** (symptom addressed,
root cause still there) / **wontfix**.

---

## 1. Streaming decoder is not fully causal — `test_streaming_parity` fails

**Status:** closed 2026-08-20 — not a bug; the test was asking the wrong question
**Found:** 2026-08-19
**Severity:** none (was: correctness)

The original complaint: `decode(32)` did not reproduce the first 32 frames of
`decode(64)` — max-abs 6.096e-03 against a 1e-4 tolerance, first deviation at
sample 4396 — which looked like the decoder's tail/overlap state depending on
future frames.

**It does not.** Four measurements, all on the vocoder alone:

1. **Repeatability.** Same input twice → `0.000e+00`. The backend is
   deterministic, so nothing here is run-to-run jitter.
2. **The decisive one — change the future, hold the length.** Decode 64 frames,
   then decode 64 frames again with only `codes[32:]` replaced. The samples the
   first 32 frames are responsible for come back **bit-identical**
   (`0.000e+00`). The future cannot reach the past. That is causality, tested
   directly, and it is clean at every length from 64 to 512 frames.
3. **Sweep over lengths.** Comparing `decode(N/2)` against `decode(N)`, every
   pair is exactly zero *except* when `N/2` falls in [32, 64]. The step sits
   precisely between 63 and 64 frames — `decode(N)[4396]` is `0.05063223` for
   N ≤ 63 and `0.05052793` for N ≥ 64, with nothing in between.
4. **Swap the attention implementation.** Replacing `ggml_flash_attn_ext` with
   the explicit mul_mat/soft_max/mul_mat path moves the failing lengths
   somewhere else entirely (8, 32, 66, 96 instead of 32–64) and changes the
   magnitudes. A structural leak would not care which kernel computes it.

**What it actually is.** Attention sums over the keys, and the order it sums
them in depends on how the kernel blocks a row — which depends on the number of
keys, i.e. the sequence length. Different order, different last bits. The conv
tower downstream then amplifies that: four transposed-conv stages with snake
activations turn a 1e-7 perturbation into ~1e-3 at the output.

The giveaway is *where* the difference starts: sample 1365, which is exactly
one frame. **Frame 0 is always bit-exact** because its softmax has a single
term — one term sums the same way in any order. Frame 1 is the first with two
terms, and it is the first to differ. Nothing about a state leak would produce
that boundary.

**Magnitude.** Chunked decode sits a steady **-55 dB** below the signal across
64–512 frames, on random codes — which drive the vocoder far off-distribution.
Real utterances measure about -67 dB (correlation 0.9999999, identical RMS).
Inaudible either way, and it is a floor, not a drift: it does not grow with
length.

**What changed.** `test_streaming_parity` now:
- makes the future-change comparison the strict causality gate (must be
  bit-exact);
- keeps the different-length comparison but labels it informational, because
  two decodes of different lengths can never be bit-identical for the reason
  above;
- judges streaming against one-shot on error energy relative to the signal
  (-45 dB limit, ~10 dB of headroom over the measured floor) plus a
  peak-relative click gate, instead of an absolute per-sample tolerance that
  means nothing on a nonlinear generator;
- reports repeatability, so a non-deterministic backend cannot masquerade as a
  correctness failure.

Reproduce:

```bash
QWEN3_TTS_TEST_VOCODER=/path/to/tokenizer.gguf ctest --test-dir build -R streaming
```

---

## 2. Voice cache is not invalidated across model variants

**Status:** fixed 2026-08-20
**Found:** 2026-08-19
**Severity:** usability

Speaker embeddings are as wide as the talker's `hidden_size` (0.6B → 1024,
1.7B → 2048), but `cache.bin` was only validated against the sample's mtime and
the cache format version — not against the model it was encoded with. Pointing
a 0.6B model at a voices directory populated by the 1.7B one failed the
synthesis outright, with no way out except deleting `cache.bin` in every voice
folder by hand — and again on the way back.

`hdr.embedding_n` was already written into the header; nothing read it. The
original note said the check was impossible because it needs the model loaded,
which would force a lazy load in a path meant to be cheap. That was wrong: the
server already snapshots `has_speaker_encoder` at load time for exactly this
reason, so `VoiceStore` now takes the width the same way and `read_cache`
compares against a plain integer, no model involved.

A mismatch is reported as staleness, not as an error, so it falls into the
existing re-encode path and the fresh cache overwrites the old one. Every cache
rejection now logs its reason — otherwise a variant switch just looks like a
mysteriously slow startup.

**Cost of a switch:** the whole library re-encodes, ~10 s per voice (measured
2026-08-20, CUDA, 14 voices, references 5–15 s long — 150 s total). That is the
audio tokenizer's conv encoder, the same conv1d wall as the vocoder (see
`ggml-notes.md`), so it will not get cheaper on its own.

Naming the cache per width instead (`cache-1024.bin` / `cache-2048.bin`) would
make switching free because both variants' caches would coexist. Not done:
switching variants is rare, and one file per voice is simpler to reason about
than two. Worth revisiting only if variant A/B testing becomes routine.

## 3. `--temperature 0` degenerates

**Status:** fixed 2026-08-20 (budget catches it, both front ends warn)
**Found:** 2026-08-19
**Severity:** usability / trap

Greedy decoding is unstable on this model: it frequently runs to the 6144-frame
cap and emits near-silence. Confirmed independent of backend and of the August
changes (reproduced on the unmodified pre-sweep Vulkan binary, same input).

The option is documented and accepted, so a user reaching for "deterministic
output" gets garbage plus a ~490-second wav. It also makes greedy useless as a
benchmarking mode, which cost real debugging time — two apparent regressions
during the sweep were this.

**Options:** detect codebook-0 repetition and break, warn when `temperature 0`
is combined with a long input, or document it loudly.

**Fixed 2026-08-20**, in two halves — as a trap, not as a mode.

The per-request frame budget added for issue 11 bounds the damage: *when*
greedy degenerates it now stops at the budget and returns an error, instead of
quietly handing back ~490 seconds of near-silence. It does not fail every
greedy run — a short line often completes normally (verified: "Проверка жадного
декодирования." at `temperature: 0` returned 200 in 36 frames). The failure is
input- and seed-dependent, which is exactly what made it confusing.

So both front ends also warn whenever `temperature 0` is selected — the CLI on
stderr before loading, the server per request — pointing at a low temperature
with a fixed seed as the way to get repeatable output. The `--temperature` help
text says the same.

Greedy remains a bad mode on this model. It is no longer a silent one.

---

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

---

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

## 7. Codebook truncation breaks generation, not just fidelity

**Status:** wontfix 2026-08-20 — the dial works, the payoff does not justify it
**Found:** 2026-08-19
**Severity:** design finding

Truncating the RVQ chain was supposed to be the tunable speed/quality dial.
It is implemented (`--codebooks N` / `TTS_CODEBOOKS`) and the cost scales
exactly as predicted — code predictor 14.6 → 11.0 → 7.1 → 3.2 ms/frame at
16/12/8/4 codebooks. The problem is what it does to the *text*:

| codebooks | ASR of the output | audio length (43 s expected) |
|---|---|---|
| 16 | clean | 43.0 s |
| 12 | clean | 41.5 s |
| 8 | "беседно исчез 3-звучая иностранный пацан ик… ЧЕЛИЧМАЛЙЧНИ…" | 58.2 s |
| 4 | "из частной психиатрической клиники — personal Aloysius Sab!!!" | 48.6 s |

The 0.6B model at 8 codebooks read the opening correctly and then ran to the
6144-frame cap: **491 s of audio for a 43 s script**.

**Why**: the codebooks are not only a vocoder input. Every predicted codebook's
embedding is summed back into the talker's next step embedding
(`step_embd += code_pred_embd[cb][code]`). Zeroing the tail feeds the talker an
input it never saw in training, it drifts out of distribution, and the *text*
degrades — this is a language-model failure, not a codec one.

So the dial does not trade fidelity for speed; below ~12 it trades coherence
for speed, which is not a trade anyone wants. And at 12 the saving is ~3% RTF.

**If revisited:** truncating only on the vocoder side keeps the talker sane but
saves nothing (the VQ sum is not the expensive part — the 15 sequential
code-predictor passes are). Making this work would need the talker fed
something in-distribution for the dropped codebooks, e.g. a learned or averaged
placeholder embedding rather than zero. Unvalidated idea.

**Measured the ceiling 2026-08-20** (0.6B, CUDA, pipeline on, 461-char line,
cloned voice, seed fixed). Per frame, because the two configurations produce
different amounts of audio:

| codebooks | code predictor | frames | total | per frame |
|---|---|---|---|---|
| 16 | 13.2 ms/frame | 313 | 20 459 ms | 65.4 ms |
| 12 | 9.7 ms/frame | 338 | 21 569 ms | 63.8 ms |

The code predictor really does get 3.5 ms/frame cheaper, and **2 ms of that
never reaches the clock** — the generate/decode overlap was already hiding it.
Net: **2.4%** at the last setting that still produces clean speech. Below 12 the
text falls apart for the reason above, so that 2.4% is the whole prize.

Closing as wontfix. The placeholder-embedding idea might unlock 8 codebooks and
perhaps 5%, but it is unvalidated, it risks quality on a dial nobody should be
reaching for, and the vocoder — not the code predictor — is the wall
(`optimization.md`). The flag stays: it is honest, documented, and useful for
exactly this kind of measurement.

---

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

**Reproducer** (deterministic): `ward_max.txt` (4067 chars) + 1.7B Base +
voice `ostro`, `--seed 6`. Rate over a seed sweep:

| text | chars | seeds | runaways |
|---|---|---|---|
| `ward.txt` | 727 | 40 | **0** |
| `ward_max.txt` | 4067 | 15 | **1** (seed 6) |

Length is the trigger, not the wording — the failure needs a long generation
to occur at all.

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

**Ruled out:** the repetition penalty is not the cause. Re-running the same 15
seeds with `--repetition-penalty 1.0` gave the same 1-in-15 rate (a different
seed failed, as expected from a different sampling stream). The observation
below still stands as a correctness nit, but it does not drive this bug.

**Divergence from HuggingFace:** the repetition penalty at
`tts_transformer.cpp:3033` is applied over *every* CB0 token generated so far
(`generated_cb0_tokens` grows monotonically for the whole utterance), where
HuggingFace applies it over a sliding window. On a 3000-frame generation this
penalises a large fraction of the codec vocabulary and flattens the
distribution. Worth an A/B at `--repetition-penalty 1.0` on the same seeds.

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
   from the input: 1.4 frames per character, 8 per Han/kana/hangul codepoint,
   plus 128 frames of slack, clamped to `MAX_AUDIO_TOKENS`. Roughly 1.8x the
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

**Candidate fixes**, best first:

1. **Split long input into sentences** and generate per chunk. The failure
   needs length; `ward.txt` at 43 s never failed in 40 tries. Also caps the
   blast radius — a bad chunk costs seconds, not minutes.
2. **Per-request frame budget** from input length (~1.4x of 0.70/char) instead
   of the global cap. Blunt but always applies; would have cut seed 6 at
   ~4000 frames instead of 6144.
3. **Level-based stop.** The pipeline already decodes audio in chunks while
   generating. Track the utterance's running level; stop after several
   consecutive seconds far below it. Would have cut seed 6 at ~250 s.
4. **Server-side retry** with a different seed when a guard fires — for
   interactive use (game dialogue) this is the difference between a missing
   line and a slightly late one.

**Also found:** the CLI never sets `params.print_progress`, so the
`decode: frame N/6144` progress lines are dead there. The server prints them
under `TTS_VERBOSE=1`. Nothing else uses the flag, so CLI users have no way to
see a runaway in progress.
