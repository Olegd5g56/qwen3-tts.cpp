# Known issues

Running log of bugs and rough edges found while working on this fork. Entries
stay here until fixed or deliberately dismissed — the point is that nothing
found in passing gets lost.

Status: **open** (not fixed) / **fixed** / **mitigated** (symptom addressed,
root cause still there) / **wontfix**.

---

## 1. Streaming decoder is not fully causal — `test_streaming_parity` fails

**Status:** open
**Found:** 2026-08-19
**Severity:** correctness (audio), currently inaudible

`test_streaming_parity --tokenizer <vocoder.gguf>` fails, and has been failing
since before the August sweep — verified by running the pre-change binary
(`build-vulkan/`), which produces a bit-identical deviation:

```
FAIL: max-abs-diff 6.096e-03 exceeds tol 1.000e-04
causality: decode(32)[0:60885] vs decode(64)[0:60885] max_diff=6.096e-03
  first deviation at 4396: half=0.050632 full=0.050528
```

The interesting half is the **causality** line, which does not involve the
streaming driver at all: decoding 32 frames does not produce the same samples
as the first 32 frames of a 64-frame decode. Something in the decoder's tail /
overlap state depends on future frames, or a state ring is seeded differently
depending on chunk length.

Practical impact looks nil so far — pipelined vs sequential full-utterance
output matches at corr 0.9999999 with identical RMS, and no seam is audible.
So either the deviation is benign numerical drift and the 1e-4 tolerance is
unrealistic, or there is a real (small) state leak worth finding.

**To investigate:** narrow down which stage diverges — `stream_state` has
per-layer KV vectors, causal-conv tail rings, and conv_transpose overlap
buffers. Bisect by dumping intermediate activations at chunk boundaries.

---

## 2. Voice cache is not invalidated across model variants

**Status:** mitigated (error message only)
**Found:** 2026-08-19
**Severity:** usability

Speaker embeddings are as wide as the talker's `hidden_size` (0.6B → 1024,
1.7B → 2048), but `cache.bin` is only validated against the sample's mtime and
the cache format version — not against the model it was encoded with. Pointing
a 0.6B model at a voices directory populated by the 1.7B one fails the
synthesis outright.

Fixed the message to explain the cause and the workaround (separate voices dir
per variant, or delete `cache.bin` to re-encode). The correct fix is for
`VoiceStore::get_or_load` to treat a width mismatch as a stale cache, the same
way an mtime mismatch is treated, and re-encode. Not done because the check
needs the model loaded, which means taking `synth_mutex_` and forcing a lazy
load in a path that is meant to be cheap.

---

## 3. `--temperature 0` degenerates

**Status:** open
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

**Status:** open
**Found:** 2026-08-19
**Severity:** docs

The "Model Architecture" section describes the talker as "28-layer Qwen2 (1024
hidden ...)". The 1.7B model actually loads as `hidden_size=2048, n_layers=28`.
The prefill-embedding and special-token sections below it were not re-checked
against the current model.

---

## 7. Codebook truncation breaks generation, not just fidelity

**Status:** open (feature shipped behind `--codebooks`, but its useful range is tiny)
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

