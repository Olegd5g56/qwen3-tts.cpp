# Qwen3-TTS GGML Optimization Report

Performance characterization of this fork. Last updated **2026-06-10** after the
June perf sweep (`d544a6f`..`b031dd7` + ICL warm-up cache).

Historical note: the original upstream version of this document described a
CPU-only baseline (RTF 1.94x, speaker encoder as the main bottleneck, one-shot
vocoder). All of that is obsolete — the speaker encode result is cached by the
voice store, the vocoder streams in chunks, and Vulkan is the primary backend.

## Summary

**Test configuration:** AMD RX 6800 XT (Vulkan, RADV), Ryzen 7 5700X,
Q8_0 1.7B Base talker + F16 vocoder, ~500-frame (~40 s) Russian clip,
ICL cloned voice.

| Metric | Before sweep | After sweep |
|--------|--------------|-------------|
| End-to-end (CLI, long clip) | 85.5 s | 28.5 s |
| RTF (lower is better) | 1.96 | **0.70** (1.4x realtime) |
| Streaming TTFA, cloned voice (cold) | ~7 s | ~7 s |
| Streaming TTFA, cloned voice (warm cache) | ~7 s | **~1.5 s** |
| MP3 size (speech-tuned VBR) | — | ~15% smaller |

## What was done (June 2026)

1. **`d544a6f` — removed `ffn_down` F32 cast** in all five hot transformer
   graphs. The cast forced a full dequant of the largest FFN matrix on every
   step; `ggml_mul_mat` handles quantized weights natively. ~5.9x on generate.
2. **`79defa7` — LAME speech settings** (VBR -V 4, `quality=5` instead of
   music-grade -V 2 / `quality=2`). Faster encode, ~15% smaller files, no
   audible difference for 24 kHz mono speech.
3. **`d8b49be` — broadcast `ggml_mul` instead of `ggml_repeat`** in snake
   activation and upsample gamma. Kills large scratch tensors in the vocoder
   tower. ~14% on decode.
4. **`b031dd7` — `ggml_flash_attn_ext`** in the vocoder pre-transformer
   attention (F16 mask, padded to 64). ~3% on decode.
5. **ICL warm-up cache** (`Qwen3TTS::warmup_decoder_for_icl`). Cloned-voice
   requests must run the ~150 reference frames through the streaming vocoder
   before synthesis. The decoder's streaming state is entirely host-side
   (per-layer KV vectors, causal-conv tail rings, conv_transpose overlap
   buffers — see `AudioTokenizerDecoder::stream_state`), so after the first
   request the snapshot (~10 MB) is cached process-wide, keyed by FNV-1a over
   ref-codes + vocoder path. Restore is a memcpy. The cache is a function-local
   static because the server destroys the `Qwen3TTS` object on idle unload;
   capped at 16 entries.

## Current profile (where the time goes)

Per `QWEN3_TTS_TIMING` instrumentation on the 506-frame clip:

- **Vocoder decode: ~64% of total.** Honest GPU compute in the conv upsampling
  tower (12.5 Hz latent → 24 kHz PCM, 480x). Runs as chunked `stream_decode`
  (100-frame batches) — larger batches were tested and are *slower* with much
  higher RSS.
- **Generation: ~36% of total.** Breakdown:
  - Code predictor ~57% of generate — 15 sequential graph dispatches per frame
    (1 prefill + 14 codebook steps); ~73% of it is real compute.
  - Talker forward ~35%, embed lookups ~4%, prefill ~3.5%.

## Tried and ruled out — do not redo

- **Persistent attention mask** (incremental upload instead of full re-upload
  per step): no measurable win. PCIe latency dominates bandwidth at these
  sizes; small `ggml_backend_tensor_set` calls cost the same as big ones.
- **Decode batch 100 → 200/400/800**: slower, RSS ballooned (2.2 → 6.7 GB at
  800). Per-chunk overhead is not the decode bottleneck.
- **Embedding lookup via host scratch**: embed lookups are only 4% of
  generate; codec/code-pred embeddings are F16 so the read is already direct.
- **One-shot vocoder `decode()` for long clips**: activation memory balloons
  past 13 GB for ~3k frames. Chunked streaming decode is the only sane path.

## Remaining ideas (architectural, descending value)

1. **Persistent GPU-side KV in vocoder pre-tfm layers** — currently each
   streaming chunk round-trips the full accumulated KV host↔GPU
   (`stream_decode` driver). Estimated total cost is small (tens of ms);
   invasive change, likely not worth it.
2. **Code predictor graph caching across steps** — ~12% of code-pred time is
   build+alloc; the talker step graph (`build_step_graph`) shows the
   n_past-as-input pattern to copy. Saves well under 1 s per long clip.
3. **Code predictor codebook batching** — would attack the 15-dispatch-per-
   frame structure itself, but changes autoregressive semantics; research
   project, not a tweak.

Current state is in the "good enough for live use" zone: 1.4x realtime
end-to-end and ~1.5 s TTFA on warm cloned voices.
