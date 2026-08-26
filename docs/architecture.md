# Architecture

The whole machine in one read: what happens to the data between a string of
text and a buffer of PCM, which device each piece runs on, and what is kept
between requests.

This file is the map. It does not carry measurements (`optimization.md`), bugs
(`known-issues.md`), ggml-side findings (`ggml-notes.md`), the streaming
derivation (`streaming_design.md`), or weight-type advice
(`quantisation.md`). `AGENTS.md` covers file layout and coding conventions;
this covers the data path.

---

## One request, end to end

```
text ──> text tokenizer ──┐
                          │
reference audio ──> speaker encoder (ECAPA-TDNN) ──> embedding [hidden_size]
        └────────> codec encoder (Mimi) ──────────> ref codes [n_ref, 16]
                          │
                          v
                 ┌────────────────────┐
                 │ talker (28 layers) │ ── hidden state ──┐
                 │  1 frame per step  │ ── codec logits ──┴─> codebook 0
                 └────────────────────┘
                          │                                     │
                          └──> code predictor (5 layers) ───────┴─> codebooks 1..15
                                    15 sequential passes
                          │
                          v  codes [n_frames, 16]  @ 12.5 Hz
                 ┌────────────────────┐
                 │ vocoder            │
                 │ (WavTokenizer dec) │
                 └────────────────────┘
                          │
                          v  PCM f32 mono @ 24 kHz
                    WAV / MP3 / Opus
```

Two things about the rate. The transformer emits **one frame per step at 12.5
Hz**, each frame being 16 codebook indices; the vocoder expands each frame to
1920 samples (`2 × 2 × 8 × 5 × 4 × 3`) to reach 24 kHz. So a second of speech
is 12.5 talker steps, 12.5 × 15 code-predictor passes, and one vocoder pass
over 1920 samples. That ratio is why generation and decoding cost what they do.

The orchestration lives in `Qwen3TTS::synthesize_internal()`
(`src/qwen3_tts.cpp`). Everything below is a stage it calls.

---

## Stage 1 — text tokenizer

`text_tokenizer.{h,cpp}` + `tokenizer_unicode*.cpp`. BPE, loaded from the same
GGUF as the talker. `encode_for_tts()` wraps the text in the chat framing the
model expects; `encode_instruct()` handles the VoiceDesign variant's
`instructions` field (ignored on 0.6B).

No model runs here. The output is a token vector that becomes part of the
prefill embedding.

## Stage 2 — the voice

Two separate encoders, both optional, both only on **Base** variants:

- **Speaker encoder** (`audio_tokenizer_encoder.{h,cpp}`) — ECAPA-TDNN over a
  128-mel spectrogram. Produces one vector, **as wide as the talker's hidden
  size**. That is why a voices directory encoded by the 0.6B (1024) cannot be
  read by the 1.7B (2048).
- **Codec encoder** (`audio_codec_encoder.{h,cpp}`) — a Mimi encoder, 8
  transformer layers, first 16 of 32 quantizers. Produces **reference codes**:
  the same kind of `[n_frames, 16]` the talker will later generate. These are
  what makes ICL cloning work — the reference audio is fed to the talker in its
  own output format.

`voice_store.{h,cpp}` caches both to `<voices>/<id>/cache.bin` so a voice is
encoded once per variant, not once per request. It is two-tier (read-only disk
voices plus an in-memory session tier) and lazy: `preload_all()` runs on a
background thread so the server starts listening immediately, and `get()`
encodes on demand anything the sweep has not reached.

## Stage 3 — the transformer

`tts_transformer.{h,cpp}`, the largest file in the project. Two models in one
GGUF:

**Talker** — 28-layer Qwen2, 16 heads / 8 KV heads / 128 head_dim. Hidden size
is 1024 on the 0.6B and 2048 on the 1.7B. It runs once over the prefill, then
one step per frame. Each step outputs a hidden state and codec logits, from
which **codebook 0** is sampled.

**Code predictor** — 5 layers, **1024 wide on both variants**. Takes the
talker's hidden state plus the codebook-0 embedding and produces codebooks 1–15
**autoregressively: 15 sequential passes per frame**, with its own small KV
cache (16 tokens). Because its width does not follow the talker's, it costs the
same on both models — which is why the 0.6B is not proportionally faster end to
end.

The prefill embedding is a fixed structure (10 positions for the thinking path,
one shorter without a language id, different again for ICL). The exact layout
is in `AGENTS.md`; it must mirror the Python pipeline position for position.

**ICL prefill is the expensive one.** With a reference voice, the reference
codes are prefilled ahead of the text — 150 frames is typical — and this is
**recomputed on every request** even though the prefix is byte-identical for a
fixed voice. See `optimization.md`, remaining idea 4.

Generation ends on `tts_eos`, on the per-request frame budget, or on abort.

## Stage 4 — the vocoder

`audio_tokenizer_decoder.{h,cpp}`. A WavTokenizer decoder, and structurally the
odd one out: it is a convolutional tower, not a transformer stack, and it is
where most of the graph-level trouble in this project lives.

The chain, in graph order:

| step | what | shape effect |
|---|---|---|
| RVQ lookup | codebook 0 through `vq_first`, codebooks 1–15 through `vq_rest`, all projected and **summed** | `[n_frames, hidden]` |
| `pre_conv` | causal conv1d, kernel 7 | — |
| `pre_tfm` | 8 transformer layers with their own KV cache and causal mask | — |
| `upsample.0/1` | `conv_transpose_1d` **stride 2**, then depthwise conv + pointwise FFN | ×2 each |
| `dec.0` | causal conv1d | — |
| `dec.1..4` | snake activation, then `conv_transpose_1d` at **stride 8, 5, 4, 3**, each with three dilated residual blocks (dilation 1, 3, 9) | ×8 ×5 ×4 ×3 |
| `dec.6` | final conv1d to one channel | → waveform |

The 15-way sum at the top is the add chain that RADV's fusion rule mishandles
(`ggml-notes.md`). The six transposed convolutions used to be 45–52% of decode;
they are now a GEMM pair instead of ggml's own op — see *Device placement*
below.

`set_active_codebooks()` lets both stages agree on how much of the RVQ chain is
in play; the talker must not predict codebooks the vocoder will ignore, and the
vocoder must not sum codebooks the talker left at zero.

---

## Device placement

This is the part that is easy to get silently wrong, and has been, twice.

**Weights decide where a graph runs.** `ggml_backend_sched` pins a node to the
buffer its weights live in. So `load_tensor_data_from_file()`
(`src/gguf_loader.cpp`) has to place weights on the accelerator too, walking the
same `IGPU → GPU → ACCEL → CPU` ladder as `init_preferred_backend()`. Getting
this wrong does not fail — the log still prints the GPU's name while the whole
graph runs on the CPU (`known-issues.md` #13).

**Concurrent stages need their own backend instance.**
`init_preferred_backend(..., exclusive=true)`. One instance owns one stream and
one memory pool, and the CUDA pool asserts frees arrive in reverse order, which
two threads cannot honour. Only the decoder needs this today.

**Conv towers must call `force_f32_matmuls()` before allocation.** Conv weights
are F16 and `ggml_conv_1d` keeps its im2col in F16, so CUDA/HIP pick a
half-precision accumulator and a deep tower loses ~30 dB. Applies to the vocoder
and the speaker encoder. Talker graphs deliberately keep F16 accumulation
(`known-issues.md` #14).

**The six transposed convolutions do not use ggml's `conv_transpose_1d` op.**
They are built as `mul_mat` + `col2im_1d` instead, which is the same
mathematics — a GEMM producing `K*OC` columns per input step, then a
scatter-add of those columns back onto the signal — and is 2.3-4.5x faster than
the op on every backend measured. The weights are repacked once at load from
ggml's `[K, OC, IC]` to the `[IC, K*OC]` a matmul contracts over, which is why
`decoder_block` carries `conv_t_kernel` and `conv_t_out_ch` rather than reading
them off the tensor.

The op itself was never reaching the GPU anyway: `supports_op` accepts
`GGML_OP_CONV_TRANSPOSE_1D` only when both inputs are F32 — `ggml-cuda.cu:5142`
and `ggml-vulkan.cpp:11549` alike — and these weights ship as F16.
`QWEN3_TTS_CONV_T_GEMM=0` restores the op, and only then does
`QWEN3_TTS_CONV_T_F32` (widen the weights, make it GPU-eligible) do anything.
Both are kept as the evidence behind the choice; the numbers are in
`optimization.md`, *The conv_transpose experiment*.

---

## Memory and chunking

The vocoder is memory-bound at the graph level, not just the weight level.
`ggml_conv_1d` is a graph composition — `im2col` then `mul_mat` — and the
im2col intermediate is large — `[672, 287445]` F16 is ~386 MB for a single
call. Decoding ~150 frames in one graph asks for ~1.5 GiB, and a full utterance
simply fails to allocate on a 6 GB card.

**So chunked decode is not an optimisation, it is what makes the GPU path fit.**
`stream_decode()` runs a fixed number of frames per graph and carries the state
across chunks: conv tails, `conv_transpose` overlaps, and the `pre_tfm` KV
cache. Chunk size costs no speed above ~100 frames; 16 is the operating point
when decode is overlapped with generation. `QWEN3_TTS_DECODE_BATCH` overrides.

Peak VRAM is ~3.2 GB for the 1.7B and ~2.1 GB for the 0.6B at `q8_0`. `bf16`
doubles the weight bytes and the 1.7B then does not fit in 6 GB
(`known-issues.md` #22).

## Concurrency

- **Decode overlaps generation.** `decode_pipeline` (`src/qwen3_tts.cpp`) runs
  the vocoder on a worker thread: the per-frame callback batches codes and hands
  them over while the talker keeps going. Gated per backend — on Vulkan two
  instances cannot coexist and it must stay off. `QWEN3_TTS_PIPELINE=1/0`
  overrides. Worth ~7% now that both stages want the same GPU; it was worth far
  more when it overlapped a GPU talker with a CPU vocoder.
- **Synthesis is serialized.** The server holds one `synth_mutex`; a second
  request waits rather than failing fast.
- **Lock order is `map_mutex` before `synth_mutex`**, and `VoiceStore` locks per
  voice so a background preload does not block the listener.

## What is kept, and what is redone

| kept | where | for how long |
|---|---|---|
| speaker embedding + ref codes | `<voices>/<id>/cache.bin`, and the session tier in RAM | across runs; per model variant |
| loaded weights | one backend buffer per component | process lifetime, minus the idle watchdog |
| talker KV | `tts_transformer` | one request |
| `pre_tfm` KV, conv tails, transpose overlaps | `audio_tokenizer_decoder` | one request, across its chunks |
| **ICL reference prefix KV** | *nowhere* | **recomputed every request** |

That last row is the one open architectural gap. Everything else about a voice
is already cached; its prefix KV is not.

---

## Two front ends, one core

- `qwen3-tts-cli` (`main.cpp`) — one-shot. Verbose by default.
- `qwen3-tts-server` (`server.cpp` + `server_args` + `server_audio`) —
  OpenAI-compatible `/v1/audio/speech`, live streaming, the voice library, an
  idle-unload watchdog. Quiet by default (`TTS_VERBOSE` / `--verbose`).
- `libqwen3tts.so` (`qwen3tts_c_api`) — a C ABI over the same core, currently
  behind it in features (no ICL, no streaming).

All three go through `Qwen3TTS`, so a change to the pipeline reaches every
front end. `install_ggml_log_bridge()` must run before the first model load or
ggml's own output bypasses the logger (`known-issues.md`, log contract).

## Where to go next

| question | file |
|---|---|
| how fast, and what was tried | `docs/optimization.md` |
| which weight type, and why F16 breaks | `docs/quantisation.md` |
| a bug, or an old decision's reasoning | `docs/known-issues.md` |
| ggml's own gaps and our patches | `docs/ggml-notes.md` |
| how streaming decode was derived | `docs/streaming_design.md` |
| tensor names in the GGUF | `docs/tensor_mapping.md` |
| conventions, file layout, prefill layout | `AGENTS.md` |
