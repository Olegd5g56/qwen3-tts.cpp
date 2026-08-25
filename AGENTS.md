# AGENTS.md

Coding conventions and architecture guide for AI agents working on this codebase.

## Project Overview

`qwen3-tts.cpp` is a pure C++17 implementation of the Qwen3-TTS text-to-speech pipeline using GGML. It converts text to speech through four stages: tokenization, speaker encoding, transformer code generation, and vocoder decoding.

## Repository Structure

```
qwen3-tts.cpp/
  src/                          # C++ source files
    main.cpp                    # CLI entry point
    qwen3_tts.{h,cpp}           # Pipeline orchestration + audio I/O
                                #   (load_audio_file/bytes, encode_wav/mp3/opus,
                                #    save_audio_file with extension sniffing)
    audio_streamers.{h,cpp}     # opus_streamer / mp3_streamer — stateful encoders
                                #   used by server live-stream AND one-shot encode_*
    voice_store.{h,cpp}         # Two-tier voice library (disk + session)
                                #   used by both CLI and server
    server.cpp                  # OpenAI-compatible TTS HTTP server entry +
                                #   handlers + idle watchdog
    server_args.{h,cpp}         # server_params, env loader, CLI parser,
                                #   --help banner, hf_resolve / hf_download
    server_audio.{h,cpp}        # PCM/WAV-header/base64 + SSE done event builder
    tts_transformer.{h,cpp}     # TTS transformer (talker + code predictor)
    text_tokenizer.{h,cpp}      # BPE text tokenizer
    audio_tokenizer_encoder.{h,cpp}  # ECAPA-TDNN speaker encoder
    audio_codec_encoder.{h,cpp}      # Mimi codec encoder (for ICL voice cloning)
    audio_tokenizer_decoder.{h,cpp}  # WavTokenizer vocoder (streaming-capable)
    qwen3tts_c_api.{h,cpp}      # C ABI shared lib (for Nim/other FFI)
    gguf_loader.{h,cpp}         # GGUF model loading
    op_profiler.{h,cpp}         # opt-in per-op GPU profiler
                                #   (QWEN3_TTS_PROFILE_OPS=1)
    coreml_code_predictor.{h,mm} # macOS-only Core ML bridge (stub on Linux)
  tests/                        # Component tests
    test_tokenizer.cpp
    test_encoder.cpp
    test_transformer.cpp        # Deterministic reference comparison
    test_decoder.cpp
    test_streaming_parity.cpp   # Streaming-vs-oneshot decoder parity
    test_icl_dump.cpp           # ICL diagnostic dump
  scripts/                      # Python utilities (HF↔GGUF conversion, refs)
                                # + bench_speed.sh: the speed protocol
  benchmarks/                   # speed.tsv: committed speed history, one row
                                #   per run — append, never tidy
                                # + bench_ru.txt and voice/: the fixed input
                                #   and reference voice. Changing either starts
                                #   a new series.
  reference/                    # Reference data (*.bin gitignored, *.json tracked)
  voices/                       # Default --voices-dir, per-id sample.{wav,mp3,txt}
                                # + cache.bin (encoded embedding + ref codes)
  docs/                         # Design notes (streaming, optimization, tensors)
                                # + known-issues.md: running bug log, append to it
                                # + quantisation.md: which weight type works, and why
                                # + ggml-notes.md: upstream ggml bugs/gaps
  CMakeLists.txt
```

### Targets

- `qwen3_tts` (STATIC) — pipeline + audio I/O + audio_streamers; PUBLIC pkg-config
  deps on `lame` and `libopusenc` (the streamer header embeds their opaque types)
- `qwen3tts_shared` (SHARED, soname `libqwen3tts.so`) — C ABI over qwen3_tts
- `qwen3-tts-cli` — CLI executable, links qwen3_tts + voice_store
- `qwen3-tts-server` — HTTP server, links qwen3_tts + voice_store + httplib +
  nlohmann_json. Gated by `-DQWEN3_TTS_SERVER=ON` (default ON).
- `qwen3-tts-quantize` — re-quantises a GGUF to a type the Python converter
  cannot write (K-quants). Links `ggml` alone: it never loads the model, it
  walks the file tensor by tensor over `ggml_quantize_chunk()`.

## Build System

- **CMake 3.14+** with C++17
- GGML is vendored under `./ggml` and linked from `./ggml/build/src`
- Build GGML first: `cmake -S ggml -B ggml/build -DGGML_METAL=ON && cmake --build ggml/build -j4`
- Build project: `cmake -S . -B build && cmake --build build -j4`
- Timing build: `cmake -S . -B build -DQWEN3_TTS_TIMING=ON && cmake --build build -j4`
- GGML headers are in `./ggml/include`

## Coding Conventions

### C++ Style

- C++17 standard, no exceptions, no RTTI
- Use `fprintf(stderr, ...)` for logging, not `std::cerr`
- Error handling: methods return `bool`, error details stored in `error_msg_` member
- Memory: GGML contexts own tensor memory; use `ggml_free()` for cleanup
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for macros
- Header guards: `#pragma once`
- All public types in `qwen3_tts` namespace

### GGML Patterns

Every forward pass follows this pattern:

```cpp
// 1. Build computation graph
struct ggml_cgraph * gf = build_xxx_graph(...);

// 2. Allocate graph memory
ggml_backend_sched_alloc_graph(state_.sched, gf);

// 3. Set input tensors
struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "input_name");
ggml_backend_tensor_set(inp, data, 0, size);

// 4. Compute
ggml_backend_sched_graph_compute(state_.sched, gf);

// 5. Get output tensors
struct ggml_tensor * out = ggml_graph_get_tensor(gf, "output_name");
ggml_backend_tensor_get(out, output_data, 0, size);

// 6. Reset scheduler
ggml_backend_sched_reset(state_.sched);
```

Important: `ggml_mul_mat` consumes F16/quantized weights directly — do NOT insert a `ggml_cast` to F32 in front of it. An old `ffn_down` cast workaround forced a full dequant of the largest FFN matrix on every step and cost ~6x in generation time (removed in d544a6f).

Not to be confused with it: `ffn_down` **does** carry a deliberate
`ggml_scale` pair (down by 128 before, up by 128 after), added in 63aad82 and
applied only when the weight type is Q4_0/Q4_1/Q5_1. That is not a dequant and
costs two elementwise ops; it exists because CUDA/HIP pack activations for those
types as `block_q8_1`, whose block sum is F16, and this model's activations
overflow it. Removing it returns inf on frame 0. See `known-issues.md` #21.

Backend initialization and scheduling notes:

- Use `init_preferred_backend()` (`src/gguf_loader.cpp`) to select backend in order: `IGPU -> GPU -> ACCEL -> CPU`
- If the selected runtime backend is not CPU, add a CPU backend as scheduler fallback (`backend_cpu`) when calling `ggml_backend_sched_new(...)`
- **Weights decide where a graph runs.** `ggml_backend_sched` pins a node to the buffer its weights live in, so `load_tensor_data_from_file()` has to land them on the accelerator too. It walks the same ladder; passing one exact device class is not enough. Getting this wrong is silent — the log still prints the compute backend's name while the whole graph runs on the CPU (`known-issues.md` #13)
- **A stage that runs concurrently with another needs `init_preferred_backend(..., exclusive=true)`.** One instance owns one stream and one memory pool, and the CUDA pool asserts frees arrive in reverse order, which two threads cannot honour. Only the decoder needs this today (`known-issues.md` #13)
- **Conv-tower graphs must call `force_f32_matmuls()` before allocation.** Conv weights are F16 and `ggml_conv_1d` keeps its im2col F16, so CUDA/HIP pick a half-precision accumulator and a deep tower loses ~30 dB. Applies to the vocoder and the speaker encoder; talker graphs deliberately keep F16 accumulation (`known-issues.md` #14)

### Model Architecture

The TTS transformer has two sub-models:

1. **Talker** — 28-layer Qwen2 transformer, 16 heads / 8 KV heads / 128 head_dim
   - **Hidden size differs by variant: 0.6B is 1024, 1.7B is 2048** (FFN 3072 /
     6144). Everything downstream of the talker inherits this, including the
     speaker embedding — which is why a voice library encoded by one variant is
     not readable by the other (see `docs/known-issues.md` 2).
   - Input: prefill embedding or step embedding (float32, `[1, hidden_size]`)
   - Output: hidden states + codec logits via `codec_head`

2. **Code Predictor** — 5-layer transformer, **1024 wide on both variants**
   (FFN 3072, codebook vocab 2048). Not "the same config as the talker": on the
   1.7B it is half the talker's width, which is why it costs the same on both
   models and why the 0.6B is not proportionally faster end to end.
   - Input: talker hidden state + codebook-0 embedding (2-token prefill)
   - Output: 15 codebook predictions (autoregressive, one per step)
   - Has its own separate KV cache (max 16 tokens)

Read the real numbers off the gguf rather than trusting this list:
`qwen3-tts.embedding_length`, `.block_count`, `.attention.head_count{,_kv}`,
`.code_predictor.*`, `.speaker_encoder.embedding_length`.

### Prefill Embedding Structure (10 positions for single-word input, thinking path)

```
Pos 0:   text_projection(<|im_start|>)
Pos 1:   text_projection(assistant)
Pos 2:   text_projection(\n)
Pos 3:   tts_pad + codec_embd(think_id)
Pos 4:   tts_pad + codec_embd(think_bos_id)
Pos 5:   tts_pad + codec_embd(language_id)
Pos 6:   tts_pad + codec_embd(think_eos_id)
Pos 7:   tts_pad + speaker_embedding
Pos 8:   tts_bos + codec_embd(pad_id)
Pos 9+:  text_projection(text_token[i]) + codec_embd(bos_id or pad_id)
```

This structure must mirror the Python pipeline exactly. Verified against
`build_prefill_graph()` on 2026-08-20 — it is accurate for the thinking path.
Two variations it does not show: with `language_id < 0` the codec prefill is
`[nothink, think_bos, think_eos]` instead of `[think, think_bos, language,
think_eos]`, one position shorter; and ICL mode assembles a different prompt
entirely (reference text and codes interleaved — see the comment on
`icl_mode`).

### Special Token IDs

```
tts_bos = 151672, tts_eos = 151673, tts_pad = 151671
codec_bos = 2149, codec_eos = 2150, codec_pad = 2148
codec_think = 2154, codec_nothink = 2155, codec_think_bos = 2156, codec_think_eos = 2157
english_language_id = 2050
russian = 2069, chinese = 2055, japanese = 2058, korean = 2064,
german = 2053, french = 2061, spanish = 2054
```

Verified against the ggufs and `tts_transformer.h` on 2026-08-20.

### Key Files to Understand

- `tts_transformer.cpp` — The core file (~3200 lines). Contains `generate()`, `forward_prefill()`, `forward_step()`, `predict_codes_autoregressive()`, and all graph builders.
- `qwen3_tts.cpp` — Pipeline orchestration. Calls tokenizer, encoder, transformer, decoder in sequence. Also hosts audio I/O helpers (`load_audio_file`, `save_audio_file`, `encode_wav/mp3/opus`).
- `audio_streamers.h/cpp` — `opus_streamer` and `mp3_streamer`. Both server-side live streaming AND the one-shot `encode_mp3`/`encode_opus` go through these, so VBR settings live in one place.
- `voice_store.h/cpp` — Two-tier voice library: disk (`<dir>/<id>/{sample.wav|mp3,sample.txt?,cache.bin}`) is read-only via the API, plus an in-memory session tier created via `POST /v1/audio/voices`. `refresh()` is cheap (directory listing only); `preload_all()` and `get()` are where encoding happens. `preload_all()` locks per voice and is cancellable, so the server runs it on a background thread and starts listening immediately; `get()` encodes on demand for anything the sweep has not reached. Disk-vs-session collisions return 409.
- `server.cpp` — HTTP entrypoints. The handlers themselves live here; everything around them was moved out: arg parsing/env/HF download → `server_args.cpp`, PCM/WAV/base64/SSE payload → `server_audio.cpp`.
- `CMakeLists.txt` — Build configuration. Each pipeline component is a separate static library; `qwen3_tts` aggregates them and embeds the audio I/O + streamer code.

### Compressed-audio encoder invariants

- MP3: LAME VBR -V 4 with `lame_set_quality(5)` (speech-tuned, ~70 kbps mono at 24 kHz), no client-facing knob — OpenAI's TTS API doesn't expose one. Both one-shot `encode_mp3` and server's `mp3_streamer` go through the same setup in `audio_streamers.cpp::mp3_streamer::open`.
- Opus: Ogg/Opus via libopusenc, mono, sample rate must be `8/12/16/24/48 kHz` (libopusenc constraint). Vocoder output is 24 kHz, so no resampling needed.
- WAV: 16-bit PCM, mono, sample rate from the result.

## Testing

### Reference Data

Deterministic reference data is generated by `scripts/generate_deterministic_reference.py` using float32 Python inference with greedy decoding. Files go in `reference/det_*.bin` (gitignored) and `reference/det_*.json` (tracked).

### Test Strategy

- `test_transformer` loads reference data and compares C++ output at each stage
- Pass criteria: prefill logits cosine > 0.99; speech codes partially match (F16 precision causes divergence)
- E2E comparison (`compare_e2e.py`): checks both pipelines produce valid non-silent audio with similar duration; waveform correlation is informational only

### Running Tests

```bash
bash scripts/run_all_tests.sh           # Full suite
./build/test_transformer --ref-dir reference/  # Transformer only
```

## Git Conventions

- Conventional commits: `feat(scope):`, `fix(scope):`, `docs:`
- Scopes: `transformer`, `vocoder`, `server`, `cli`, `voice`, `audio`, `test`, `timing`, `build`
- One logical change per commit
- Do not commit model files (*.gguf), reference binaries (*.bin), or build directories (`build/`, `build-vulkan/`, etc.)

## Known Limitations

- **F16 talker weights do not work at all** — not "divergence", not "perceptually
  equivalent". The code predictor's SwiGLU reaches ~185k on the 0.6B and F16
  stops at 65504, so the FFN below it returns inf on frame 0 of every request
  and the model generates noise forever. Use `--type bf16` (same size, and the
  checkpoint is already bf16) or the default `q8_0`. See `known-issues.md` #16
- M-RoPE uses 1D positions (equivalent for single-batch, may differ for batched inference)
- `--top-p` is parsed in CLI params but currently not used in transformer sampling
- Top-level CMake expects vendored GGML at `./ggml`

## Performance Profile

Measured 2026-08-24 on CUDA (GTX 1660 SUPER, Q8_0 1.7B, `ward.txt`), stages
timed apart with `QWEN3_TTS_PIPELINE=0`:

| stage | ms/frame |
|---|---|
| talker | 10.1 |
| code predictor | 13.2 |
| generate total | 25.1 |
| vocoder decode | 12.9 |

RTF 0.44 (2.3x realtime) end to end with the pipeline on; 0.30 on ROCm/6800 XT.

- **Generation is now ~2x the vocoder**, and the code predictor is the largest
  single line. It costs the same on the 0.6B and the 1.7B — 15 sequential
  5-layer passes over weights that do not scale with the talker — and it is
  ~70% weight-read bandwidth, not arithmetic. Quantising it works and was
  rejected on audible quality.
- Inside the vocoder, `CONV_TRANSPOSE_1D` is 52% of decode and `IM2COL` is 1.7%.
  Anything premised on avoiding im2col is not worth doing for this model.
- Vocoder decode runs on a worker thread concurrently with generation
  (`decode_pipeline` in `qwen3_tts.cpp`), gated per backend. Worth ~7% now that
  both stages share the GPU; it was worth far more when it overlapped a GPU
  talker with a CPU vocoder. On Vulkan two instances cannot run at once and it
  must stay off. `QWEN3_TTS_PIPELINE=1/0` overrides.
- Attention graphs view only the populated part of the KV cache
  (`QWEN3_TTS_KV_STEP` granularity). Viewing the full `MAX_AUDIO_TOKENS`-sized
  cache used to make flash-attn 31% of generation.
- Peak VRAM: ~3.2 GB for the 1.7B, ~2.1 GB for the 0.6B — with `q8_0` weights.
  `bf16` doubles the weight bytes and the 1.7B then does not fit in a 6 GB card
  (`known-issues.md` #22); `q4_0` is ~30% smaller than q8_0 and 13% faster.
- Speaker embeddings are hidden-size wide, so a voices directory is tied to one
  model variant.

See `docs/optimization.md` for the full breakdown, the list of approaches that
were tried and ruled out, and the measurement pitfalls (compare ms/frame, not
wall time; `--temperature 0` is unstable on this model).

## Historical Performance Profile

Superseded numbers live in `docs/optimization.md`, which records why each old
conclusion changed. Do not re-derive advice from them. In particular, **every
figure measured before 2026-08-24 described a build whose vocoder ran on the
CPU** — including the "vocoder is the wall, write a conv1d kernel" conclusion,
which is retracted.
