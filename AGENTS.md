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
    coreml_code_predictor.{h,mm} # macOS-only Core ML bridge (stub on Linux)
  tests/                        # Component tests
    test_tokenizer.cpp
    test_encoder.cpp
    test_transformer.cpp        # Deterministic reference comparison
    test_decoder.cpp
    test_streaming_parity.cpp   # Streaming-vs-oneshot decoder parity
    test_icl_dump.cpp           # ICL diagnostic dump
  scripts/                      # Python utilities (HF↔GGUF conversion, refs)
  reference/                    # Reference data (*.bin gitignored, *.json tracked)
  voices/                       # Default --voices-dir, per-id sample.{wav,mp3,txt}
                                # + cache.bin (encoded embedding + ref codes)
  docs/                         # Design notes (streaming, optimization, tensors)
  CMakeLists.txt
```

### Targets

- `qwen3_tts` (STATIC) — pipeline + audio I/O + audio_streamers; PUBLIC pkg-config
  deps on `lame` and `libopusenc` (the streamer header embeds their opaque types)
- `qwen3tts_shared` (SHARED, soname `libqwen3tts.so`) — C ABI over qwen3_tts
- `qwen3-tts-cli` — CLI executable, links qwen3_tts + voice_store
- `qwen3-tts-server` — HTTP server, links qwen3_tts + voice_store + httplib +
  nlohmann_json. Gated by `-DQWEN3_TTS_SERVER=ON` (default ON).

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

Backend initialization and scheduling notes:

- Use `init_preferred_backend()` (`src/gguf_loader.cpp`) to select backend in order: `IGPU -> GPU -> ACCEL -> CPU`
- If the selected runtime backend is not CPU, add a CPU backend as scheduler fallback (`backend_cpu`) when calling `ggml_backend_sched_new(...)`
- Decoder follows the same backend preference; load decoder weights with `GGML_BACKEND_DEVICE_TYPE_IGPU` preference for Metal-first execution

### Model Architecture

The TTS transformer has two sub-models:

1. **Talker** — 28-layer Qwen2 transformer (1024 hidden, 16 heads, 8 KV heads, 128 head_dim)
   - Input: prefill embedding or step embedding (float32, [1, 1024])
   - Output: hidden states + codec logits via `codec_head`

2. **Code Predictor** — 5-layer transformer (same attention config)
   - Input: talker hidden state + codebook-0 embedding (2-token prefill)
   - Output: 15 codebook predictions (autoregressive, one per step)
   - Has its own separate KV cache (max 16 tokens)

### Prefill Embedding Structure (10 positions for single-word input)

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

This structure must mirror the Python pipeline exactly.

### Special Token IDs

```
tts_bos = 151672, tts_eos = 151673, tts_pad = 151671
codec_bos = 2149, codec_eos = 2150, codec_pad = 2148
codec_think = 2154, codec_think_bos = 2156, codec_think_eos = 2157
english_language_id = 2050
```

### Key Files to Understand

- `tts_transformer.cpp` — The core file (~3200 lines). Contains `generate()`, `forward_prefill()`, `forward_step()`, `predict_codes_autoregressive()`, and all graph builders.
- `qwen3_tts.cpp` — Pipeline orchestration. Calls tokenizer, encoder, transformer, decoder in sequence. Also hosts audio I/O helpers (`load_audio_file`, `save_audio_file`, `encode_wav/mp3/opus`).
- `audio_streamers.h/cpp` — `opus_streamer` and `mp3_streamer`. Both server-side live streaming AND the one-shot `encode_mp3`/`encode_opus` go through these, so VBR settings live in one place.
- `voice_store.h/cpp` — Two-tier voice library: disk (`<dir>/<id>/{sample.wav|mp3,sample.txt?,cache.bin}`) is read-only via the API, plus an in-memory session tier created via `POST /v1/audio/voices`. `refresh()` is cheap (directory listing only); `preload_all()` and `get()` are where encoding happens. Disk-vs-session collisions return 409.
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

- F16 model weights cause autoregressive divergence vs Python's float32 — speech codes differ but audio is perceptually equivalent
- M-RoPE uses 1D positions (equivalent for single-batch, may differ for batched inference)
- `--top-p` is parsed in CLI params but currently not used in transformer sampling
- Top-level CMake expects vendored GGML at `./ggml`

## Performance Profile

Measured June 2026 on Vulkan (RX 6800 XT, Q8_0 1.7B, ~500-frame clip), after the `d544a6f`..`b031dd7` perf sweep:

- Vocoder decode is ~64% of total time, code generation ~36%. Overall ~1.4x realtime.
- Within generation, the code predictor dominates (~57%) because it runs 15 sequential graph dispatches per frame (1 prefill + 14 codebook steps); the talker is ~35%.
- ICL (cloned-voice) requests must run the reference frames through the streaming vocoder before synthesis so the output starts in the reference timbre. This warm-up (~5 s for 150 frames) is cached process-wide per voice (`Qwen3TTS::warmup_decoder_for_icl`, keyed by ref-codes + vocoder path); only the first request per voice pays it. The cache intentionally lives in a function-local static because the server destroys the whole `Qwen3TTS` object on idle unload.

See `docs/optimization.md` for the full breakdown and the list of approaches that were tried and ruled out.
