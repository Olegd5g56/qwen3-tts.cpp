# Running it — server, CLI, voices, knobs

The operating manual for both front ends. The README has the shortest path to a
working server; this has every flag, endpoint and field.

- [Server flags](#server-flags)
- [How it behaves under load](#how-it-behaves-under-load)
- [Endpoints](#endpoints)
- [POST /v1/audio/speech](#post-v1audiospeech)
- [Streaming](#streaming)
- [Voice library](#voice-library)
- [CLI](#cli)
- [Environment knobs](#environment-knobs)

## Server flags

Every flag has a matching `TTS_*` environment variable. Precedence is
CLI > environment > default.

| Flag | Env | Default | What it does |
|------|-----|---------|--------------|
| `-m, --model` | `TTS_MODEL` | — | talker GGUF (file or directory) |
| `-v, --vocoder` | `TTS_VOCODER` | auto | vocoder GGUF |
| `--hf-repo` / `--hf-repo-v` | `TTS_HF_REPO` / `TTS_HF_REPO_V` | — | auto-download from HuggingFace |
| `--hf-file` / `--hf-file-v` | `TTS_HF_FILE` / `TTS_HF_FILE_V` | derived | override the GGUF filename |
| `-H, --host` | `TTS_HOST` | `127.0.0.1` | listen address |
| `-p, --port` | `TTS_PORT` | `8080` | listen port |
| `-j, --threads` | `TTS_THREADS` | `4` | compute threads |
| `--voices-dir` | `TTS_VOICES_DIR` | — | voice library directory |
| `--idle-timeout` | `TTS_IDLE_TIMEOUT` | `0` | unload the model after N idle seconds (0 = never); it reloads lazily and the voice library survives |
| `--max-queue` | `TTS_MAX_QUEUE` | `2` | requests allowed to wait for the synthesis slot; past that, `429` with `Retry-After` (0 = unbounded) |
| `-V, --verbose` | `TTS_VERBOSE` | off | per-stage progress and timing |
| `--temperature`, `--top-k`, `--repetition-penalty`, `--seed` | `TTS_*` | `0.9` / `50` / `1.05` / `-1` | sampling defaults, overridable per request |

Models come from either `--hf-repo <repo>[:<quant>]` (default quant `Q8_0`,
cached in `~/.cache/huggingface/`) or a local `-m talker.gguf -v vocoder.gguf`.
`-m` also accepts a directory and finds the vocoder in it.

## How it behaves under load

**One request at a time.** The model is a single instance behind a mutex, so a
second request waits for the first — tens of seconds on long text. One in
progress plus `--max-queue` waiting is admitted; past that the server answers
`429` with `Retry-After` rather than accepting work it will finish long after
the caller has given up. `--max-queue 0` restores unbounded queueing; do that
only if your client does not retry, because a refused request is a lost one.

**A client that hangs up stops the work.** The connection is polled while
synthesis runs, and a dropped client aborts generation instead of burning the
GPU for nobody. The same check covers time spent queued.

**The port opens before the voice library is warm.** Voices encode on demand,
so requests work immediately; the first call for a not-yet-warmed voice pays
the encode.

**`--temperature 0` is not a "make it deterministic" switch** on this model —
greedy decoding loops and burns the whole frame budget on near-silence. For
repeatable output use a low temperature and a fixed `--seed`.

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `GET` | `/health` | health check; answers while busy |
| `GET` | `/v1/models` | the currently-loaded model id |
| `GET` | `/v1/audio/languages` | supported language codes |
| `GET` | `/v1/audio/voices` | built-in and library voices |
| `POST` | `/v1/audio/voices` | upload an in-memory session voice (multipart) |
| `DELETE` | `/v1/audio/voices/:id` | drop a session voice |
| `POST` | `/v1/audio/speech` | synthesize |

`/health`:

```json
{"status":"ok","model_loaded":true,"busy":false,
 "voices":{"total":312,"warmed":312,"warming":false}}
```

`voices` is absent when no voice library is configured.

## POST /v1/audio/speech

```json
{
  "input": "Text to synthesize (max 4096 UTF-8 codepoints)",
  "voice": "default | <built-in name> | <library voice id>",
  "instructions": "(VoiceDesign models only) describe the desired voice",
  "language": "en",
  "response_format": "mp3 (default, as in OpenAI) | wav | pcm | opus",
  "stream_format": "audio | sse",
  "stream_batch_size": 16,
  "temperature": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.05,
  "seed": -1
}
```

`language` applies to the whole request — there is no per-clause markup. Set it
to the carrier language; Latin names and embedded foreign sentences ride along
fine.

PCM is 24 kHz mono S16LE. MP3 is LAME VBR `-V 4` (~70 kbps mono, fixed — the
OpenAI spec has no bitrate knob).

The per-request frame budget is estimated from the input length, so a short
line cannot run away for 8 minutes. Dense scripts and digits are budgeted
higher than letters, because one han character or one number is spoken as a
whole word.

## Streaming

Live streaming needs **both** `stream_format` and a non-zero
`stream_batch_size`.

- **Default** (no `stream_format`) — the full audio body once generation
  finishes.
- **`stream_format=audio`** — HTTP chunked transfer. WAV goes out with a
  placeholder-size header so playback starts immediately; Opus is a
  self-contained Ogg stream; MP3 is self-framing.
- **`stream_format=sse`** — `speech.audio.delta` frames (base64) followed by
  `speech.audio.done`.

```bash
curl -sN -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello, streaming PCM straight to ALSA.",
       "response_format": "pcm", "stream_format": "audio",
       "stream_batch_size": 16, "language": "en"}' \
| aplay -q -f S16_LE -r 24000 -c 1
```

How the streaming vocoder keeps chunk boundaries seamless is in
[streaming_design.md](streaming_design.md).

## Voice library

Two sources that never overlap. Both need a **Base** model.

**Disk-backed**, via `--voices-dir` — managed by editing files:

```
<voices-dir>/
  alice/
    sample.wav     # or sample.mp3 (wav wins if both exist)
    sample.txt     # the transcript — this is what enables ICL cloning
    cache.bin      # generated: embedding + reference codes
```

`sample.txt` must be what is actually said in the clip. Without it only the
speaker embedding is used and the likeness is much weaker; with a transcript
that does not match, generation runs to the frame budget and returns noise
([known-issues.md](known-issues.md) #23). Keep the clip to a few seconds of
clean speech — only the first 150 frames are used.

A voice is re-encoded (~1.4 s) whenever `sample.*` changes or the model variant
does — no need to delete `cache.bin` by hand. The first use of an ICL voice
also pays a ~2 s vocoder warm-up, cached in-process afterwards. `DELETE` on a
disk voice returns **409**; remove the directory instead.

**Session voices**: `POST /v1/audio/voices` holds a voice in RAM. It survives
an idle unload, dies on restart, and returns **409** on a name that collides
with a disk voice.

## CLI

One-shot synthesis. Same environment variables, same voice layout, same model
rules as the server. The output format follows the extension (`.wav`, `.mp3`,
`.opus`/`.ogg`).

```bash
./build/qwen3-tts-cli -m ./models -t "Hello!" -o hello.wav
echo "From a pipe" | ./build/qwen3-tts-cli -m ./models -o piped.wav

# library voice / inline reference / built-in speaker
./build/qwen3-tts-cli -m ./base --voices-dir ./voices -v bob -t "Hi bob" -o bob.wav
./build/qwen3-tts-cli -m ./base -r reference.mp3 --ref-text "what the clip says" \
                      -t "Hello" -o cloned.wav
./build/qwen3-tts-cli -m ./customvoice -v alice -t "Hi" -o alice.wav

./build/qwen3-tts-cli -m ./model --voices-dir ./voices --list-voices
```

Voices load lazily — only the one passed to `-v` is encoded.

CLI-only flags: `-t/--text`, `-o/--output`, `-r/--reference`, `--ref-text`,
`-l/--language`, `-i/--instructions`, `--list-voices`, `--codebooks`,
`--streaming-batch-size`. Do not set `--codebooks` below 12: it wrecks the
spoken text, not just the timbre ([known-issues.md](known-issues.md) #7).

Unlike the server, the CLI is verbose by default.

## Environment knobs

Everything here is optional. `src/env_config.h` is the complete list — a
switch that is not in it does not exist — and every one of them is read **once**,
on first use, so setting one after start has no effect.

### Tuning

| Variable | Default | Effect |
|---|---|---|
| `QWEN3_TTS_PIPELINE` | auto | Overlap the vocoder with generation. Auto-on for CUDA/ROCm/Metal, off for Vulkan. `1`/`0` forces it. |
| `QWEN3_TTS_DECODE_BATCH` | 16 overlapped / 100 sequential | Frames per vocoder batch; overrides both paths. `8` saves ~120 MB VRAM and costs ~7%. Above ~100 frames larger batches buy nothing. |
| `QWEN3_TTS_GRAPH_REUSE` | on | Keep each per-frame graph built and allocated instead of rebuilding it every frame. Worth 13.4% of a whole request on ROCm and ~3% on CUDA; bit-identical output. `0` puts every frame back through the scheduler. Off automatically under `QWEN3_TTS_PROFILE_OPS` or `QWEN3_TTS_PROBE_NUM`, which hook the scheduler a cached graph never reaches. |
| `QWEN3_TTS_FRAME_BUDGET` | on | The runaway guard that caps generated frames from the input length. `0` disables it. |
| `QWEN3_TTS_LOW_MEM` | off | Never keep the talker and the vocoder resident at once. Much lower peak VRAM, one model load per request, no overlap on the one-shot path. |
| `QWEN3_TTS_FORCE_CPU` | off | `1` — and only `1` — keeps everything on the CPU. |
| `QWEN3_TTS_VOCODER` | gpu | `cpu` moves the vocoder off the accelerator. Much slower everywhere measured; it exists for diagnosis. |
| `QWEN3_TTS_PREFIX_CACHE` | 8 | Voices whose reference block and prompt-head KV are kept for reuse across requests, evicted least-recently-used. ~7 MB of host memory per voice on the 1.7B, more for a longer reference transcript. `0` disables the reuse. |
| `QWEN3_TTS_CONV_T_GEMM` | on | Run the vocoder's transposed convolutions as `mul_mat` + `col2im_1d` instead of ggml's `conv_transpose_1d`. 2.3–4.5x faster decode on every backend, +34 MB. `0` restores the op. |
| `QWEN3_TTS_CONV_T_F32` | off | Only reachable with `QWEN3_TTS_CONV_T_GEMM=0`. Widens the six `conv_transpose` weights to F32 (+51 MB) so ggml's op is eligible on the GPU at all. Diagnostic; slower than both the default and the CPU fallback on CUDA and ROCm. |
| `QWEN3_TTS_USE_COREML` | on where built | Apple only. `0`/`false`/`off`/`no` keeps the code predictor on ggml. |
| `QWEN3_TTS_COREML_MODEL` | next to the gguf | Path to the `code_predictor.mlpackage` to use instead of `<model dir>/coreml/code_predictor.mlpackage`. |

### Diagnostics

None of these affect a normal run. Several cost a lot of speed.

| Variable | Default | Effect |
|---|---|---|
| `QWEN3_TTS_PROFILE_OPS` | off | Per-op timing table. Disables fusion and syncs per node, so read its proportions, not its absolute numbers. |
| `QWEN3_TTS_PROBE_NUM` | off | Reports the largest activation magnitude per node — what says whether a tensor survives a cast to F16 (the limit is 65504). |
| `QWEN3_TTS_PROBE_TOP` | 25 | Rows in that report. Nodes over the F16 limit are always shown, however many there are. |
| `QWEN3_TTS_DUMP_PREFILL` | off | Path to write the prefill's last hidden state and logits as raw f32. Prefill is deterministic, so this is where two builds are compared number to number before the sampler diverges. |
| `QWEN3_TTS_DUMP_LOGITS` | off | Top-5 codebook-0 logits and the hidden-state norm for the first five frames. Presence-only: any value, including `0`, enables it. |
| `QWEN3_TTS_DUMP_CODES` | off | Path for the generated speech codes; a `%d` in it becomes a per-call counter. Only written on the sequential decode path, so pair it with `QWEN3_TTS_PIPELINE=0`. |
| `QWEN3_TTS_DUMP_STAGES` | off | Path prefix for the codec encoder's per-stage tensors, for Python-parity bisection. |
| `QWEN3_TTS_DUMP_FEATURES` | off | Path for the codec encoder's pre-VQ features. |
| `QWEN3_TTS_SKIP_REF_CODES` | off | Drops the ICL reference frames from the prompt. Presence-only, as above. |
| `QWEN3_TTS_KEEP_RUNAWAY` | off | Return the audio from a runaway generation instead of reporting failure, so it can be listened to. [known-issues.md](known-issues.md) #11. |

## Using it as a library

`libqwen3tts.so` exposes the same engine over a C ABI for FFI consumers —
`src/qwen3tts_c_api.h` is the reference, and `tests/test_c_api.c` is a worked
example: prepare a voice, synthesize, stream, cache the voice's parts.
