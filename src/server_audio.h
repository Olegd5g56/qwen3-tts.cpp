#pragma once

#include "qwen3_tts.h"

#include <string>
#include <vector>

// Audio packaging helpers used by the speech endpoint. None of these touch
// the model — pure data formatting. Compressed encoders (Opus/MP3) live in
// audio_streamers.h / qwen3_tts.cpp encode_* and are not duplicated here.

// Float32 mono PCM -> int16 LE byte buffer. Used for response_format=pcm and
// for the PCM payload inside streaming WAV/SSE deltas.
std::string encode_pcm(const std::vector<float> & samples);

// 44-byte WAV/RIFF header with 0xFFFFFFFF placeholder sizes for streaming.
// Players that tolerate non-finite sizes (ffmpeg, vlc, most browsers) start
// playing before EOF arrives.
std::string wav_streaming_header(int sample_rate);

// Minimal RFC 4648 base64 encoder, no line wrapping. Used for the audio
// bytes inside SSE speech.audio.delta frames.
std::string base64_encode(const char * data, size_t len);

// Build the speech.audio.done SSE payload (a JSON string; the caller adds
// the "event: speech.audio.done\ndata: " prefix and trailing blank line).
// Schema mixes openai `usage` with llama.cpp-style `timings` so llama-swap
// can scrape the same fields it does for llama-server.
std::string build_done_event(const qwen3_tts::tts_result & result);
