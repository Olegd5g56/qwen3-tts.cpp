#pragma once

// Stateful compressed-audio encoders for live streaming. Both share the same
// on_page callback contract: each emitted chunk is handed to on_page (raw
// Ogg pages for Opus, MPEG frames for MP3); on_page returns false to abort.
// finish() flushes the encoder's internal buffer.
//
// One-shot encode_mp3/encode_opus in qwen3_tts.cpp use these too — they just
// install an on_page that appends to a std::string. That way the encoder
// parameters (VBR -V 2, mono, sample-rate constraints) live in exactly one
// place and CLI/server can't drift.

#include <opusenc.h>
#include <lame/lame.h>

#include <cstddef>
#include <functional>
#include <vector>

namespace qwen3_tts {

// Ogg/Opus encoder. sample_rate must be one of 8/12/16/24/48 kHz
// (libopusenc constraint). 24 kHz matches the vocoder output.
struct opus_streamer {
    OggOpusEnc *      enc      = nullptr;
    OggOpusComments * comments = nullptr;
    std::function<bool(const char *, size_t)> on_page;
    bool failed = false;

    bool open(int sample_rate);
    bool write(const float * pcm, size_t n_samples);
    void finish();
    ~opus_streamer();

private:
    static int write_cb(void * user, const unsigned char * ptr, opus_int32 len);
    static int close_cb(void *);
};

// LAME MP3 encoder. VBR -V 2 (~190 kbps avg), mono, no client-facing knob.
// write() may produce 0 bytes when LAME is buffering for its next frame;
// finish() emits the residual frames.
struct mp3_streamer {
    lame_t                                    lame = nullptr;
    std::function<bool(const char *, size_t)> on_page;
    bool                                      failed = false;
    std::vector<unsigned char>                buf;

    bool open(int sample_rate);
    bool write(const float * pcm, size_t n_samples);
    void finish();
    ~mp3_streamer();
};

} // namespace qwen3_tts
