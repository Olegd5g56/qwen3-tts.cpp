#include "audio_streamers.h"

namespace qwen3_tts {

// --- opus_streamer ---

bool opus_streamer::open(int sample_rate) {
    comments = ope_comments_create();
    OpusEncCallbacks cb;
    cb.write = &opus_streamer::write_cb;
    cb.close = &opus_streamer::close_cb;
    int error = 0;
    enc = ope_encoder_create_callbacks(&cb, this, comments, sample_rate, 1, 0, &error);
    if (enc == nullptr || error != OPE_OK) return false;
    // Drop the 512-byte padding libopusenc reserves in OpusTags by default for
    // future metadata edits — we don't write tags, so it's just trailing junk
    // and makes strict parsers (ffmpeg) warn "N bytes of comment header remain".
    ope_encoder_ctl(enc, OPE_SET_COMMENT_PADDING_REQUEST, 0);
    return true;
}

bool opus_streamer::write(const float * pcm, size_t n_samples) {
    if (!enc || failed) return false;
    return ope_encoder_write_float(enc, pcm, (int)n_samples) == OPE_OK && !failed;
}

void opus_streamer::finish() {
    if (enc) ope_encoder_drain(enc);
}

opus_streamer::~opus_streamer() {
    if (enc) ope_encoder_destroy(enc);
    if (comments) ope_comments_destroy(comments);
}

int opus_streamer::write_cb(void * user, const unsigned char * ptr, opus_int32 len) {
    auto * s = static_cast<opus_streamer *>(user);
    if (s->failed) return 1;
    if (!s->on_page(reinterpret_cast<const char *>(ptr), (size_t)len)) {
        s->failed = true;
        return 1;
    }
    return 0;
}

int opus_streamer::close_cb(void *) { return 0; }

// --- mp3_streamer ---

bool mp3_streamer::open(int sample_rate) {
    lame = lame_init();
    if (!lame) return false;
    lame_set_in_samplerate(lame, sample_rate);
    lame_set_num_channels (lame, 1);
    lame_set_mode         (lame, MONO);
    lame_set_VBR          (lame, vbr_default);
    lame_set_VBR_quality  (lame, 2.0f);
    lame_set_quality      (lame, 2);
    if (lame_init_params(lame) < 0) {
        lame_close(lame);
        lame = nullptr;
        return false;
    }
    return true;
}

bool mp3_streamer::write(const float * pcm, size_t n_samples) {
    if (!lame || failed) return false;
    const size_t cap = (size_t)(1.25 * (double)n_samples) + 7200;
    if (buf.size() < cap) buf.resize(cap);
    int written = lame_encode_buffer_ieee_float(lame, pcm, nullptr,
                                                (int)n_samples,
                                                buf.data(), (int)buf.size());
    if (written < 0) { failed = true; return false; }
    if (written > 0 && on_page) {
        if (!on_page(reinterpret_cast<const char *>(buf.data()), (size_t)written)) {
            failed = true;
            return false;
        }
    }
    return true;
}

void mp3_streamer::finish() {
    if (!lame || failed) return;
    if (buf.size() < 7200) buf.resize(7200);
    int flushed = lame_encode_flush(lame, buf.data(), (int)buf.size());
    if (flushed > 0 && on_page) {
        on_page(reinterpret_cast<const char *>(buf.data()), (size_t)flushed);
    }
}

mp3_streamer::~mp3_streamer() {
    if (lame) lame_close(lame);
}

} // namespace qwen3_tts
