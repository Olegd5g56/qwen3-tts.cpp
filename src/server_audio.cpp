#include "server_audio.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>

using json = nlohmann::json;

std::string encode_pcm(const std::vector<float> & samples) {
    std::string buf;
    buf.resize(samples.size() * sizeof(int16_t));
    int16_t * dst = reinterpret_cast<int16_t *>(buf.data());
    for (size_t i = 0; i < samples.size(); i++) {
        float s = samples[i];
        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        dst[i] = (int16_t)(s * 32767.0f);
    }
    return buf;
}

std::string wav_streaming_header(int sample_rate) {
    const int num_channels = 1;
    const int bits_per_sample = 16;
    const int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const int block_align = num_channels * bits_per_sample / 8;

    std::string buf;
    buf.resize(44);
    char * p = buf.data();

    auto write_u32 = [](char * dst, uint32_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
        dst[2] = (char)((v >> 16) & 0xff);
        dst[3] = (char)((v >> 24) & 0xff);
    };
    auto write_u16 = [](char * dst, uint16_t v) {
        dst[0] = (char)(v & 0xff);
        dst[1] = (char)((v >> 8) & 0xff);
    };

    memcpy(p, "RIFF", 4);       write_u32(p + 4, 0xFFFFFFFF);
    memcpy(p + 8, "WAVE", 4);
    memcpy(p + 12, "fmt ", 4);  write_u32(p + 16, 16);
    write_u16(p + 20, 1);
    write_u16(p + 22, num_channels);
    write_u32(p + 24, sample_rate);
    write_u32(p + 28, byte_rate);
    write_u16(p + 32, block_align);
    write_u16(p + 34, bits_per_sample);
    memcpy(p + 36, "data", 4);  write_u32(p + 40, 0xFFFFFFFF);
    return buf;
}

std::string base64_encode(const char * data, size_t len) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    int val = 0, valb = -6;
    for (size_t i = 0; i < len; i++) {
        val = (val << 8) + (uint8_t)data[i];
        valb += 8;
        while (valb >= 0) {
            out.push_back(tbl[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(tbl[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

// Semantics:
//   usage.input_tokens   — tokens in the user's `input` text (openai billing).
//   usage.output_tokens  — generated audio codec frames.
//   timings.prompt_n     — real transformer prefill length (text + instruct
//                          + ref_text + ref_codes + framing).
//   timings.prompt_ms    — build_prefill_graph + forward_prefill wall time.
//   timings.predicted_n  — n_audio_tokens.
//   timings.predicted_ms — transformer autoregressive loop only (excludes
//                          vocoder and prefill).
std::string build_done_event(const qwen3_tts::tts_result & result) {
    const int32_t input_tokens   = result.n_text_tokens;
    const int32_t output_tokens  = result.n_audio_tokens;
    const int32_t prefill_tokens = result.n_prefill_tokens;
    const int64_t prompt_ms      = result.t_prefill_ms;

    // transformer decode loop only. if get_last_prefill_ms() overshoots
    // t_generate_ms by rounding, clamp to 0 rather than emit a negative.
    int64_t predicted_ms = result.t_generate_ms - result.t_prefill_ms;
    if (predicted_ms < 0) predicted_ms = 0;

    const double pps = prompt_ms    > 0 ? (double)prefill_tokens * 1000.0 / (double)prompt_ms    : 0.0;
    const double tps = predicted_ms > 0 ? (double)output_tokens  * 1000.0 / (double)predicted_ms : 0.0;

    json ev = {
        {"type", "speech.audio.done"},
        {"usage", {
            {"input_tokens",  input_tokens},
            {"output_tokens", output_tokens},
            {"total_tokens",  input_tokens + output_tokens},
        }},
        {"timings", {
            {"prompt_n",             prefill_tokens},
            {"predicted_n",          output_tokens},
            {"prompt_ms",            prompt_ms},
            {"predicted_ms",         predicted_ms},
            {"prompt_per_second",    pps},
            {"predicted_per_second", tps},
            // project extras (llama-swap ignores unknown keys):
            {"tokenize_ms",          result.t_tokenize_ms},
            {"encode_ms",            result.t_encode_ms},
            {"generate_ms",          result.t_generate_ms},
            {"decode_ms",            result.t_decode_ms},
            {"total_ms",             result.t_total_ms},
            {"n_text_tokens",        input_tokens},
        }},
    };
    return ev.dump();
}
