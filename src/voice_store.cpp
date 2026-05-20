#include "voice_store.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>

namespace fs = std::filesystem;

namespace qwen3_tts {

namespace {

constexpr char     CACHE_MAGIC[8] = {'Q','3','T','V','O','I','C','E'};
constexpr uint32_t CACHE_VERSION  = 1;

#pragma pack(push, 1)
struct cache_header {
    char     magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t wav_mtime_ns;
    uint64_t txt_mtime_ns;   // 0 if no sample.txt present
    uint32_t embedding_n;
    uint32_t ref_codes_n;
    uint32_t n_ref_frames;
    uint32_t reserved2;
};
#pragma pack(pop)
static_assert(sizeof(cache_header) == 48, "voice cache header layout");

std::string read_text_trimmed(const std::string & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                          s.back() == ' '  || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}

void resample_to_24k(std::vector<float> & samples, int sample_rate) {
    if (sample_rate == 24000 || sample_rate <= 0 || samples.empty()) return;
    int64_t new_len = (int64_t)samples.size() * 24000 / sample_rate;
    std::vector<float> resampled(new_len);
    for (int64_t i = 0; i < new_len; i++) {
        float src = (float)i * sample_rate / 24000.0f;
        int idx = (int)src;
        float frac = src - idx;
        if (idx + 1 < (int)samples.size()) {
            resampled[i] = samples[idx] * (1 - frac) + samples[idx + 1] * frac;
        } else {
            resampled[i] = samples[std::min(idx, (int)samples.size() - 1)];
        }
    }
    samples = std::move(resampled);
}

} // namespace

VoiceStore::VoiceStore(std::string root_dir, Qwen3TTS * tts, std::mutex * synth_mutex)
    : root_(std::move(root_dir)), tts_(tts), synth_mutex_(synth_mutex) {}

bool VoiceStore::can_encode_new() const {
    return tts_->has_speaker_encoder();
}

bool VoiceStore::valid_id(const std::string & id) {
    if (id.empty() || id.size() > 64) return false;
    if (id[0] == '.') return false;
    if (id == "default") return false;
    for (char c : id) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!ok) return false;
    }
    return true;
}

uint64_t VoiceStore::file_mtime_ns(const std::string & path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return 0;
    return (uint64_t)st.st_mtim.tv_sec * 1000000000ULL + (uint64_t)st.st_mtim.tv_nsec;
}

bool VoiceStore::refresh() {
    std::lock_guard<std::mutex> lock(map_mutex_);

    std::error_code ec;
    if (!fs::exists(root_, ec)) {
        fs::create_directories(root_, ec);
        if (ec) {
            fprintf(stderr, "voice_store: cannot create %s: %s\n",
                    root_.c_str(), ec.message().c_str());
            return false;
        }
    }

    std::map<std::string, voice_entry> next;
    for (const auto & ent : fs::directory_iterator(root_, ec)) {
        if (ec) break;
        if (!ent.is_directory(ec)) continue;
        std::string id = ent.path().filename().string();
        if (!valid_id(id)) {
            fprintf(stderr, "voice_store: skipping '%s' (invalid name)\n", id.c_str());
            continue;
        }
        voice_entry v;
        std::string err;
        if (load_voice_locked(id, v, err)) {
            next.emplace(id, std::move(v));
        } else {
            fprintf(stderr, "voice_store: skipping '%s': %s\n", id.c_str(), err.c_str());
        }
    }
    voices_ = std::move(next);
    return true;
}

bool VoiceStore::load_voice_locked(const std::string & id, voice_entry & out, std::string & error) {
    const std::string dir = root_ + "/" + id;
    const std::string wav = dir + "/sample.wav";
    const std::string txt = dir + "/sample.txt";

    const uint64_t wav_mtime = file_mtime_ns(wav);
    if (wav_mtime == 0) {
        error = "missing sample.wav";
        return false;
    }
    const uint64_t txt_mtime = file_mtime_ns(txt); // 0 = absent, which is valid

    voice_entry cached;
    std::string cerr;
    if (read_cache(dir, cached, wav_mtime, txt_mtime, cerr)) {
        cached.id = id;
        if (txt_mtime != 0) cached.ref_text = read_text_trimmed(txt);
        out = std::move(cached);
        return true;
    }

    // No valid cache — must re-encode.
    if (!tts_->has_speaker_encoder()) {
        error = "no valid cache and model lacks speaker encoder";
        return false;
    }

    voice_entry v;
    v.id = id;
    if (txt_mtime != 0) v.ref_text = read_text_trimmed(txt);

    {
        std::lock_guard<std::mutex> sl(*synth_mutex_);

        if (!tts_->extract_speaker_embedding(wav, v.embedding)) {
            // ICL with ref_text alone can still work without an embedding,
            // so only fail outright when there's nothing left to fall back on.
            if (v.ref_text.empty()) {
                error = "extract_speaker_embedding failed: " + tts_->get_error();
                return false;
            }
            v.embedding.clear();
        }

        if (!v.ref_text.empty()) {
            std::vector<float> samples;
            int sr = 0;
            if (!load_audio_file(wav, samples, sr)) {
                error = "load_audio_file failed";
                return false;
            }
            resample_to_24k(samples, sr);
            if (!tts_->encode_speech_codes(samples.data(), (int32_t)samples.size(),
                                            v.ref_codes, v.n_ref_frames)) {
                error = "encode_speech_codes failed: " + tts_->get_error();
                return false;
            }
        }
    }

    std::string werr;
    if (!write_cache(dir, v, wav_mtime, txt_mtime, werr)) {
        fprintf(stderr, "voice_store: %s: cache write failed (%s) — voice still usable in memory\n",
                id.c_str(), werr.c_str());
    }
    out = std::move(v);
    return true;
}

bool VoiceStore::read_cache(const std::string & dir, voice_entry & out,
                             uint64_t expected_wav_mtime, uint64_t expected_txt_mtime,
                             std::string & error) const {
    const std::string path = dir + "/.cache.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "no cache"; return false; }
    cache_header hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f) { error = "short read"; return false; }
    if (std::memcmp(hdr.magic, CACHE_MAGIC, 8) != 0) { error = "bad cache magic"; return false; }
    if (hdr.version != CACHE_VERSION) { error = "incompatible cache version"; return false; }
    if (hdr.wav_mtime_ns != expected_wav_mtime) { error = "stale (sample.wav mtime mismatch)"; return false; }
    if (hdr.txt_mtime_ns != expected_txt_mtime) { error = "stale (sample.txt mtime mismatch)"; return false; }

    out.embedding.assign(hdr.embedding_n, 0.0f);
    if (hdr.embedding_n) {
        f.read(reinterpret_cast<char*>(out.embedding.data()),
               (std::streamsize)hdr.embedding_n * sizeof(float));
        if (!f) { error = "short read (embedding)"; return false; }
    }
    out.ref_codes.assign(hdr.ref_codes_n, 0);
    if (hdr.ref_codes_n) {
        f.read(reinterpret_cast<char*>(out.ref_codes.data()),
               (std::streamsize)hdr.ref_codes_n * sizeof(int32_t));
        if (!f) { error = "short read (codes)"; return false; }
    }
    out.n_ref_frames = (int32_t)hdr.n_ref_frames;
    return true;
}

bool VoiceStore::write_cache(const std::string & dir, const voice_entry & v,
                              uint64_t wav_mtime, uint64_t txt_mtime,
                              std::string & error) const {
    const std::string path = dir + "/.cache.bin";
    const std::string tmp  = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) { error = std::strerror(errno); return false; }

    cache_header hdr{};
    std::memcpy(hdr.magic, CACHE_MAGIC, 8);
    hdr.version      = CACHE_VERSION;
    hdr.wav_mtime_ns = wav_mtime;
    hdr.txt_mtime_ns = txt_mtime;
    hdr.embedding_n  = (uint32_t)v.embedding.size();
    hdr.ref_codes_n  = (uint32_t)v.ref_codes.size();
    hdr.n_ref_frames = (uint32_t)v.n_ref_frames;

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    if (!v.embedding.empty()) {
        f.write(reinterpret_cast<const char*>(v.embedding.data()),
                (std::streamsize)v.embedding.size() * sizeof(float));
    }
    if (!v.ref_codes.empty()) {
        f.write(reinterpret_cast<const char*>(v.ref_codes.data()),
                (std::streamsize)v.ref_codes.size() * sizeof(int32_t));
    }
    f.close();
    if (!f) { error = "write failed"; return false; }

    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        error = ec.message();
        return false;
    }
    return true;
}

std::vector<std::string> VoiceStore::list() {
    refresh();
    std::lock_guard<std::mutex> lock(map_mutex_);
    std::vector<std::string> out;
    out.reserve(voices_.size());
    for (const auto & kv : voices_) out.push_back(kv.first);
    return out;
}

bool VoiceStore::get(const std::string & id, voice_entry & out) {
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = voices_.find(id);
        if (it != voices_.end()) { out = it->second; return true; }
    }
    refresh();
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = voices_.find(id);
    if (it == voices_.end()) return false;
    out = it->second;
    return true;
}

bool VoiceStore::create(const std::string & id, const std::string & wav_bytes,
                         const std::string & ref_text, std::string & error) {
    if (!valid_id(id)) {
        error = "invalid voice name (must be [A-Za-z0-9_.-]{1,64}, not '.'-prefixed, not 'default')";
        return false;
    }
    if (!tts_->has_speaker_encoder()) {
        error = "model lacks speaker encoder; voice cloning requires the Base variant";
        return false;
    }
    if (wav_bytes.empty()) {
        error = "empty audio";
        return false;
    }

    const std::string dir      = root_ + "/" + id;
    const std::string wav_path = dir + "/sample.wav";
    const std::string txt_path = dir + "/sample.txt";

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { error = ec.message(); return false; }

    {
        std::ofstream f(wav_path, std::ios::binary | std::ios::trunc);
        if (!f) { error = std::strerror(errno); return false; }
        f.write(wav_bytes.data(), (std::streamsize)wav_bytes.size());
        if (!f) { error = "writing sample.wav failed"; return false; }
    }
    if (ref_text.empty()) {
        fs::remove(txt_path, ec);
    } else {
        std::ofstream f(txt_path, std::ios::binary | std::ios::trunc);
        if (!f) { error = std::strerror(errno); return false; }
        f.write(ref_text.data(), (std::streamsize)ref_text.size());
        if (!f) { error = "writing sample.txt failed"; return false; }
    }
    // force re-encode by deleting any stale cache
    fs::remove(dir + "/.cache.bin", ec);

    voice_entry v;
    std::string lerr;
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (!load_voice_locked(id, v, lerr)) {
            error = lerr;
            return false;
        }
        voices_[id] = v;
    }
    return true;
}

bool VoiceStore::remove(const std::string & id, std::string & error) {
    if (!valid_id(id)) { error = "invalid voice id"; return false; }
    const std::string dir = root_ + "/" + id;
    std::error_code ec;
    if (!fs::exists(dir, ec)) { error = "not found"; return false; }
    fs::remove_all(dir, ec);
    if (ec) { error = ec.message(); return false; }
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        voices_.erase(id);
    }
    return true;
}

} // namespace qwen3_tts
