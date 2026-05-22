#include "voice_store.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
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
constexpr const char * CACHE_FILE = "cache.bin";

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

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

VoiceStore::VoiceStore(std::string root_dir, EnsureLoadedFn ensure_loaded,
                        bool has_speaker_encoder, std::mutex * synth_mutex)
    : root_(std::move(root_dir)),
      ensure_loaded_(std::move(ensure_loaded)),
      has_speaker_encoder_(has_speaker_encoder),
      synth_mutex_(synth_mutex) {}

bool VoiceStore::can_encode_new() const {
    return has_speaker_encoder_;
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

    if (root_.empty()) return false;

    std::error_code ec;
    if (!fs::exists(root_, ec)) {
        fs::create_directories(root_, ec);
        if (ec) {
            fprintf(stderr, "voice_store: cannot create %s: %s\n",
                    root_.c_str(), ec.message().c_str());
            return false;
        }
    }

    // Cheap pass: list directory entries and record sample paths. No audio
    // is decoded and no model is touched here — that's preload_all()'s job.
    std::map<std::string, disk_voice_info> next_index;
    for (const auto & ent : fs::directory_iterator(root_, ec)) {
        if (ec) break;
        if (!ent.is_directory(ec)) continue;
        std::string id = ent.path().filename().string();
        if (!valid_id(id)) {
            fprintf(stderr, "voice_store: skipping '%s' (invalid name)\n", id.c_str());
            continue;
        }
        const std::string dir = root_ + "/" + id;
        // Prefer sample.wav; fall back to sample.mp3.
        std::string sample = dir + "/sample.wav";
        if (file_mtime_ns(sample) == 0) {
            std::string mp3 = dir + "/sample.mp3";
            if (file_mtime_ns(mp3) == 0) {
                fprintf(stderr, "voice_store: skipping '%s' (no sample.wav or sample.mp3)\n",
                        id.c_str());
                continue;
            }
            sample = mp3;
        }
        disk_voice_info info;
        info.sample_path = sample;
        const std::string txt = dir + "/sample.txt";
        if (file_mtime_ns(txt) != 0) info.ref_text_path = txt;
        next_index.emplace(std::move(id), std::move(info));
    }

    // Drop loaded disk-backed voices whose source directory disappeared.
    // Session voices stay regardless.
    for (auto it = voices_.begin(); it != voices_.end(); ) {
        if (it->second.disk_backed && next_index.find(it->first) == next_index.end()) {
            it = voices_.erase(it);
        } else {
            ++it;
        }
    }

    disk_index_ = std::move(next_index);
    return true;
}

size_t VoiceStore::preload_all(std::function<void(const preload_progress &)> on_voice) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    size_t loaded = 0;
    for (const auto & kv : disk_index_) {
        const std::string & id = kv.first;
        if (voices_.find(id) != voices_.end()) {
            loaded++;
            continue;
        }
        preload_progress prog;
        prog.id = id;
        const int64_t t0 = now_ms();
        voice_entry v;
        bool from_cache = false;
        if (load_voice_locked(id, v, from_cache, prog.error)) {
            voices_[id] = std::move(v);
            prog.from_cache = from_cache;
            prog.ms = now_ms() - t0;
            if (on_voice) on_voice(prog);
            loaded++;
        } else {
            prog.ms = now_ms() - t0;
            if (on_voice) on_voice(prog);
        }
    }
    return loaded;
}

bool VoiceStore::load_voice_locked(const std::string & id, voice_entry & out,
                                    bool & from_cache, std::string & error) {
    from_cache = false;
    auto it = disk_index_.find(id);
    if (it == disk_index_.end()) {
        error = "voice not present in disk index";
        return false;
    }
    const std::string & wav = it->second.sample_path;
    const std::string & txt = it->second.ref_text_path;

    const uint64_t wav_mtime = file_mtime_ns(wav);
    if (wav_mtime == 0) {
        error = "sample file disappeared: " + wav;
        return false;
    }
    const uint64_t txt_mtime = txt.empty() ? 0 : file_mtime_ns(txt);

    const std::string dir = root_ + "/" + id;

    voice_entry cached;
    std::string cerr;
    if (read_cache(dir, cached, wav_mtime, txt_mtime, cerr)) {
        cached.id = id;
        cached.disk_backed = true;
        if (!txt.empty()) cached.ref_text = read_text_trimmed(txt);
        out = std::move(cached);
        from_cache = true;
        return true;
    }

    // No valid cache — must re-encode.
    if (!has_speaker_encoder_) {
        error = "no valid cache and model lacks speaker encoder";
        return false;
    }

    voice_entry v;
    v.id = id;
    v.disk_backed = true;
    if (!txt.empty()) v.ref_text = read_text_trimmed(txt);

    {
        std::lock_guard<std::mutex> sl(*synth_mutex_);

        // Lazy-load: if the server has idle-unloaded the model, reload it
        // here (still under the synth mutex so no concurrent synth races).
        Qwen3TTS * tts = ensure_loaded_ ? ensure_loaded_() : nullptr;
        if (!tts) {
            error = "model not available (load failed)";
            return false;
        }

        if (!tts->extract_speaker_embedding(wav, v.embedding)) {
            // ICL with ref_text alone can still work without an embedding,
            // so only fail outright when there's nothing left to fall back on.
            if (v.ref_text.empty()) {
                error = "extract_speaker_embedding failed: " + tts->get_error();
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
            if (!tts->encode_speech_codes(samples.data(), (int32_t)samples.size(),
                                            v.ref_codes, v.n_ref_frames)) {
                error = "encode_speech_codes failed: " + tts->get_error();
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
    const std::string path = dir + "/" + CACHE_FILE;
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
    const std::string path = dir + "/" + CACHE_FILE;
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
    std::lock_guard<std::mutex> lock(map_mutex_);
    std::vector<std::string> out;
    out.reserve(disk_index_.size() + voices_.size());
    for (const auto & kv : disk_index_) out.push_back(kv.first);
    for (const auto & kv : voices_) {
        if (disk_index_.find(kv.first) == disk_index_.end()) {
            // session voice not shadowed by disk
            out.push_back(kv.first);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool VoiceStore::get(const std::string & id, voice_entry & out) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto vit = voices_.find(id);
    if (vit != voices_.end()) { out = vit->second; return true; }
    if (disk_index_.find(id) == disk_index_.end()) return false;

    // Discovered but not loaded yet — pay the cost now.
    voice_entry v;
    bool from_cache = false;
    std::string err;
    if (!load_voice_locked(id, v, from_cache, err)) {
        fprintf(stderr, "voice_store: failed to load '%s': %s\n", id.c_str(), err.c_str());
        return false;
    }
    voices_[id] = v;
    out = std::move(v);
    return true;
}

bool VoiceStore::create(const std::string & id, const std::string & audio_bytes,
                         const std::string & ref_text, std::string & error) {
    if (!valid_id(id)) {
        error = "invalid voice name (must be [A-Za-z0-9_.-]{1,64}, not '.'-prefixed, not 'default')";
        return false;
    }
    if (!has_speaker_encoder_) {
        error = "model lacks speaker encoder; voice cloning requires the Base variant";
        return false;
    }
    if (audio_bytes.empty()) {
        error = "empty audio";
        return false;
    }

    // Refuse to shadow a disk-backed voice — whether already loaded into
    // voices_ or merely indexed by refresh() and not yet encoded. Disk voices
    // are curated by file management; silently overlaying them in-memory
    // would confuse the operator.
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (disk_index_.find(id) != disk_index_.end()) {
            error = "voice id is reserved by a disk-backed voice; pick a different name";
            return false;
        }
        auto it = voices_.find(id);
        if (it != voices_.end() && it->second.disk_backed) {
            error = "voice id is reserved by a disk-backed voice; pick a different name";
            return false;
        }
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_audio_bytes(audio_bytes.data(), audio_bytes.size(), samples, sample_rate)) {
        error = "failed to decode reference audio (expected WAV or MP3)";
        return false;
    }

    voice_entry v;
    v.id = id;
    v.ref_text = ref_text;
    v.disk_backed = false;

    {
        std::lock_guard<std::mutex> sl(*synth_mutex_);

        Qwen3TTS * tts = ensure_loaded_ ? ensure_loaded_() : nullptr;
        if (!tts) {
            error = "model not available (load failed)";
            return false;
        }

        if (!tts->extract_speaker_embedding(samples.data(), (int32_t)samples.size(),
                                             sample_rate, v.embedding)) {
            if (v.ref_text.empty()) {
                error = "extract_speaker_embedding failed: " + tts->get_error();
                return false;
            }
            v.embedding.clear();
        }

        if (!v.ref_text.empty()) {
            std::vector<float> ref_samples = samples;
            resample_to_24k(ref_samples, sample_rate);
            if (!tts->encode_speech_codes(ref_samples.data(), (int32_t)ref_samples.size(),
                                            v.ref_codes, v.n_ref_frames)) {
                error = "encode_speech_codes failed: " + tts->get_error();
                return false;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        // Re-check collision in case a disk refresh raced and added a
        // matching disk voice while we were encoding.
        if (disk_index_.find(id) != disk_index_.end()) {
            error = "voice id is reserved by a disk-backed voice; pick a different name";
            return false;
        }
        auto it = voices_.find(id);
        if (it != voices_.end() && it->second.disk_backed) {
            error = "voice id is reserved by a disk-backed voice; pick a different name";
            return false;
        }
        voices_[id] = std::move(v);
    }
    return true;
}

bool VoiceStore::remove(const std::string & id, std::string & error) {
    if (!valid_id(id)) { error = "invalid voice id"; return false; }
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (disk_index_.find(id) != disk_index_.end()) {
        error = "voice is disk-backed; remove the directory on disk instead";
        return false;
    }
    auto it = voices_.find(id);
    if (it == voices_.end()) { error = "not found"; return false; }
    if (it->second.disk_backed) {
        // belt-and-braces — disk_index_ should have caught this above
        error = "voice is disk-backed; remove the directory on disk instead";
        return false;
    }
    voices_.erase(it);
    return true;
}

} // namespace qwen3_tts
