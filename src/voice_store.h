#pragma once

#include "qwen3_tts.h"

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace qwen3_tts {

// Lazy accessor: returns a loaded Qwen3TTS pointer, or nullptr on failure.
// Called only when the model is actually needed (encoding a fresh voice or
// refreshing a stale cache). MUST be invoked with the associated synth mutex
// held by the caller — the loader is not internally serialized.
using EnsureLoadedFn = std::function<Qwen3TTS*()>;

// One cloned voice. embedding holds the 1024-float speaker vector (may be
// empty for pure-ICL voices). When ref_text is non-empty, ref_codes carries
// the encoded reference frames and the voice is used in ICL mode.
//
// disk_backed=true means the voice was loaded from <root>/<id>/ and survives
// process restarts. disk_backed=false means it was created in-memory via the
// upload endpoint and dies on restart or DELETE.
struct voice_entry {
    std::string id;
    std::vector<float> embedding;
    std::string ref_text;
    std::vector<int32_t> ref_codes;
    int32_t n_ref_frames = 0;
    bool disk_backed = true;
};

// Voice library combining two sources:
//   * Disk: <root>/<id>/{sample.wav|sample.mp3, sample.txt?, .cache.bin}.
//     Scanned on startup, refreshed on demand. `.cache.bin` stores the
//     encoder outputs keyed by sample/text mtimes, so warm starts skip the
//     expensive re-encoding. The server treats these as read-only — they
//     are curated by manipulating files on disk.
//   * Session: created via the upload endpoint, kept entirely in RAM. They
//     survive idle-unload of the model (just data), but not process restart.
//
// Thread-safe. Lock order: callers must release the internal map mutex
// before acquiring the synthesis mutex externally. Public entry points
// respect this themselves.
class VoiceStore {
public:
    VoiceStore(std::string root_dir, EnsureLoadedFn ensure_loaded,
               bool has_speaker_encoder, std::mutex * synth_mutex);

    // Scan root_, drop disk entries whose source dirs vanished, (re)load
    // anything with a stale or missing cache. Session voices are not
    // touched. Cheap on warm starts (just stats files). Returns true if
    // the root dir is usable (created on demand if missing).
    bool refresh();

    // Names of all currently-known voices, sorted.
    std::vector<std::string> list();

    // Copy out a voice by id. If not in the in-memory map, refreshes from
    // disk once before giving up. Returns false if no such voice exists.
    bool get(const std::string & id, voice_entry & out);

    // Decode an uploaded audio buffer (WAV or MP3, sniffed by magic bytes),
    // encode embedding + ref_codes in RAM, and register as a session voice.
    // Nothing is written to disk. Returns false (and sets error) on:
    //   * invalid id / missing speaker encoder
    //   * empty or malformed audio
    //   * collision with an existing disk-backed voice (id reserved)
    //   * encoder failure
    // If a session voice with the same id already exists it is replaced.
    bool create(const std::string & id, const std::string & audio_bytes,
                const std::string & ref_text, std::string & error);

    // Drop a session voice from the in-memory map. Disk-backed voices are
    // immutable through this API and removing them returns false with
    // error="voice is disk-backed; remove the directory on disk instead".
    bool remove(const std::string & id, std::string & error);

    // True only on Base variants — needed for create() and for refreshing
    // a stale cache, but not for using voices already cached on disk.
    bool can_encode_new() const;

    // [A-Za-z0-9_.-]{1,64}, must not start with '.', and "default" is reserved.
    static bool valid_id(const std::string & id);

private:
    bool load_voice_locked(const std::string & id, voice_entry & out, std::string & error);
    bool read_cache(const std::string & dir, voice_entry & out,
                    uint64_t expected_wav_mtime, uint64_t expected_txt_mtime,
                    std::string & error) const;
    bool write_cache(const std::string & dir, const voice_entry & v,
                     uint64_t wav_mtime, uint64_t txt_mtime,
                     std::string & error) const;
    static uint64_t file_mtime_ns(const std::string & path);

    std::string     root_;
    EnsureLoadedFn  ensure_loaded_;
    bool            has_speaker_encoder_;
    std::mutex *    synth_mutex_;
    std::mutex      map_mutex_;
    std::map<std::string, voice_entry> voices_;
};

} // namespace qwen3_tts
