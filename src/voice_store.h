#pragma once

#include "qwen3_tts.h"

#include <atomic>
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

// Result of a single voice load reported through preload_all. ms is wall-clock
// load time; from_cache distinguishes a fast cache.bin hit from a real encode.
// error is empty on success.
struct preload_progress {
    std::string id;
    bool        from_cache = false;
    int64_t     ms = 0;
    std::string error;
};

// Voice library combining two sources:
//   * Disk: <root>/<id>/{sample.wav|sample.mp3, sample.txt?, cache.bin}.
//     refresh() discovers ids but does NOT touch audio or model — encoding is
//     deferred until the voice is actually requested via get(), or until the
//     caller explicitly calls preload_all() (server startup uses this for
//     pre-warming with progress logging; the CLI uses lazy get() to avoid
//     paying the cost for voices it isn't going to use).
//   * Session: created via the upload endpoint, kept entirely in RAM. They
//     survive idle-unload of the model (just data), but not process restart.
//
// Thread-safe. Lock order: callers must release the internal map mutex
// before acquiring the synthesis mutex externally. Public entry points
// respect this themselves.
class VoiceStore {
public:
    // embedding_width is the talker hidden size of the currently loaded
    // model (0.6B = 1024, 1.7B = 2048). Caches written by a different variant
    // are treated as stale and re-encoded instead of failing at synth time.
    // Pass 0 to skip the check.
    VoiceStore(std::string root_dir, EnsureLoadedFn ensure_loaded,
               bool has_speaker_encoder, int32_t embedding_width,
               std::mutex * synth_mutex);

    // Rescan root_ and rebuild the disk index. Cheap: only directory listing
    // and a few stat() calls per voice, no audio I/O or encoding. Drops
    // loaded disk entries whose source dirs disappeared. Session voices are
    // preserved. Returns true if the root dir is usable (created on demand).
    bool refresh();

    // Force-load every discovered disk voice (cache hit or fresh encode), one
    // by one. on_voice fires once per voice with its load time and outcome.
    // Returns the number of voices that loaded successfully. Safe to call
    // multiple times — already-loaded voices are skipped silently.
    //
    // The map mutex is taken and released **per voice**, not for the whole
    // sweep, so a get() racing this waits for at most the one voice being
    // encoded rather than the entire library. That is what makes it safe to
    // run on a background thread while the server is already answering.
    //
    // If `cancel` is non-null the sweep checks it between voices and stops
    // early, so shutdown does not have to wait out a large library.
    size_t preload_all(std::function<void(const preload_progress &)> on_voice,
                       const std::atomic<bool> * cancel = nullptr);

    // Names of all currently-known voices, sorted. Cheap — does not load.
    std::vector<std::string> list();

    // Copy out a voice by id. Lazy: if the voice is in the disk index but
    // hasn't been loaded yet, this is the call that pays the encoding cost.
    // Returns false if no such voice exists.
    bool get(const std::string & id, voice_entry & out);

    // Decode an uploaded audio buffer (WAV or MP3, sniffed by magic bytes),
    // encode embedding + ref_codes in RAM, and register as a session voice.
    // Nothing is written to disk. Returns false (and sets error) on:
    //   * invalid id / missing speaker encoder
    //   * empty or malformed audio
    //   * collision with a disk-backed voice (loaded OR merely indexed)
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
    // What refresh() records per discovered disk voice. Holding only paths
    // here keeps refresh fast — the actual audio is decoded only on demand.
    struct disk_voice_info {
        std::string sample_path;     // sample.wav or sample.mp3 (whichever exists; wav wins)
        std::string ref_text_path;   // sample.txt, or empty when absent
    };

    bool load_voice_locked(const std::string & id, voice_entry & out,
                           bool & from_cache, std::string & error);
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
    int32_t         embedding_width_;
    std::mutex *    synth_mutex_;
    std::mutex      map_mutex_;
    std::map<std::string, disk_voice_info> disk_index_;
    std::map<std::string, voice_entry>     voices_;
};

} // namespace qwen3_tts
