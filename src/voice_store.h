#pragma once

#include "qwen3_tts.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace qwen3_tts {

// One cloned voice. id matches the directory name on disk. embedding holds
// the 1024-float speaker vector (may be empty for pure-ICL voices). When
// ref_text is non-empty, ref_codes carries the encoded reference frames
// and the voice is used in ICL mode.
struct voice_entry {
    std::string id;
    std::vector<float> embedding;
    std::string ref_text;
    std::vector<int32_t> ref_codes;
    int32_t n_ref_frames = 0;
};

// Filesystem-backed voice library at <root>/<id>/{sample.wav, sample.txt?, .cache.bin}.
// .cache.bin stores the encoder outputs (embedding + ref_codes) keyed by
// the mtimes of sample.wav and sample.txt, so subsequent startups skip the
// expensive re-encoding. Thread-safe.
//
// Lock order: callers must release the internal map mutex before acquiring
// the synthesis mutex externally. get()/list()/create()/remove() are the
// only public entry points and they respect this themselves.
class VoiceStore {
public:
    VoiceStore(std::string root_dir, Qwen3TTS * tts, std::mutex * synth_mutex);

    // Scan root_, drop entries whose source dirs vanished, (re)load anything
    // with a stale or missing cache. Cheap on warm starts (just stats files).
    // Returns true if the root dir is usable (created on demand if missing).
    bool refresh();

    // Names of all currently-known voices, sorted.
    std::vector<std::string> list();

    // Copy out a voice by id. If not in the in-memory map, refreshes from
    // disk once before giving up. Returns false if no such voice exists.
    bool get(const std::string & id, voice_entry & out);

    // Write sample.wav (+ sample.txt if ref_text non-empty), encode, persist
    // .cache.bin, and register in the map. Fails on invalid id, missing
    // speaker encoder, or any I/O / encoding failure.
    bool create(const std::string & id, const std::string & wav_bytes,
                const std::string & ref_text, std::string & error);

    // rm -rf the voice dir and drop it from the map. Returns false if absent.
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

    std::string  root_;
    Qwen3TTS *   tts_;
    std::mutex * synth_mutex_;
    std::mutex   map_mutex_;
    std::map<std::string, voice_entry> voices_;
};

} // namespace qwen3_tts
