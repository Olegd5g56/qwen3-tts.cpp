// openai-compatible tts server for qwen3-tts.cpp
//
// endpoints:
//   GET  /health              - health check
//   GET  /v1/models           - list loaded model
//   GET  /v1/audio/languages  - list supported languages
//   GET  /v1/audio/voices     - list available voices
//   POST /v1/audio/voices     - create custom voice from reference audio
//   DELETE /v1/audio/voices/X - delete custom voice
//   POST /v1/audio/speech     - synthesize speech (supports voice cloning)

#include "qwen3_tts.h"
#include "voice_store.h"
#include "audio_streamers.h"
#include "server_args.h"
#include "server_audio.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using namespace qwen3_tts;

// supported languages and their model codec token IDs
static const std::vector<std::pair<std::string, int>> SUPPORTED_LANGUAGES = {
    {"en", 2050}, {"zh", 2055}, {"ja", 2058}, {"ko", 2064}, {"ru", 2069},
    {"de", 2053}, {"fr", 2061}, {"es", 2054}, {"it", 2070}, {"pt", 2071},
};

// language string to model language_id (returns -1 if unknown)
static int language_to_id(const std::string & lang) {
    if (lang.empty()) return 2050;
    for (const auto & [code, id] : SUPPORTED_LANGUAGES) {
        if (lang == code) return id;
    }
    return -1;
}

// Encoders/streamers live in qwen3_tts (audio_streamers.h, encode_*).
// PCM/WAV-header/base64/SSE-done payload helpers live in server_audio.h.
// server_params, --help banner, env loader, CLI parser, hf_resolve live in
// server_args.h. This file keeps main, HTTP handlers, and the watchdog.

int main(int argc, char ** argv) {
    server_params sp;
    if (!load_env(sp)) {
        return 1;
    }
    if (!parse_args(argc, argv, sp)) {
        print_usage(argv[0]);
        return 1;
    }

    // resolve --hf-repo to local file paths
    if (!sp.hf_repo.empty()) {
        sp.model = hf_resolve(sp.hf_repo, sp.hf_file);
        if (sp.model.empty()) return 1;
        fprintf(stderr, "resolved model: %s\n", sp.model.c_str());
    }
    if (!sp.hf_repo_v.empty()) {
        sp.vocoder = hf_resolve(sp.hf_repo_v, sp.hf_file_v, "F16");
        if (sp.vocoder.empty()) return 1;
        fprintf(stderr, "resolved vocoder: %s\n", sp.vocoder.c_str());
    }

    // model lives in a unique_ptr so the idle watchdog can unload it; the
    // synth_mutex serializes all access (synthesis itself isn't thread-safe
    // and the watchdog needs an exclusive window to destroy the model).
    std::unique_ptr<Qwen3TTS> tts;
    std::mutex synth_mutex;
    auto last_used = std::chrono::steady_clock::now();

    auto load_model_into = [&](Qwen3TTS & t) -> bool {
        if (!t.load_model_files(sp.model, sp.vocoder)) {
            fprintf(stderr, "model load failed: %s\n", t.get_error().c_str());
            return false;
        }
        t.set_n_threads(sp.n_threads);
        return true;
    };

    // ensure_loaded: caller MUST hold synth_mutex. (Re)constructs the model
    // if it has been idle-unloaded, refreshes last_used, returns the raw ptr.
    // Returns nullptr only if a reload attempt actually fails.
    std::function<Qwen3TTS*()> ensure_loaded = [&]() -> Qwen3TTS* {
        if (!tts) {
            fprintf(stderr, "reloading model: %s\n", sp.model.c_str());
            auto t = std::make_unique<Qwen3TTS>();
            if (!load_model_into(*t)) return nullptr;
            tts = std::move(t);
        }
        last_used = std::chrono::steady_clock::now();
        return tts.get();
    };

    // initial load — fail fast on bad path / missing file, otherwise subsequent
    // requests would hit the same error and confuse clients.
    fprintf(stderr, "loading model: %s\n", sp.model.c_str());
    if (!sp.vocoder.empty()) {
        fprintf(stderr, "loading vocoder: %s\n", sp.vocoder.c_str());
    }
    tts = std::make_unique<Qwen3TTS>();
    if (!load_model_into(*tts)) {
        fprintf(stderr, "fatal: initial model load failed\n");
        return 1;
    }
    // reset the idle clock *after* the load so the watchdog doesn't count the
    // load itself toward the timeout (otherwise short timeouts would unload
    // immediately on startup).
    last_used = std::chrono::steady_clock::now();

    // Snapshot model metadata so /v1/models, /v1/audio/voices listing and the
    // built-in-speaker check work without forcing a reload when idle-unloaded.
    const std::string              cached_model_type          = tts->get_model_type();
    const std::vector<std::string> cached_speaker_names       = tts->get_speaker_names();
    const bool                     cached_has_speaker_encoder = tts->has_speaker_encoder();

    fprintf(stderr, "models loaded (type=%s, speakers=%zu, threads=%d)\n",
            cached_model_type.c_str(), cached_speaker_names.size(), sp.n_threads);
    if (sp.idle_timeout_sec > 0) {
        fprintf(stderr, "idle-timeout: model will unload after %d seconds idle\n",
                sp.idle_timeout_sec);
    }

    // derive model id from filename (e.g. "qwen3-tts-0.6b-f16" from path)
    std::string model_id = sp.model;
    auto slash = model_id.rfind('/');
    if (slash != std::string::npos) model_id = model_id.substr(slash + 1);
    auto dot = model_id.rfind('.');
    if (dot != std::string::npos) model_id = model_id.substr(0, dot);

    // Cloned voices only make sense on Base — they're encoded by the Base
    // speaker_encoder and the embedding space doesn't transfer to other variants.
    if (cached_model_type != "base" && !sp.voices_dir.empty()) {
        fprintf(stderr, "warning: model type '%s' cannot use cloned voices; "
                        "voice library at '%s' will be ignored. "
                        "Use a Base model for voice cloning.\n",
                cached_model_type.c_str(), sp.voices_dir.c_str());
        sp.voices_dir.clear();
    }

    // persistent on-disk voice library; manages embeddings and ICL caches.
    VoiceStore voice_store(sp.voices_dir, ensure_loaded,
                            cached_has_speaker_encoder, &synth_mutex);
    if (!sp.voices_dir.empty()) {
        voice_store.refresh();
        auto discovered = voice_store.list();
        fprintf(stderr, "voice library: %s (%zu voice%s found, preloading...)\n",
                sp.voices_dir.c_str(), discovered.size(),
                discovered.size() == 1 ? "" : "s");
        const auto t_start = std::chrono::steady_clock::now();
        const size_t loaded = voice_store.preload_all([](const preload_progress & p) {
            if (!p.error.empty()) {
                fprintf(stderr, "  %s: FAILED (%s)\n", p.id.c_str(), p.error.c_str());
            } else if (p.from_cache) {
                fprintf(stderr, "  %s: cached (%lldms)\n", p.id.c_str(), (long long)p.ms);
            } else {
                fprintf(stderr, "  %s: encoded in %.1fs\n",
                        p.id.c_str(), (double)p.ms / 1000.0);
            }
        });
        const auto t_total = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();
        std::string failed_suffix;
        if (loaded < discovered.size()) {
            failed_suffix = ", " + std::to_string(discovered.size() - loaded) + " failed";
        }
        fprintf(stderr, "voice library: ready (%zu loaded%s, %.1fs total)\n",
                loaded, failed_suffix.c_str(), (double)t_total / 1000.0);
    }

    httplib::Server svr;

    // log all requests
    svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        fprintf(stderr, "%s %s%s%s -> %d\n",
                req.method.c_str(), req.path.c_str(),
                req.params.empty() ? "" : "?",
                req.params.empty() ? "" : [&]() {
                    static thread_local std::string qs;
                    qs.clear();
                    for (auto & [k, v] : req.params) {
                        if (!qs.empty()) qs += '&';
                        qs += k + "=" + v;
                    }
                    return qs.c_str();
                }(),
                res.status);
    });

    // --- GET /health ---
    svr.Get("/health", [&tts, &synth_mutex](const httplib::Request &, httplib::Response & res) {
        bool loaded;
        {
            std::lock_guard<std::mutex> lock(synth_mutex);
            loaded = (tts != nullptr);
        }
        json h = {{"status", "ok"}, {"model_loaded", loaded}};
        res.set_content(h.dump(), "application/json");
    });

    // --- GET /v1/models ---
    svr.Get("/v1/models", [&model_id](const httplib::Request &, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", json::array({
                {{"id", model_id}, {"object", "model"}, {"owned_by", "qwen"}},
            })},
        };
        res.set_content(models.dump(), "application/json");
    });

    // --- GET /v1/audio/languages ---
    svr.Get("/v1/audio/languages", [](const httplib::Request &, httplib::Response & res) {
        json lang_list = json::array();
        for (const auto & [code, id] : SUPPORTED_LANGUAGES) {
            lang_list.push_back({{"code", code}, {"id", id}});
        }
        res.set_content(json({{"languages", lang_list}}).dump(), "application/json");
    });

    // --- GET /v1/audio/voices ---
    svr.Get("/v1/audio/voices",
        [&model_id, &cached_speaker_names, &voice_store](const httplib::Request &, httplib::Response & res) {
        json voice_list = json::array({"default"});

        // add built-in speakers from model metadata (custom_voice models)
        for (auto & name : cached_speaker_names) {
            voice_list.push_back(name);
        }

        // add on-disk cloned voices
        for (auto & id : voice_store.list()) {
            voice_list.push_back(id);
        }
        res.set_content(json({{model_id, voice_list}}).dump(), "application/json");
    });

    // --- POST /v1/audio/voices --- create or replace voice from reference audio
    svr.Post("/v1/audio/voices",
        [&cached_model_type, &voice_store](const httplib::Request & req, httplib::Response & res) {

        if (!voice_store.can_encode_new()) {
            res.status = 400;
            json err = {{"error", {
                {"message", "this model variant (" + cached_model_type +
                            ") does not support voice cloning from audio; "
                            "use the Base variant, or pick a built-in voice via GET /v1/audio/voices"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // expect multipart form: name (string) + audio_sample (file) [+ ref_text]
        if (!req.has_file("audio_sample")) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'audio_sample' file is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }
        std::string name;
        if (req.has_param("name")) name = req.get_param_value("name");
        if (req.has_file("name")) name = req.get_file_value("name").content;
        if (name.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'name' is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }
        if (!VoiceStore::valid_id(name)) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid 'name': must be [A-Za-z0-9_.-]{1,64}, not '.'-prefixed, not 'default'","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        std::string ref_text;
        if (req.has_param("ref_text")) ref_text = req.get_param_value("ref_text");
        if (req.has_file("ref_text"))  ref_text = req.get_file_value("ref_text").content;

        auto audio_file = req.get_file_value("audio_sample");
        std::string err;
        if (!voice_store.create(name, audio_file.content, ref_text, err)) {
            // Name collision with a disk-backed voice → 409 Conflict; everything
            // else is treated as a malformed request.
            const bool conflict = err.find("reserved by a disk-backed voice") != std::string::npos;
            res.status = conflict ? 409 : 400;
            json e = {{"error", {{"message", "failed to create voice: " + err},
                                  {"type", conflict ? "conflict" : "invalid_request_error"}}}};
            res.set_content(e.dump(), "application/json");
            return;
        }

        fprintf(stderr, "created session voice '%s'%s\n", name.c_str(),
                ref_text.empty() ? "" : " (ICL mode)");
        json resp = {{"id", name}, {"name", name}, {"ephemeral", true}};
        if (!ref_text.empty()) resp["mode"] = "icl";
        res.set_content(resp.dump(), "application/json");
    });

    // --- DELETE /v1/audio/voices/:id ---
    svr.Delete(R"(/v1/audio/voices/(.+))",
        [&voice_store](const httplib::Request & req, httplib::Response & res) {
        std::string voice_id = req.matches[1];
        std::string err;
        if (voice_store.remove(voice_id, err)) {
            res.set_content(R"({"deleted":true})", "application/json");
        } else {
            // Disk-backed voices can't be removed through the API. Missing
            // ids → 404, disk refusal → 409, anything else → 400.
            const bool not_found = (err == "not found");
            const bool conflict  = err.find("disk-backed") != std::string::npos;
            res.status = not_found ? 404 : (conflict ? 409 : 400);
            const char * type = not_found ? "not_found"
                              : conflict  ? "conflict"
                              : "invalid_request_error";
            json e = {{"error", {{"message", err}, {"type", type}}}};
            res.set_content(e.dump(), "application/json");
        }
    });

    // --- POST /v1/audio/speech ---
    svr.Post("/v1/audio/speech",
        [&ensure_loaded, &cached_speaker_names, &last_used,
         &synth_mutex, &sp, &voice_store](const httplib::Request & req, httplib::Response & res) {

        // parse request body
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception &) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid JSON","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        // extract parameters
        std::string input = body.value("input", "");
        if (input.empty()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'input' is required","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        // openai text limit is 4096 chars
        if (input.size() > 4096) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"'input' exceeds 4096 characters","type":"invalid_request_error"}})",
                            "application/json");
            return;
        }

        std::string response_format = body.value("response_format", "wav");
        std::string stream_format   = body.value("stream_format", "");
        std::string voice           = body.value("voice", "");
        std::string instructions    = body.value("instructions", "");
        std::string language        = body.value("language", "en");
        float       temperature     = body.value("temperature", sp.temperature);
        int         top_k           = body.value("top_k", sp.top_k);
        float       repetition_penalty = body.value("repetition_penalty", sp.repetition_penalty);
        int64_t     seed               = body.value("seed", sp.seed);
        int         stream_batch_size  = body.value("stream_batch_size", 0);
        if (stream_batch_size < 0) stream_batch_size = 0;
        if (stream_batch_size > 256) stream_batch_size = 256;

        fprintf(stderr, "request: voice=%s lang=%s fmt=%s temp=%.2f seed=%lld len=%zu\n",
                voice.empty() ? "default" : voice.c_str(),
                language.c_str(), response_format.c_str(),
                temperature, (long long)seed, input.size());

        // validate language
        int language_id = language_to_id(language);
        if (language_id < 0) {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported language '" + language +
                            "', see GET /v1/audio/languages"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // validate response format
        if (response_format != "wav" && response_format != "pcm" &&
            response_format != "opus" && response_format != "mp3") {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported response_format '" + response_format +
                            "', supported: wav, pcm, opus, mp3"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // validate stream format (empty = one-shot, openai-spec values = chunked)
        if (!stream_format.empty() && stream_format != "audio" && stream_format != "sse") {
            res.status = 400;
            json err = {{"error", {
                {"message", "unsupported stream_format '" + stream_format +
                            "', supported: audio, sse"},
                {"type", "invalid_request_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        // resolve voice to speaker embedding (and optional ICL data)
        std::vector<float> voice_embedding;
        std::string voice_ref_text;
        std::vector<int32_t> voice_ref_codes;
        int32_t voice_n_ref_frames = 0;
        if (!voice.empty() && voice != "default") {
            // built-in speaker (CustomVoice models). Membership check uses the
            // metadata snapshot so an idle-unloaded model isn't reloaded just
            // to recognize a name; the actual embedding fetch happens below
            // under synth_mutex (which reloads the model if needed).
            bool is_builtin = false;
            for (const auto & name : cached_speaker_names) {
                if (name == voice) { is_builtin = true; break; }
            }
            if (is_builtin) {
                std::lock_guard<std::mutex> lock(synth_mutex);
                Qwen3TTS * t = ensure_loaded();
                if (!t) {
                    res.status = 503;
                    res.set_content(R"({"error":{"message":"model reload failed","type":"server_error"}})",
                                    "application/json");
                    return;
                }
                if (!t->get_speaker_embedding(voice, voice_embedding)) {
                    res.status = 500;
                    json err = {{"error", {
                        {"message", "failed to get speaker embedding: " + t->get_error()},
                        {"type", "server_error"},
                    }}};
                    res.set_content(err.dump(), "application/json");
                    return;
                }
            } else {
                // on-disk cloned voice
                voice_entry ve;
                if (!voice_store.get(voice, ve)) {
                    res.status = 400;
                    json err = {{"error", {
                        {"message", "unknown voice '" + voice + "'"},
                        {"type", "invalid_request_error"},
                    }}};
                    res.set_content(err.dump(), "application/json");
                    return;
                }
                voice_embedding    = std::move(ve.embedding);
                voice_ref_text     = std::move(ve.ref_text);
                voice_ref_codes    = std::move(ve.ref_codes);
                voice_n_ref_frames = ve.n_ref_frames;
            }
        }

        // set up synthesis params
        tts_params params;
        params.n_threads          = sp.n_threads;
        params.temperature        = temperature;
        params.top_k              = top_k;
        params.repetition_penalty = repetition_penalty;
        params.seed               = seed;
        params.language_id        = language_id;
        params.print_progress     = sp.verbose;
        params.print_timing       = sp.verbose;
        params.instructions       = instructions;
        params.ref_text           = voice_ref_text;

        // live streaming path: when stream_format is set and stream_batch_size
        // > 0, synthesis runs INSIDE set_chunked_content_provider so PCM
        // batches flush to the wire as they're produced. stream_batch_size=0
        // preserves the legacy "synthesize-then-chunk" behavior for clients
        // that want a single delta event.
        const bool live_stream = !stream_format.empty() && stream_batch_size > 0;
        if (live_stream) {
            const bool is_sse  = (stream_format == "sse");
            const bool is_wav  = (response_format == "wav");
            const bool is_opus = (response_format == "opus");
            const bool is_mp3  = (response_format == "mp3");
            const char * ctype = is_sse ? "text/event-stream"
                                        : (is_wav  ? "audio/wav"
                                        : (is_opus ? "audio/ogg"
                                        : (is_mp3  ? "audio/mpeg" : "audio/pcm")));

            // capture synthesis inputs; move into provider lambda below.
            res.set_chunked_content_provider(ctype,
                [ensure_loaded_p = &ensure_loaded, last_used_p = &last_used,
                 input = std::move(input), params = std::move(params),
                 voice_embedding = std::move(voice_embedding),
                 voice_ref_codes = std::move(voice_ref_codes),
                 voice_n_ref_frames,
                 stream_batch_size, is_sse, is_wav, is_opus, is_mp3,
                 synth_mutex = &synth_mutex, sample_rate_fallback = 24000]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    std::lock_guard<std::mutex> lock(*synth_mutex);
                    Qwen3TTS * this_tts = (*ensure_loaded_p)();
                    if (!this_tts) {
                        // can't synthesize without a model; close the stream
                        // gracefully so the client doesn't hang on partial data.
                        sink.done();
                        return false;
                    }

                    // wav header up front (audio mode only). for SSE, the wav
                    // bytes per-delta are raw pcm — clients reconstruct wav.
                    // opus emits its own ogg headers via libopusenc on first
                    // write, so no manual header is needed.
                    bool header_written = false;
                    auto ensure_header = [&]() {
                        if (!header_written && !is_sse && is_wav) {
                            std::string hdr = wav_streaming_header(sample_rate_fallback);
                            sink.write(hdr.data(), hdr.size());
                        }
                        header_written = true;
                    };

                    // compressed encoders are created lazily so we don't pay the
                    // cost on a request that fails before producing any pcm. the
                    // on_page callback is the same shape for both: it either
                    // ships raw bytes (audio mode) or wraps them in an SSE delta.
                    auto wrap_compressed = [&](const char * p, size_t n) -> bool {
                        if (is_sse) {
                            json delta = {
                                {"type", "speech.audio.delta"},
                                {"audio", base64_encode(p, n)},
                            };
                            std::string frame = "event: speech.audio.delta\ndata: "
                                              + delta.dump() + "\n\n";
                            return sink.write(frame.data(), frame.size());
                        }
                        return sink.write(p, n);
                    };

                    opus_streamer opus;
                    bool opus_open = false;
                    auto ensure_opus = [&]() -> bool {
                        if (opus_open || !is_opus) return opus_open || !is_opus;
                        opus.on_page = wrap_compressed;
                        opus_open = opus.open(sample_rate_fallback);
                        return opus_open;
                    };

                    mp3_streamer mp3;
                    bool mp3_open = false;
                    auto ensure_mp3 = [&]() -> bool {
                        if (mp3_open || !is_mp3) return mp3_open || !is_mp3;
                        mp3.on_page = wrap_compressed;
                        mp3_open = mp3.open(sample_rate_fallback);
                        return mp3_open;
                    };

                    streaming_opts sopts;
                    sopts.batch_size = stream_batch_size;
                    sopts.on_pcm = [&](const float * pcm, size_t n) -> bool {
                        ensure_header();
                        if (is_opus) {
                            if (!ensure_opus()) return false;
                            return opus.write(pcm, n);
                        }
                        if (is_mp3) {
                            if (!ensure_mp3()) return false;
                            return mp3.write(pcm, n);
                        }
                        std::string bytes = encode_pcm(std::vector<float>(pcm, pcm + n));
                        if (is_sse) {
                            json delta = {
                                {"type", "speech.audio.delta"},
                                {"audio", base64_encode(bytes.data(), bytes.size())},
                            };
                            std::string frame = "event: speech.audio.delta\ndata: "
                                              + delta.dump() + "\n\n";
                            return sink.write(frame.data(), frame.size());
                        }
                        return sink.write(bytes.data(), bytes.size());
                    };

                    tts_result result;
                    if (!voice_ref_codes.empty()) {
                        result = this_tts->synthesize_with_embedding(
                            input, voice_embedding.data(), (int32_t)voice_embedding.size(),
                            params, voice_ref_codes.data(), voice_n_ref_frames, &sopts);
                    } else if (!voice_embedding.empty()) {
                        result = this_tts->synthesize_with_embedding(
                            input, voice_embedding.data(), (int32_t)voice_embedding.size(),
                            params, nullptr, 0, &sopts);
                    } else {
                        result = this_tts->synthesize(input, params, &sopts);
                    }

                    // ensure a header went out even if no pcm was produced.
                    ensure_header();
                    // flush samples sitting in the compressed-encoder lookahead.
                    if (opus_open) opus.finish();
                    if (mp3_open)  mp3.finish();

                    if (is_sse) {
                        std::string done_frame = "event: speech.audio.done\ndata: "
                                               + build_done_event(result) + "\n\n";
                        sink.write(done_frame.data(), done_frame.size());
                    }
                    // bump idle timer so the watchdog doesn't unload mid-stream
                    *last_used_p = std::chrono::steady_clock::now();
                    sink.done();
                    return false;
                });
            return;
        }

        // synthesize (serialized), using voice embedding if provided
        tts_result result;
        {
            std::lock_guard<std::mutex> lock(synth_mutex);
            Qwen3TTS * t = ensure_loaded();
            if (!t) {
                res.status = 503;
                res.set_content(R"({"error":{"message":"model reload failed","type":"server_error"}})",
                                "application/json");
                return;
            }
            if (!voice_ref_codes.empty()) {
                result = t->synthesize_with_embedding(
                    input, voice_embedding.data(), (int32_t)voice_embedding.size(), params,
                    voice_ref_codes.data(), voice_n_ref_frames);
            } else if (!voice_embedding.empty()) {
                result = t->synthesize_with_embedding(
                    input, voice_embedding.data(), (int32_t)voice_embedding.size(), params);
            } else {
                result = t->synthesize(input, params);
            }
            last_used = std::chrono::steady_clock::now();
        }

        if (!result.success) {
            res.status = 500;
            json err = {{"error", {
                {"message", "synthesis failed: " + result.error_msg},
                {"type", "server_error"},
            }}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (result.audio.empty()) {
            res.status = 500;
            res.set_content(R"({"error":{"message":"synthesis produced no audio","type":"server_error"}})",
                            "application/json");
            return;
        }

        fprintf(stderr, "synthesized %.2fs audio (%zu samples) in %lldms\n",
                (float)result.audio.size() / result.sample_rate,
                result.audio.size(), (long long)result.t_total_ms);

        // one-shot (no stream_format): preserve legacy behavior
        if (stream_format.empty()) {
            if (response_format == "pcm") {
                res.set_content(encode_pcm(result.audio), "audio/pcm");
            } else if (response_format == "opus") {
                res.set_content(encode_opus(result.audio, result.sample_rate), "audio/ogg");
            } else if (response_format == "mp3") {
                res.set_content(encode_mp3(result.audio, result.sample_rate), "audio/mpeg");
            } else {
                res.set_content(encode_wav(result.audio, result.sample_rate), "audio/wav");
            }
            return;
        }

        // stream_format=audio: raw chunked bytes in the chosen response_format.
        // wav uses a placeholder-size header so playback can start immediately.
        // opus produces a self-contained Ogg stream (no separate header), and
        // mp3 is self-framing (each MPEG frame stands alone).
        if (stream_format == "audio") {
            std::string header;
            std::string body_bytes;
            const char * ctype = "audio/pcm";
            if (response_format == "wav") {
                header     = wav_streaming_header(result.sample_rate);
                body_bytes = encode_pcm(result.audio);
                ctype      = "audio/wav";
            } else if (response_format == "opus") {
                body_bytes = encode_opus(result.audio, result.sample_rate);
                ctype      = "audio/ogg";
            } else if (response_format == "mp3") {
                body_bytes = encode_mp3(result.audio, result.sample_rate);
                ctype      = "audio/mpeg";
            } else {
                body_bytes = encode_pcm(result.audio);
            }

            res.set_chunked_content_provider(ctype,
                [header = std::move(header), body_bytes = std::move(body_bytes)]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    if (!header.empty()) {
                        sink.write(header.data(), header.size());
                    }
                    sink.write(body_bytes.data(), body_bytes.size());
                    sink.done();
                    return false;
                });
            return;
        }

        // stream_format=sse: emit speech.audio.delta + speech.audio.done.
        // response_format still selects the bytes carried inside delta (wav,
        // pcm, or opus). usage/timings on the done event are shaped to be
        // consumed by both openai clients and llama-swap's metrics_monitor.
        {
            std::string audio_bytes;
            if (response_format == "wav") {
                audio_bytes = encode_wav(result.audio, result.sample_rate);
            } else if (response_format == "opus") {
                audio_bytes = encode_opus(result.audio, result.sample_rate);
            } else if (response_format == "mp3") {
                audio_bytes = encode_mp3(result.audio, result.sample_rate);
            } else {
                audio_bytes = encode_pcm(result.audio);
            }

            json delta = {
                {"type", "speech.audio.delta"},
                {"audio", base64_encode(audio_bytes.data(), audio_bytes.size())},
            };
            std::string delta_frame = "event: speech.audio.delta\ndata: " + delta.dump() + "\n\n";
            std::string done_frame  = "event: speech.audio.done\ndata: "
                                    + build_done_event(result)
                                    + "\n\n";

            res.set_chunked_content_provider("text/event-stream",
                [delta_frame = std::move(delta_frame), done_frame = std::move(done_frame)]
                (size_t /*offset*/, httplib::DataSink & sink) mutable -> bool {
                    sink.write(delta_frame.data(), delta_frame.size());
                    sink.write(done_frame.data(),  done_frame.size());
                    sink.done();
                    return false;
                });
            return;
        }
    });

    // idle watchdog: drops the model after idle_timeout_sec of inactivity.
    // try_lock avoids racing with an in-flight synth; if it can't grab the
    // lock, it just skips the tick and tries again next second.
    std::atomic<bool> watchdog_stop{false};
    std::thread       watchdog_thread;
    if (sp.idle_timeout_sec > 0) {
        watchdog_thread = std::thread([&watchdog_stop, &tts, &synth_mutex, &last_used, &sp]() {
            using namespace std::chrono;
            while (!watchdog_stop.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(seconds(1));
                if (watchdog_stop.load(std::memory_order_relaxed)) break;
                std::unique_lock<std::mutex> lock(synth_mutex, std::try_to_lock);
                if (!lock.owns_lock() || !tts) continue;
                auto idle = steady_clock::now() - last_used;
                if (idle >= seconds(sp.idle_timeout_sec)) {
                    fprintf(stderr, "idle %llds: unloading model\n",
                            (long long)duration_cast<seconds>(idle).count());
                    tts.reset();
                }
            }
        });
    }

    fprintf(stderr, "server listening on %s:%d\n", sp.host.c_str(), sp.port);
    bool listen_ok = svr.listen(sp.host, sp.port);

    // signal the watchdog to exit so it doesn't outlive the locals it captures.
    watchdog_stop.store(true, std::memory_order_relaxed);
    if (watchdog_thread.joinable()) watchdog_thread.join();

    if (!listen_ok) {
        fprintf(stderr, "fatal: failed to bind to %s:%d\n", sp.host.c_str(), sp.port);
        return 1;
    }

    return 0;
}
