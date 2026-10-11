#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <json.hpp>
#include "common/common.h"
#include "common/profile.h"
#include "common/resource_owners.hpp"
#include "stable-diffusion.h"

namespace httplib {
    struct Response;
}

// Why a generation could not be served, in terms a client can act on.
struct GenerationFailure {
    sd_error_code_t code = SD_ERROR_NONE;
    std::string message;
    // No usable context at this moment (it is being rebuilt, or could not be); code holds the cause.
    bool unavailable = false;
    // How long a client should wait before trying again; 0 when waiting is not expected to help.
    int retry_after_seconds = 0;
};

// Out of memory and backend failures are the ones a lighter configuration can get around.
bool failure_is_recoverable(sd_error_code_t code);
// "out_of_memory", "backend", "model_file", "invalid_argument", "unavailable", ..., and
// "generation_failed" for failures without a more specific cause.
const char* failure_code_name(const GenerationFailure& failure);
int failure_http_status(const GenerationFailure& failure);
GenerationFailure out_of_memory_failure();
GenerationFailure failure_from_exception(const std::exception& error);
// 503 (with Retry-After) for lack of memory or of a context, 400 for invalid input, 500 otherwise.
void send_generation_failure(httplib::Response& res, const GenerationFailure& failure);
// Serializes without ever throwing on text that is not valid UTF-8 (library messages can carry file names).
std::string safe_json_dump(const nlohmann::json& value);

// Owns the diffusion context of the server and keeps it serving when memory runs short.
//
// A request that fails because the machine ran out of memory (or because a backend failed) makes the
// keeper free the context, load the model again with the next lighter profile of the ladder
// max > balanced > lite > lite-disk > lite-cpu, and run the same request again. Once the machine has been
// quiet for a while, it climbs back one step at a time; a climb that fails doubles the wait.
//
// The mutex returned by mutex() serializes everything that uses the compute devices: it is held for
// the whole of a run() and by the upscale endpoint. Reading the health or the resources never takes it.
class ContextKeeper {
public:
    using Clock = std::chrono::steady_clock;

    struct Settings {
        bool degradation     = true;
        int cooldown_seconds = 300;  // wait before climbing back one step; 0 = stay where the server stepped down
    };

    // Everything that touches the library or the machine, replaceable so that the logic can be tested
    // without a model. An empty function selects the real behavior.
    struct Hooks {
        // Loads the model; returns null and describes the reason when it cannot. Default: new_sd_ctx.
        std::function<SDCtxPtr(const sd_ctx_params_t&, GenerationFailure&)> create_context;
        // The reason the attempt that just returned false failed. Default: sd_get_last_error().
        std::function<GenerationFailure()> read_failure;
        // Whether a device other than the CPU is present. Default: asks the library.
        std::function<bool()> has_accelerator;
        std::function<Clock::time_point()> now;
    };

    // What an attempt needs to run one generation.
    struct ContextView {
        sd_ctx_t* ctx           = nullptr;
        sd_profile::Level level = sd_profile::Level::MAX;
        // The server stepped down to a lite profile: decode the VAE in tiles whatever the request asked.
        bool force_vae_tiling = false;
    };

    // Runs one generation. Must return false only when the generation itself failed, and must not call
    // the library again after that call: the reason is read on the same thread right after it returns.
    using Attempt = std::function<bool(const ContextView&)>;

    ContextKeeper(const SDContextParams& base_params,
                  sd_profile::Level base_level,
                  const Settings& settings,
                  const Hooks& hooks = Hooks());
    ContextKeeper(const ContextKeeper&)            = delete;
    ContextKeeper& operator=(const ContextKeeper&) = delete;

    // Loads the model, stepping down the ladder when the memory is not enough. Returns false when no
    // profile could load it; failure holds the reason of the last attempt.
    bool start(GenerationFailure& failure);

    std::mutex& mutex() {
        return mutex_;
    }

    // Safe to call from any thread, without the mutex. Both stay true while the context is being rebuilt.
    bool has_model() const {
        return model_loaded_.load();
    }
    bool supports_image_generation() const {
        return supports_image_.load();
    }
    bool supports_video_generation() const {
        return supports_video_.load();
    }

    // Runs attempt on the context, recovering from a lack of memory as described above. Returns true when
    // an attempt succeeded.
    bool run(const Attempt& attempt, GenerationFailure& failure);

    // The settings the context in use was created with; for the upscale endpoint, which loads its own model.
    struct BackendChoice {
        std::string backend;
        std::string params_backend;
        bool diffusion_conv_direct = false;
        int n_threads              = -1;
    };
    BackendChoice backend_choice() const;

    // The profiles the server can step down through, e.g. "max > lite > lite-cpu".
    std::string describe_ladder() const;
    bool can_step_down() const {
        return settings_.degradation && ladder_.size() > 1;
    }

    nlohmann::json health_json() const;
    // "ok", "degraded", "recovering", "unavailable" or "no_model"
    const char* status_name() const;

private:
    enum class Status { NO_MODEL,
                        OK,
                        DEGRADED,
                        RECOVERING,
                        UNAVAILABLE };

    struct Counters {
        uint64_t requests          = 0;
        uint64_t succeeded         = 0;
        uint64_t failed            = 0;
        uint64_t recovered         = 0;  // requests that succeeded after a step down
        uint64_t step_downs        = 0;
        uint64_t promotions        = 0;
        uint64_t failed_promotions = 0;
        uint64_t rebuilds          = 0;
    };

    struct LastError {
        std::string code;
        std::string message;
        int64_t at = 0;
    };

    std::unique_ptr<SDContextParams> params_for(sd_profile::Level level) const;
    std::string fingerprint(sd_profile::Level level) const;
    bool force_tiling_at(sd_profile::Level level) const;
    void build_ladder();

    bool build_from(size_t index, GenerationFailure& failure);
    bool try_build(size_t index, GenerationFailure& failure);
    bool restore(GenerationFailure& failure);
    bool step_down(GenerationFailure& failure);
    void promote_if_due();
    void release_context();
    void note_build_failure(const GenerationFailure& failure);
    void record_error(const GenerationFailure& failure);
    Status status_locked() const;
    static const char* status_text(Status status);

    // Settings and the ladder do not change after construction.
    const Settings settings_;
    const Hooks hooks_;
    const SDContextParams base_params_;
    const sd_profile::Level base_level_;
    std::vector<sd_profile::Level> ladder_;

    // Held for the whole of a run(); guards ctx_ and the parameters it was created with.
    std::mutex mutex_;
    SDCtxPtr ctx_;
    std::unique_ptr<SDContextParams> ctx_params_;

    // Small lock for the figures health reads; always taken after mutex_ when both are needed.
    mutable std::mutex state_mutex_;
    size_t index_         = 0;  // position in ladder_
    bool has_ctx_         = false;
    bool rebuilding_      = false;
    bool probation_       = false;  // the context in use is a promotion nobody has tried yet
    int cooldown_seconds_ = 300;
    Clock::time_point next_promotion_;
    Clock::time_point restore_not_before_;
    GenerationFailure restore_failure_;  // why the last attempt to load the model again failed
    Counters counters_;
    LastError last_error_;
    BackendChoice backend_choice_;
    const Clock::time_point started_at_;
    int active_runs_ = 0;

    std::atomic<bool> model_loaded_{false};
    std::atomic<bool> supports_image_{false};
    std::atomic<bool> supports_video_{false};
};
