// httplib.h must come before any header that reaches <windows.h> (common/log.h does): windows.h pulls in winsock.h,
// which clashes with the winsock2.h included by httplib.h.
#include "httplib.h"

#include "recovery.h"

#include <algorithm>
#include <ctime>
#include <exception>
#include <initializer_list>
#include <new>
#include <utility>

#include "common/log.h"

namespace {

    constexpr int kMaxCooldownSeconds           = 3600;
    constexpr int kRestoreBackoffSeconds        = 15;
    constexpr int kOutOfMemoryRetryAfterSeconds = 30;

    int64_t unix_now() {
        return static_cast<int64_t>(std::time(nullptr));
    }

    // The fields a profile changes, so that two levels that would load the same configuration compare equal.
    std::string describe(const SDContextParams& params) {
        std::string text = std::to_string(params.memory_guard) + '|' + params.backend + '|' + params.params_backend + '|';
        for (bool flag : {params.clip_on_cpu, params.vae_on_cpu, params.control_net_cpu, params.offload_params_to_cpu, params.batched_cfg, params.eager_load}) {
            text += flag ? '1' : '0';
        }
        return text;
    }

    bool has_accelerator() {
        const size_t count = sd_get_device_count();
        for (size_t i = 0; i < count; ++i) {
            sd_device_info_t info;
            if (sd_get_device_info(i, &info) && info.type != SD_DEVICE_TYPE_CPU) {
                return true;
            }
        }
        return false;
    }

    GenerationFailure capture_failure() {
        GenerationFailure failure;
        char message[1024];
        message[0]      = '\0';
        failure.code    = sd_get_last_error(message, sizeof(message));
        failure.message = message;
        if (failure.code == SD_ERROR_NONE) {
            failure.code = SD_ERROR_UNKNOWN;
        }
        if (failure.message.empty()) {
            failure.message = "the generation failed without a reported reason";
        }
        if (failure.code == SD_ERROR_OUT_OF_MEMORY) {
            failure.retry_after_seconds = kOutOfMemoryRetryAfterSeconds;
        }
        return failure;
    }

}  // namespace

bool failure_is_recoverable(sd_error_code_t code) {
    return code == SD_ERROR_OUT_OF_MEMORY || code == SD_ERROR_BACKEND;
}

const char* failure_code_name(const GenerationFailure& failure) {
    if (failure.unavailable) {
        return "unavailable";
    }
    switch (failure.code) {
        case SD_ERROR_NONE:
        case SD_ERROR_UNKNOWN:
            return "generation_failed";
        default:
            return sd_error_code_name(failure.code);
    }
}

int failure_http_status(const GenerationFailure& failure) {
    if (failure.unavailable) {
        return 503;
    }
    switch (failure.code) {
        case SD_ERROR_OUT_OF_MEMORY:
        case SD_ERROR_BUSY:
            return 503;
        case SD_ERROR_INVALID_ARGUMENT:
            return 400;
        default:
            return 500;
    }
}

GenerationFailure out_of_memory_failure() {
    GenerationFailure failure;
    failure.code                = SD_ERROR_OUT_OF_MEMORY;
    failure.message             = "the server ran out of memory while handling the request";
    failure.retry_after_seconds = kOutOfMemoryRetryAfterSeconds;
    return failure;
}

GenerationFailure failure_from_exception(const std::exception& error) {
    if (dynamic_cast<const std::bad_alloc*>(&error) != nullptr) {
        return out_of_memory_failure();
    }
    GenerationFailure failure;
    failure.code    = SD_ERROR_UNKNOWN;
    failure.message = error.what();
    return failure;
}

std::string safe_json_dump(const nlohmann::json& value) {
    return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

void send_generation_failure(httplib::Response& res, const GenerationFailure& failure) {
    res.status = failure_http_status(failure);

    nlohmann::json body;
    body["error"] = failure.message.empty() ? std::string("the generation failed") : failure.message;
    body["code"]  = failure_code_name(failure);
    if (failure.unavailable) {
        body["cause"] = sd_error_code_name(failure.code);
    }
    if (res.status == 503) {
        const int wait              = std::max(1, failure.retry_after_seconds > 0 ? failure.retry_after_seconds : kOutOfMemoryRetryAfterSeconds);
        body["retry_after_seconds"] = wait;
        res.set_header("Retry-After", std::to_string(wait));
    }
    res.set_content(safe_json_dump(body), "application/json");
}

static ContextKeeper::Hooks with_defaults(ContextKeeper::Hooks hooks) {
    if (!hooks.create_context) {
        hooks.create_context = [](const sd_ctx_params_t& params, GenerationFailure& failure) {
            SDCtxPtr ctx(new_sd_ctx(&params));
            if (ctx == nullptr) {
                failure = capture_failure();
            }
            return ctx;
        };
    }
    if (!hooks.read_failure) {
        hooks.read_failure = capture_failure;
    }
    if (!hooks.has_accelerator) {
        hooks.has_accelerator = has_accelerator;
    }
    if (!hooks.now) {
        hooks.now = [] { return ContextKeeper::Clock::now(); };
    }
    if (!hooks.cancel_context) {
        hooks.cancel_context = [](sd_ctx_t* ctx) { sd_cancel_generation(ctx, SD_CANCEL_ALL); };
    }
    return hooks;
}

ContextKeeper::ContextKeeper(const SDContextParams& base_params,
                             sd_profile::Level base_level,
                             const Settings& settings,
                             const Hooks& hooks)
    : settings_(settings),
      hooks_(with_defaults(hooks)),
      base_params_(base_params),
      base_level_(base_level),
      cooldown_seconds_(std::max(settings.cooldown_seconds, 0)),
      started_at_(hooks_.now()) {
    backend_choice_.backend               = base_params.backend;
    backend_choice_.params_backend        = base_params.params_backend;
    backend_choice_.diffusion_conv_direct = base_params.diffusion_conv_direct;
    backend_choice_.n_threads             = base_params.n_threads;
    build_ladder();
}

std::string ContextKeeper::describe_ladder() const {
    std::string text;
    for (sd_profile::Level level : ladder_) {
        text += text.empty() ? "" : " > ";
        text += sd_profile::level_name(level);
    }
    return text;
}

std::unique_ptr<SDContextParams> ContextKeeper::params_for(sd_profile::Level level) const {
    auto params = std::make_unique<SDContextParams>(base_params_);
    if (level > base_level_) {
        SDGenerationParams unused;
        sd_profile::apply(level, *params, unused, /*respect_explicit=*/false);
    }
    return params;
}

bool ContextKeeper::force_tiling_at(sd_profile::Level level) const {
    return level > base_level_ && level >= sd_profile::Level::LITE;
}

std::string ContextKeeper::fingerprint(sd_profile::Level level) const {
    return describe(*params_for(level)) + (force_tiling_at(level) ? "|tiled" : "");
}

void ContextKeeper::build_ladder() {
    ladder_.push_back(base_level_);
    std::string previous    = fingerprint(base_level_);
    const bool accelerator  = hooks_.has_accelerator();
    sd_profile::Level level = base_level_;
    while (level != sd_profile::Level::LITE_CPU) {
        level = sd_profile::lighter(level);
        if (level == sd_profile::Level::LITE_CPU && !accelerator) {
            break;
        }
        const std::string current = fingerprint(level);
        if (current == previous) {
            continue;
        }
        ladder_.push_back(level);
        previous = current;
    }
}

bool ContextKeeper::start(GenerationFailure& failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure = GenerationFailure();
    if (!build_from(0, failure)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> state(state_mutex_);
        if (index_ > 0) {
            next_promotion_ = hooks_.now() + std::chrono::seconds(cooldown_seconds_);
        }
    }
    model_loaded_ = true;
    return true;
}

void ContextKeeper::release_context() {
    ctx_.reset();
    ctx_params_.reset();
    std::lock_guard<std::mutex> state(state_mutex_);
    has_ctx_ = false;
}

void ContextKeeper::record_error(const GenerationFailure& failure) {
    if (failure.code == SD_ERROR_NONE || failure.code == SD_ERROR_INVALID_ARGUMENT || failure.code == SD_ERROR_CANCELLED) {
        return;
    }
    std::lock_guard<std::mutex> state(state_mutex_);
    last_error_.code    = failure_code_name(failure);
    last_error_.message = failure.message;
    last_error_.at      = unix_now();
}

void ContextKeeper::note_build_failure(const GenerationFailure& failure) {
    std::lock_guard<std::mutex> state(state_mutex_);
    restore_not_before_ = hooks_.now() + std::chrono::seconds(kRestoreBackoffSeconds);
    restore_failure_    = failure;
}

bool ContextKeeper::try_build(size_t index, GenerationFailure& failure) {
    const sd_profile::Level level           = ladder_[index];
    std::unique_ptr<SDContextParams> params = params_for(level);
    // The context may keep pointing into these strings, so they stay where they are for as long as it lives.
    sd_ctx_params_t raw = params->to_sd_ctx_params_t(false);

    LOG_INFO("loading the model with profile %s", sd_profile::level_name(level));
    SDCtxPtr ctx = hooks_.create_context(raw, failure);
    if (ctx == nullptr) {
        if (failure.code == SD_ERROR_NONE) {
            failure.code    = SD_ERROR_UNKNOWN;
            failure.message = "the model could not be loaded";
        }
        record_error(failure);
        LOG_WARN("the model could not be loaded with profile %s: %s: %s",
                 sd_profile::level_name(level),
                 sd_error_code_name(failure.code),
                 failure.message.c_str());
        return false;
    }

    const bool image = sd_ctx_supports_image_generation(ctx.get());
    const bool video = sd_ctx_supports_video_generation(ctx.get());
    ctx_             = std::move(ctx);
    ctx_params_      = std::move(params);
    {
        std::lock_guard<std::mutex> state(state_mutex_);
        index_   = index;
        has_ctx_ = true;
        if (model_loaded_) {
            ++counters_.rebuilds;
        }
        backend_choice_.backend               = ctx_params_->backend;
        backend_choice_.params_backend        = ctx_params_->params_backend;
        backend_choice_.diffusion_conv_direct = ctx_params_->diffusion_conv_direct;
        backend_choice_.n_threads             = ctx_params_->n_threads;
    }
    supports_image_ = image;
    supports_video_ = video;
    LOG_INFO("the model is loaded (profile %s)", sd_profile::level_name(level));
    return true;
}

bool ContextKeeper::build_from(size_t index, GenerationFailure& failure) {
    {
        std::lock_guard<std::mutex> state(state_mutex_);
        rebuilding_ = true;
    }
    struct Done {
        ContextKeeper& keeper;
        ~Done() {
            std::lock_guard<std::mutex> state(keeper.state_mutex_);
            keeper.rebuilding_ = false;
        }
    } done{*this};

    while (index < ladder_.size()) {
        if (try_build(index, failure)) {
            return true;
        }
        if (!settings_.degradation || !failure_is_recoverable(failure.code)) {
            return false;
        }
        // A failing backend is not cured by trimming memory: go straight to the profile that avoids the devices.
        size_t next = index + 1;
        if (failure.code == SD_ERROR_BACKEND && ladder_.size() > next) {
            next = ladder_.size() - 1;
        }
        if (next < ladder_.size()) {
            LOG_WARN("trying profile %s instead of %s", sd_profile::level_name(ladder_[next]), sd_profile::level_name(ladder_[index]));
        }
        index = next;
    }
    return false;
}

bool ContextKeeper::restore(GenerationFailure& failure) {
    size_t index = 0;
    {
        std::lock_guard<std::mutex> state(state_mutex_);
        const auto now = hooks_.now();
        if (now < restore_not_before_) {
            failure             = restore_failure_;
            failure.unavailable = true;
            failure.message     = "the model is not loaded and the last attempt to load it failed: " + restore_failure_.message;
            failure.retry_after_seconds =
                static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(restore_not_before_ - now).count()) + 1;
            return false;
        }
        index = index_;
    }

    LOG_WARN("the model is not loaded; loading it again");
    GenerationFailure rebuild_failure;
    if (build_from(index, rebuild_failure)) {
        std::lock_guard<std::mutex> state(state_mutex_);
        if (index_ > 0) {
            next_promotion_ = hooks_.now() + std::chrono::seconds(cooldown_seconds_);
        }
        return true;
    }
    note_build_failure(rebuild_failure);
    failure                     = rebuild_failure;
    failure.unavailable         = true;
    failure.retry_after_seconds = kRestoreBackoffSeconds;
    return false;
}

bool ContextKeeper::step_down(GenerationFailure& failure) {
    size_t from = 0;
    {
        std::lock_guard<std::mutex> state(state_mutex_);
        from = index_;
    }
    const size_t last = ladder_.size() - 1;
    if (from >= last) {
        LOG_ERROR("%s with profile %s and no lighter profile is left: %s",
                  sd_error_code_name(failure.code),
                  sd_profile::level_name(ladder_[from]),
                  failure.message.c_str());
        return false;
    }

    const size_t to = failure.code == SD_ERROR_BACKEND ? last : from + 1;
    LOG_WARN("%s with profile %s: %s; loading the model again with profile %s to run the request again",
             sd_error_code_name(failure.code),
             sd_profile::level_name(ladder_[from]),
             failure.message.c_str(),
             sd_profile::level_name(ladder_[to]));
    {
        std::lock_guard<std::mutex> state(state_mutex_);
        ++counters_.step_downs;
        if (probation_) {
            probation_ = false;
            ++counters_.failed_promotions;
            cooldown_seconds_ = std::min(cooldown_seconds_ * 2, std::max(kMaxCooldownSeconds, settings_.cooldown_seconds));
        }
        next_promotion_ = hooks_.now() + std::chrono::seconds(cooldown_seconds_);
    }

    // The old context goes first: both would not fit.
    release_context();
    GenerationFailure rebuild_failure;
    if (!build_from(to, rebuild_failure)) {
        {
            // The next attempt to load the model starts from the profile that was just tried.
            std::lock_guard<std::mutex> state(state_mutex_);
            index_ = to;
        }
        note_build_failure(rebuild_failure);
        failure                     = rebuild_failure;
        failure.unavailable         = true;
        failure.retry_after_seconds = kRestoreBackoffSeconds;
        return false;
    }
    return true;
}

void ContextKeeper::promote_if_due() {
    if (!settings_.degradation || settings_.cooldown_seconds <= 0 || ctx_ == nullptr) {
        return;
    }
    size_t index = 0;
    {
        std::lock_guard<std::mutex> state(state_mutex_);
        if (index_ == 0 || hooks_.now() < next_promotion_) {
            return;
        }
        index = index_;
    }

    const size_t target = index - 1;
    LOG_INFO("trying profile %s again", sd_profile::level_name(ladder_[target]));
    release_context();
    GenerationFailure failure;
    bool built = build_from(target, failure);
    if (!built) {
        // The profile in use before the climb worked, so give it another go before giving up.
        built = build_from(index, failure);
    }

    if (!built) {
        note_build_failure(failure);
        return;
    }
    std::lock_guard<std::mutex> state(state_mutex_);
    if (index_ <= target) {
        ++counters_.promotions;
        probation_ = true;
    } else {
        ++counters_.failed_promotions;
        cooldown_seconds_ = std::min(cooldown_seconds_ * 2, std::max(kMaxCooldownSeconds, settings_.cooldown_seconds));
    }
    next_promotion_ = hooks_.now() + std::chrono::seconds(cooldown_seconds_);
}

bool ContextKeeper::cancel(CancelToken& token) {
    token.requested = true;
    std::lock_guard<std::mutex> state(state_mutex_);
    if (running_token_ != &token || running_ctx_ == nullptr) {
        return false;
    }
    hooks_.cancel_context(running_ctx_);
    return true;
}

bool ContextKeeper::run(const Attempt& attempt, GenerationFailure& failure, const CancelToken* token) {
    std::lock_guard<std::mutex> lock(mutex_);
    failure = GenerationFailure();

    struct Active {
        ContextKeeper& keeper;
        explicit Active(ContextKeeper& owner)
            : keeper(owner) {
            std::lock_guard<std::mutex> state(keeper.state_mutex_);
            ++keeper.active_runs_;
            ++keeper.counters_.requests;
        }
        ~Active() {
            std::lock_guard<std::mutex> state(keeper.state_mutex_);
            --keeper.active_runs_;
        }
    } active(*this);

    auto fail = [&]() {
        std::lock_guard<std::mutex> state(state_mutex_);
        ++counters_.failed;
        return false;
    };

    auto wanted_stop = [&]() {
        return token != nullptr && token->requested.load();
    };
    auto stop = [&]() {
        failure         = GenerationFailure();
        failure.code    = SD_ERROR_CANCELLED;
        failure.message = "the generation was cancelled";
        std::lock_guard<std::mutex> state(state_mutex_);
        ++counters_.cancelled;
        return false;
    };

    // The context only exists to the library while an attempt runs, so cancel() can reach it only then.
    struct Running {
        ContextKeeper& keeper;
        Running(ContextKeeper& owner, const CancelToken* token, sd_ctx_t* ctx)
            : keeper(owner) {
            std::lock_guard<std::mutex> state(keeper.state_mutex_);
            keeper.running_token_ = token;
            keeper.running_ctx_   = ctx;
        }
        ~Running() {
            std::lock_guard<std::mutex> state(keeper.state_mutex_);
            keeper.running_token_ = nullptr;
            keeper.running_ctx_   = nullptr;
        }
    };

    if (!model_loaded_) {
        failure.code    = SD_ERROR_INVALID_ARGUMENT;
        failure.message = "no model loaded";
        return fail();
    }

    // Before promote_if_due(): a job nobody wants any more must not cost a reload of the model.
    if (wanted_stop()) {
        return stop();
    }

    promote_if_due();
    if (ctx_ == nullptr && !restore(failure)) {
        return fail();
    }

    bool stepped_down = false;
    // The wait before the next climb counts from the end of a request that had to step down, not from its
    // start: reloading the model and generating on a lighter profile can take longer than the wait itself.
    struct Rearm {
        ContextKeeper& keeper;
        const bool& armed;
        ~Rearm() {
            if (armed) {
                std::lock_guard<std::mutex> state(keeper.state_mutex_);
                keeper.next_promotion_ = keeper.hooks_.now() + std::chrono::seconds(keeper.cooldown_seconds_);
            }
        }
    } rearm{*this, stepped_down};

    for (;;) {
        if (wanted_stop()) {
            return stop();
        }

        ContextView view;
        view.ctx = ctx_.get();
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            view.level = ladder_[index_];
        }
        view.force_vae_tiling = force_tiling_at(view.level);

        bool generated = false;
        {
            Running running(*this, token, view.ctx);
            generated = attempt(view);
        }
        if (generated) {
            std::lock_guard<std::mutex> state(state_mutex_);
            ++counters_.succeeded;
            if (stepped_down) {
                ++counters_.recovered;
            }
            if (probation_) {
                probation_        = false;
                cooldown_seconds_ = std::max(settings_.cooldown_seconds, 0);
                LOG_INFO("profile %s works; it stays", sd_profile::level_name(view.level));
            }
            return true;
        }

        failure = hooks_.read_failure();
        if (failure.code == SD_ERROR_CANCELLED) {
            return stop();
        }
        record_error(failure);
        // Reloading the model on a lighter profile to run a request nobody waits for any more would only delay the next one.
        if (wanted_stop() && failure_is_recoverable(failure.code)) {
            return stop();
        }
        if (!settings_.degradation || !failure_is_recoverable(failure.code)) {
            return fail();
        }
        if (!step_down(failure)) {
            return fail();
        }
        stepped_down = true;
    }
}

ContextKeeper::BackendChoice ContextKeeper::backend_choice() const {
    std::lock_guard<std::mutex> state(state_mutex_);
    return backend_choice_;
}

ContextKeeper::Status ContextKeeper::status_locked() const {
    if (!model_loaded_) {
        return Status::NO_MODEL;
    }
    if (rebuilding_) {
        return Status::RECOVERING;
    }
    if (!has_ctx_) {
        return Status::UNAVAILABLE;
    }
    return index_ > 0 ? Status::DEGRADED : Status::OK;
}

const char* ContextKeeper::status_text(Status status) {
    switch (status) {
        case Status::NO_MODEL:
            return "no_model";
        case Status::OK:
            return "ok";
        case Status::DEGRADED:
            return "degraded";
        case Status::RECOVERING:
            return "recovering";
        case Status::UNAVAILABLE:
            return "unavailable";
    }
    return "ok";
}

const char* ContextKeeper::status_name() const {
    std::lock_guard<std::mutex> state(state_mutex_);
    return status_text(status_locked());
}

nlohmann::json ContextKeeper::health_json() const {
    std::lock_guard<std::mutex> state(state_mutex_);
    nlohmann::json out;

    const auto now = hooks_.now();

    out["status"]       = status_text(status_locked());
    out["model_loaded"] = model_loaded_.load();
    out["profile"]      = model_loaded_ ? nlohmann::json(sd_profile::level_name(ladder_[index_])) : nlohmann::json(nullptr);
    out["base_profile"] = sd_profile::level_name(base_level_);
    out["degradation"]  = settings_.degradation;
    out["busy"]         = active_runs_ > 0;
    out["counters"]     = {
        {"requests", counters_.requests},
        {"succeeded", counters_.succeeded},
        {"failed", counters_.failed},
        {"cancelled", counters_.cancelled},
        {"recovered", counters_.recovered},
        {"step_downs", counters_.step_downs},
        {"promotions", counters_.promotions},
        {"failed_promotions", counters_.failed_promotions},
        {"rebuilds", counters_.rebuilds},
    };
    if (last_error_.code.empty()) {
        out["last_error"] = nullptr;
    } else {
        out["last_error"] = {
            {"code", last_error_.code},
            {"message", last_error_.message},
            {"at", last_error_.at},
        };
    }
    if (model_loaded_ && has_ctx_ && index_ > 0 && settings_.degradation && settings_.cooldown_seconds > 0) {
        const auto remaining          = std::chrono::duration_cast<std::chrono::seconds>(next_promotion_ - now).count();
        out["next_promotion_seconds"] = std::max<int64_t>(remaining, 0);
    } else {
        out["next_promotion_seconds"] = nullptr;
    }
    out["uptime_seconds"] = std::chrono::duration_cast<std::chrono::seconds>(now - started_at_).count();
    out["version"]        = version_string();
    return out;
}
