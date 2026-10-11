// Test of the context keeper of the server (examples/server/recovery.*): the ladder of profiles, stepping
// down when a request runs out of memory, climbing back, loading the model again after a failure, the
// HTTP answers for failures, and use from several threads. No model is needed: the library is replaced by
// scripted answers and the clock by a variable.
// Build (from the repository root, after building the library into build/); the sources go in one command:
//   cc -c thirdparty/zip.c -o zip.o
//   c++ -std=c++17 -Iexamples -Iexamples/server -Iinclude -Ithirdparty -Isrc test/server_recovery_test.cpp
//       examples/server/recovery.cpp examples/common/common.cpp examples/common/log.cpp
//       examples/common/media_io.cpp examples/common/profile.cpp zip.o
//       -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o server_recovery_test
// Run: ./server_recovery_test ; exit code 0 = all checks passed.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"
#include "recovery.h"

using Level = sd_profile::Level;
using Clock = ContextKeeper::Clock;
using json  = nlohmann::json;

static int failures = 0;
#define CHECK(cond, msg)               \
    do {                               \
        if (!(cond)) {                 \
            printf("FAIL: %s\n", msg); \
            failures++;                \
        } else {                       \
            printf("ok:   %s\n", msg); \
        }                              \
    } while (0)

static GenerationFailure failure_of(sd_error_code_t code, const char* message) {
    GenerationFailure failure;
    failure.code    = code;
    failure.message = message;
    return failure;
}
static GenerationFailure oom() {
    return failure_of(SD_ERROR_OUT_OF_MEMORY, "unet ran out of memory");
}
static GenerationFailure backend_failure() {
    return failure_of(SD_ERROR_BACKEND, "the backend failed");
}
static GenerationFailure model_file_failure() {
    return failure_of(SD_ERROR_MODEL_FILE, "failed to open the model");
}
static const GenerationFailure ok_result;

// What the keeper asked the library to load, reduced to the options a profile changes.
struct Load {
    int memory_guard;
    bool batched_cfg;
    std::string backend;
    std::string params_backend;
    bool eager_load;

    Level level() const {
        if (backend == "cpu") {
            return Level::LITE_CPU;
        }
        if (params_backend == "disk") {
            return Level::LITE_DISK;
        }
        if (memory_guard == 85) {
            return Level::LITE;
        }
        if (memory_guard == 90) {
            return Level::BALANCED;
        }
        return Level::MAX;
    }
};

// A scripted library and clock.
struct Env {
    std::vector<Load> loads;
    std::function<GenerationFailure(const Load&)> load_result;  // code NONE = the model loads
    GenerationFailure next_failure;                             // what the library reports for the attempt that failed
    Clock::time_point now = Clock::now();
    bool accelerator      = true;

    ContextKeeper::Hooks hooks() {
        ContextKeeper::Hooks hooks;
        hooks.create_context = [this](const sd_ctx_params_t& params, GenerationFailure& failure) -> SDCtxPtr {
            Load load{params.memory_guard,
                      params.batched_cfg,
                      params.backend != nullptr ? params.backend : "",
                      params.params_backend != nullptr ? params.params_backend : "",
                      params.eager_load};
            loads.push_back(load);
            const GenerationFailure result = load_result ? load_result(load) : ok_result;
            if (result.code != SD_ERROR_NONE) {
                failure = result;
                return nullptr;
            }
            // free_sd_ctx() accepts a context that holds no model.
            return SDCtxPtr(static_cast<sd_ctx_t*>(calloc(1, 64)));
        };
        hooks.read_failure    = [this] { return next_failure; };
        hooks.has_accelerator = [this] { return accelerator; };
        hooks.now             = [this] { return now; };
        return hooks;
    }

    void advance(int seconds) {
        now += std::chrono::seconds(seconds);
    }
};

struct Seen {
    std::vector<Level> levels;
    std::vector<bool> tiled;
    std::vector<sd_ctx_t*> contexts;
};

using Script = std::function<GenerationFailure(const ContextKeeper::ContextView&)>;

static bool run_with(ContextKeeper& keeper, Env& env, Seen& seen, const Script& script, GenerationFailure& failure) {
    return keeper.run(
        [&](const ContextKeeper::ContextView& view) {
            seen.levels.push_back(view.level);
            seen.tiled.push_back(view.force_vae_tiling);
            seen.contexts.push_back(view.ctx);
            const GenerationFailure result = script(view);
            if (result.code == SD_ERROR_NONE) {
                return true;
            }
            env.next_failure = result;
            return false;
        },
        failure);
}

static SDContextParams base_params() {
    SDContextParams params;
    params.model_path = "model.safetensors";
    params.n_threads  = 2;
    return params;
}

static ContextKeeper::Settings settings(bool degradation = true, int cooldown = 10) {
    ContextKeeper::Settings s;
    s.degradation      = degradation;
    s.cooldown_seconds = cooldown;
    return s;
}

static int64_t counter(const json& health, const char* name) {
    return health["counters"][name].get<int64_t>();
}

static const GenerationFailure succeed(const ContextKeeper::ContextView&) {
    return ok_result;
}

static void test_ladder() {
    Env env;
    {
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > balanced > lite > lite-disk > lite-cpu", "ladder with an accelerator");
        CHECK(keeper.can_step_down(), "a full ladder can step down");
    }
    env.accelerator = false;
    {
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > balanced > lite > lite-disk", "no cpu profile without an accelerator");
    }

    // The configuration of a host that already asks for most of it: steps that change nothing are left out.
    env.accelerator              = true;
    SDContextParams explicit_set = base_params();
    explicit_set.memory_guard    = 90;
    explicit_set.clip_on_cpu     = true;
    explicit_set.params_backend  = "disk";
    {
        ContextKeeper keeper(explicit_set, Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > lite > lite-cpu", "profiles that would load the same configuration are skipped");
    }
    {
        ContextKeeper keeper(base_params(), Level::LITE, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "lite > lite-disk > lite-cpu", "the ladder starts at the profile the server was started with");
    }
    {
        // Reading every weight into memory at load time is what streaming them from disk avoids.
        SDContextParams eager = explicit_set;
        eager.eager_load      = true;
        ContextKeeper keeper(eager, Level::MAX, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "max > lite > lite-disk > lite-cpu",
              "lite-disk is kept when it only differs by not loading every weight at start");
    }
    {
        ContextKeeper keeper(base_params(), Level::LITE_CPU, settings(), env.hooks());
        CHECK(keeper.describe_ladder() == "lite-cpu" && !keeper.can_step_down(), "nothing is lighter than the cpu profile");
    }
    {
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        CHECK(!keeper.can_step_down(), "no stepping down when degradation is disabled");
    }
}

static void test_start() {
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(!keeper.has_model(), "no model before start");
        CHECK(keeper.status_name() == std::string("no_model"), "status no_model before start");
        CHECK(keeper.start(failure), "start loads the model");
        CHECK(keeper.has_model(), "model after start");
        CHECK(env.loads.size() == 1 && env.loads[0].level() == Level::MAX, "one load, with the profile of the server");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["profile"] == "max" && health["base_profile"] == "max", "healthy after start");
        CHECK(health["next_promotion_seconds"].is_null(), "nothing to climb back to at the base profile");
        CHECK(health["last_error"].is_null(), "no error yet");
        CHECK(health["model_loaded"] == true && health["degradation"] == true, "model_loaded and degradation reported");
    }
    {
        Env env;
        env.load_result = [](const Load& load) { return load.level() < Level::LITE ? oom() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure), "start steps down when the memory is not enough");
        CHECK(env.loads.size() == 3 && env.loads[2].level() == Level::LITE, "max, balanced and then lite were tried");
        const json health = keeper.health_json();
        CHECK(health["status"] == "degraded" && health["profile"] == "lite" && health["base_profile"] == "max",
              "degraded after a start that had to step down");
        CHECK(health["next_promotion_seconds"] == 10, "the climb back is scheduled");
        CHECK(health["last_error"]["code"] == "out_of_memory", "the lack of memory is the last error");
    }
    {
        // The weights cannot be streamed from disk if the load reads all of them into memory first.
        Env env;
        env.load_result       = [](const Load& load) { return load.eager_load ? oom() : ok_result; };
        SDContextParams eager = base_params();
        eager.eager_load      = true;
        ContextKeeper keeper(eager, Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && env.loads.size() == 4 && env.loads[3].level() == Level::LITE_DISK && !env.loads[3].eager_load,
              "a server started with --eager-load reaches the disk profile without the eager read");
        CHECK(env.loads[0].eager_load && env.loads[2].eager_load, "the lighter profiles above it keep the option");
    }
    {
        Env env;
        env.load_result = [](const Load&) { return model_file_failure(); };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(!keeper.start(failure) && env.loads.size() == 1, "a damaged model file is not a reason to step down");
        CHECK(failure.code == SD_ERROR_MODEL_FILE, "the failure of the start is reported");
    }
    {
        Env env;
        env.load_result = [](const Load&) { return oom(); };
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        CHECK(!keeper.start(failure) && env.loads.size() == 1, "with degradation disabled the start gives up at once");
    }
    {
        Env env;
        env.load_result = [](const Load& load) { return load.level() == Level::MAX ? backend_failure() : ok_result; };
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        CHECK(keeper.start(failure) && env.loads.size() == 2 && env.loads[1].level() == Level::LITE_CPU,
              "a backend failure at start goes straight to the cpu profile");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        Seen seen;
        GenerationFailure failure;
        CHECK(!run_with(keeper, env, seen, succeed, failure) && failure.code == SD_ERROR_INVALID_ARGUMENT,
              "running without a model is refused");
    }
}

static void test_step_down() {
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(run_with(keeper, env, seen, succeed, failure), "a request that fits succeeds");
        CHECK(seen.levels.size() == 1 && seen.levels[0] == Level::MAX && !seen.tiled[0] && seen.contexts[0] != nullptr,
              "it ran once, at max, without forced tiling");
        const json health = keeper.health_json();
        CHECK(counter(health, "requests") == 1 && counter(health, "succeeded") == 1 && counter(health, "step_downs") == 0,
              "counted as a plain success");
        CHECK(health["busy"] == false, "not busy once the request is over");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        // Fits only from the lite-disk profile on.
        const bool ok = run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE_DISK ? oom() : ok_result; }, failure);
        CHECK(ok, "the request succeeds after stepping down");
        CHECK(seen.levels == std::vector<Level>({Level::MAX, Level::BALANCED, Level::LITE, Level::LITE_DISK}),
              "the same request was run at each profile in turn");
        CHECK(seen.tiled == std::vector<bool>({false, false, true, true}), "tiling is forced from the lite profile on");
        CHECK(seen.contexts[0] != nullptr && seen.contexts[1] != nullptr, "every attempt had a context");
        CHECK(env.loads.size() == 4, "the model was loaded three more times");
        const json health = keeper.health_json();
        CHECK(counter(health, "step_downs") == 3 && counter(health, "recovered") == 1 && counter(health, "rebuilds") == 3 &&
                  counter(health, "failed") == 0,
              "counters of a request that recovered");
        CHECK(health["status"] == "degraded" && health["profile"] == "lite-disk", "the server stays on the profile that worked");
        CHECK(health["next_promotion_seconds"] == 10, "the climb back is scheduled");
        CHECK(health["last_error"]["code"] == "out_of_memory" && health["last_error"]["message"] == "unet ran out of memory",
              "the last error is kept");

        Seen second;
        CHECK(run_with(keeper, env, second, succeed, failure) && second.levels == std::vector<Level>({Level::LITE_DISK}),
              "the next request runs on the profile that worked, with no rebuild");
        CHECK(env.loads.size() == 4, "no extra load for it");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return oom(); }, failure), "a request that fits nowhere fails");
        CHECK(failure.code == SD_ERROR_OUT_OF_MEMORY && !failure.unavailable && failure.message == "unet ran out of memory",
              "with the out of memory reason, not as unavailable");
        CHECK(seen.levels.size() == 4, "every profile was tried");
        const json health = keeper.health_json();
        CHECK(health["status"] == "degraded" && health["profile"] == "lite-disk", "the server keeps the lightest context");
        CHECK(counter(health, "failed") == 1, "one failed request");
        Seen later;
        CHECK(run_with(keeper, env, later, succeed, failure), "a smaller request is served afterwards");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return failure_of(SD_ERROR_INVALID_ARGUMENT, "bad size"); }, failure), "an invalid request fails");
        CHECK(failure.code == SD_ERROR_INVALID_ARGUMENT && seen.levels.size() == 1 && env.loads.size() == 1,
              "without stepping down");
        const json health = keeper.health_json();
        CHECK(health["last_error"].is_null(), "a mistake of the client is not an error of the server");
        CHECK(counter(health, "failed") == 1, "but it is a failed request");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return oom(); }, failure), "with degradation disabled a request out of memory fails");
        CHECK(failure.code == SD_ERROR_OUT_OF_MEMORY && seen.levels.size() == 1 && env.loads.size() == 1,
              "after one attempt and without loading the model again");
        const json health = keeper.health_json();
        CHECK(health["status"] == "ok" && health["profile"] == "max" && health["degradation"] == false, "the server stays as it was");
        CHECK(health["last_error"]["code"] == "out_of_memory", "and reports why");
    }
    {
        Env env;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const bool ok = run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level == Level::MAX ? backend_failure() : ok_result; }, failure);
        CHECK(ok && seen.levels == std::vector<Level>({Level::MAX, Level::LITE_CPU}),
              "a backend failure goes straight to the cpu profile");
        CHECK(env.loads.back().backend == "cpu", "which loads the model on the cpu");
        CHECK(keeper.backend_choice().backend == "cpu", "and the upscaler follows");
    }
}

static void test_promotion() {
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE ? oom() : ok_result; }, failure);
        const size_t loads = env.loads.size();

        env.advance(5);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 5, "the time left is reported");
        Seen early;
        run_with(keeper, env, early, succeed, failure);
        CHECK(env.loads.size() == loads && early.levels == std::vector<Level>({Level::LITE}), "no climb before the wait is over");

        env.advance(6);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure), "a request after the wait succeeds");
        CHECK(climb.levels == std::vector<Level>({Level::BALANCED}) && env.loads.size() == loads + 1,
              "on the next heavier profile, loaded for it");
        json health = keeper.health_json();
        CHECK(counter(health, "promotions") == 1 && counter(health, "failed_promotions") == 0, "a promotion is counted");
        CHECK(health["status"] == "degraded" && health["profile"] == "balanced", "one step at a time");

        env.advance(11);
        Seen top;
        run_with(keeper, env, top, succeed, failure);
        health = keeper.health_json();
        CHECK(top.levels == std::vector<Level>({Level::MAX}) && health["status"] == "ok" && health["profile"] == "max",
              "back at the profile the server was started with");
        CHECK(health["next_promotion_seconds"].is_null() && counter(health, "promotions") == 2, "and nothing left to climb");
        env.advance(1000);
        Seen quiet;
        run_with(keeper, env, quiet, succeed, failure);
        CHECK(quiet.levels == std::vector<Level>({Level::MAX}), "it never goes above the base profile");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(true, 0), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::BALANCED ? oom() : ok_result; }, failure);
        env.advance(100000);
        Seen later;
        run_with(keeper, env, later, succeed, failure);
        CHECK(later.levels == std::vector<Level>({Level::BALANCED}), "a cooldown of 0 never climbs back");
        CHECK(keeper.health_json()["next_promotion_seconds"].is_null(), "and reports no climb");
    }
    {
        // A climb that fails: the first request on the heavier profile runs out of memory.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const Script needs_balanced = [](const ContextKeeper::ContextView& view) {
            return view.level < Level::BALANCED ? oom() : ok_result;
        };
        run_with(keeper, env, seen, needs_balanced, failure);
        env.advance(11);
        Seen failed_climb;
        CHECK(run_with(keeper, env, failed_climb, needs_balanced, failure), "the request is served even if the climb fails");
        CHECK(failed_climb.levels == std::vector<Level>({Level::MAX, Level::BALANCED}), "after going back down");
        json health = keeper.health_json();
        CHECK(counter(health, "promotions") == 1 && counter(health, "failed_promotions") == 1, "a failed climb is counted");
        CHECK(health["next_promotion_seconds"] == 20, "and the wait doubles");
        env.advance(21);
        Seen again;
        run_with(keeper, env, again, needs_balanced, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 40, "again after another failed climb");

        // A climb that holds puts the wait back to its start.
        env.advance(41);
        Seen holds;
        run_with(keeper, env, holds, succeed, failure);
        CHECK(holds.levels == std::vector<Level>({Level::MAX}), "a later climb that succeeds reaches max");
        Seen drop;
        run_with(keeper, env, drop, needs_balanced, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 10, "the wait starts again from the configured one");
    }
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(true, 2000), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        const Script needs_balanced = [](const ContextKeeper::ContextView& view) {
            return view.level < Level::BALANCED ? oom() : ok_result;
        };
        run_with(keeper, env, seen, needs_balanced, failure);
        env.advance(2001);
        run_with(keeper, env, seen, needs_balanced, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 3600, "the wait stops doubling at an hour");
    }
    {
        // The wait counts from the end of a request that had to step down, however long that took.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [&env](const ContextKeeper::ContextView& view) {
            env.advance(100);
            return view.level < Level::BALANCED ? oom() : ok_result; }, failure);
        CHECK(keeper.health_json()["next_promotion_seconds"] == 10, "a slow request does not use up the wait");
    }
    {
        // The heavier profile cannot be loaded because of the memory: the climb falls back to the one before.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE ? oom() : ok_result; }, failure);
        env.load_result = [](const Load& load) { return load.level() == Level::BALANCED ? oom() : ok_result; };
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::LITE}),
              "a climb that cannot load the profile serves the request one step down");
        const json health = keeper.health_json();
        CHECK(counter(health, "failed_promotions") == 1 && health["next_promotion_seconds"] == 20, "counted as a failed climb");
    }
    {
        // The heavier profile fails for a reason that is not the memory: the profile in use before it comes back.
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE ? oom() : ok_result; }, failure);
        env.load_result = [](const Load& load) { return load.level() == Level::BALANCED ? model_file_failure() : ok_result; };
        env.advance(11);
        Seen climb;
        CHECK(run_with(keeper, env, climb, succeed, failure) && climb.levels == std::vector<Level>({Level::LITE}),
              "the request is served on the previous profile");
    }
}

static void test_unavailable() {
    Env env;
    env.accelerator = false;
    ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
    GenerationFailure failure;
    keeper.start(failure);
    env.load_result = [](const Load& load) { return load.level() == Level::BALANCED ? model_file_failure() : ok_result; };
    Seen seen;
    CHECK(!run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return oom(); }, failure), "a request fails when the model cannot be loaded again");
    CHECK(failure.unavailable && failure.code == SD_ERROR_MODEL_FILE && failure.retry_after_seconds > 0,
          "as unavailable, with the cause and a time to retry");
    json health = keeper.health_json();
    CHECK(health["status"] == "unavailable", "the health says so");
    CHECK(health["last_error"]["code"] == "model_file", "and why");

    const size_t loads = env.loads.size();
    Seen blocked;
    CHECK(!run_with(keeper, env, blocked, succeed, failure) && failure.unavailable && blocked.levels.empty(),
          "requests wait instead of trying again at once");
    CHECK(env.loads.size() == loads && failure.retry_after_seconds >= 1 && failure.retry_after_seconds <= 16,
          "no load, and the time left is reported");

    env.advance(16);
    env.load_result = nullptr;
    Seen restored;
    CHECK(run_with(keeper, env, restored, succeed, failure), "the model is loaded again after the wait");
    CHECK(env.loads.size() == loads + 1 && env.loads.back().level() == Level::BALANCED, "from the profile it had stepped down to");
    CHECK(keeper.health_json()["status"] == "degraded", "and the server serves again");
}

static void test_reporting() {
    {
        Env env;
        env.accelerator = false;
        ContextKeeper keeper(base_params(), Level::MAX, settings(false), env.hooks());
        GenerationFailure failure;
        keeper.start(failure);
        Seen seen;
        run_with(keeper, env, seen, [](const ContextKeeper::ContextView&) { return failure_of(SD_ERROR_OUT_OF_MEMORY, "bad \xff\xfe name"); }, failure);
        std::string text;
        bool threw = false;
        try {
            text = safe_json_dump(keeper.health_json());
        } catch (...) {
            threw = true;
        }
        CHECK(!threw && text.find("out_of_memory") != std::string::npos, "text that is not UTF-8 never breaks the health");
        const json health = keeper.health_json();
        CHECK(health.contains("uptime_seconds") && health.contains("version") && health.contains("counters"), "the health has its fields");
    }

    GenerationFailure f = oom();
    CHECK(failure_http_status(f) == 503 && std::string(failure_code_name(f)) == "out_of_memory", "out of memory is 503");
    f.unavailable = true;
    CHECK(failure_http_status(f) == 503 && std::string(failure_code_name(f)) == "unavailable", "unavailable is 503");
    CHECK(failure_http_status(failure_of(SD_ERROR_INVALID_ARGUMENT, "x")) == 400, "an invalid argument is 400");
    CHECK(failure_http_status(model_file_failure()) == 500 && failure_http_status(backend_failure()) == 500, "model file and backend are 500");
    CHECK(failure_http_status(failure_of(SD_ERROR_UNKNOWN, "x")) == 500 &&
              std::string(failure_code_name(failure_of(SD_ERROR_UNKNOWN, "x"))) == "generation_failed",
          "an unknown failure keeps the code generation_failed");
    CHECK(failure_is_recoverable(SD_ERROR_OUT_OF_MEMORY) && failure_is_recoverable(SD_ERROR_BACKEND) &&
              !failure_is_recoverable(SD_ERROR_MODEL_FILE) && !failure_is_recoverable(SD_ERROR_INVALID_ARGUMENT),
          "only memory and backend failures are worth a lighter profile");

    {
        httplib::Response res;
        GenerationFailure failure   = oom();
        failure.retry_after_seconds = 42;
        send_generation_failure(res, failure);
        const json body = json::parse(res.body);
        CHECK(res.status == 503 && res.get_header_value("Retry-After") == "42", "503 carries Retry-After");
        CHECK(body["code"] == "out_of_memory" && body["retry_after_seconds"] == 42 && body["error"] == "unet ran out of memory",
              "and the code, the wait and the reason in the body");
    }
    {
        httplib::Response res;
        send_generation_failure(res, failure_of(SD_ERROR_INVALID_ARGUMENT, "bad size"));
        const json body = json::parse(res.body);
        CHECK(res.status == 400 && !res.has_header("Retry-After") && body["code"] == "invalid_argument", "400 has no Retry-After");
    }
    {
        httplib::Response res;
        GenerationFailure failure = model_file_failure();
        failure.unavailable       = true;
        send_generation_failure(res, failure);
        const json body = json::parse(res.body);
        CHECK(res.status == 503 && res.has_header("Retry-After") && body["code"] == "unavailable" && body["cause"] == "model_file",
              "unavailable names its cause");
    }
    {
        httplib::Response res;
        send_generation_failure(res, failure_of(SD_ERROR_OUT_OF_MEMORY, "bad \xff\xfe"));
        CHECK(res.status == 503 && !res.body.empty() && json::accept(res.body), "an answer with text that is not UTF-8 is still JSON");
    }
    CHECK(failure_from_exception(std::bad_alloc()).code == SD_ERROR_OUT_OF_MEMORY, "bad_alloc is a lack of memory");
    const GenerationFailure other = failure_from_exception(std::runtime_error("boom"));
    CHECK(other.code == SD_ERROR_UNKNOWN && other.message == "boom", "other exceptions keep their text");
}

static void test_backend_choice() {
    Env env;
    SDContextParams params = base_params();
    params.backend         = "vulkan";
    params.params_backend  = "cpu";
    params.n_threads       = 3;
    ContextKeeper keeper(params, Level::MAX, settings(), env.hooks());
    ContextKeeper::BackendChoice choice = keeper.backend_choice();
    CHECK(choice.backend == "vulkan" && choice.params_backend == "cpu" && choice.n_threads == 3,
          "before a model is loaded, the upscaler uses the options of the server");
    GenerationFailure failure;
    keeper.start(failure);
    Seen seen;
    run_with(keeper, env, seen, [](const ContextKeeper::ContextView& view) { return view.level < Level::LITE_CPU ? oom() : ok_result; }, failure);
    choice = keeper.backend_choice();
    CHECK(choice.backend == "cpu" && choice.n_threads == 3, "after the cpu profile, the upscaler runs on the cpu");
}

static void test_threads() {
    Env env;
    env.accelerator = false;
    ContextKeeper keeper(base_params(), Level::MAX, settings(), env.hooks());
    GenerationFailure start_failure;
    keeper.start(start_failure);

    std::atomic<int> inside{0};
    std::atomic<int> max_inside{0};
    std::atomic<bool> done{false};
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};

    std::thread observer([&] {
        while (!done) {
            const json health = keeper.health_json();
            (void)health["status"];
            (void)keeper.status_name();
            (void)keeper.backend_choice();
            (void)keeper.supports_image_generation();
        }
    });

    std::vector<std::thread> workers;
    for (int t = 0; t < 4; t++) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < 40; i++) {
                GenerationFailure failure;
                const bool ok = keeper.run(
                    [&](const ContextKeeper::ContextView& view) {
                        const int now_inside = ++inside;
                        int seen_max         = max_inside.load();
                        while (now_inside > seen_max && !max_inside.compare_exchange_weak(seen_max, now_inside)) {
                        }
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                        --inside;
                        // Now and then the request does not fit on the heavier profiles.
                        if ((i + t) % 7 == 0 && view.level < Level::LITE) {
                            env.next_failure = oom();
                            return false;
                        }
                        return true;
                    },
                    failure);
                (ok ? succeeded : failed)++;
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    done = true;
    observer.join();

    const json health = keeper.health_json();
    CHECK(max_inside.load() == 1, "never two generations at the same time");
    CHECK(succeeded + failed == 160 && counter(health, "requests") == 160, "every request was counted");
    CHECK(health["busy"] == false, "idle at the end");
    CHECK(counter(health, "step_downs") >= 1 && health["status"] == "degraded", "the steps down of one request serve the next ones");
}

int main() {
    sd_set_log_callback([](enum sd_log_level_t, const char*, void*) {}, nullptr);

    test_ladder();
    test_start();
    test_step_down();
    test_promotion();
    test_unavailable();
    test_reporting();
    test_backend_choice();
    test_threads();

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
