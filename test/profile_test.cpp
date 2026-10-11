// Test of the resource profiles (examples/common/profile.*): names, the order of the steps, the choice
// made from the size of the model and the free memory, and which options a profile may set.
// Build (from the repository root, after building the library into build/):
//   cc -c thirdparty/zip.c -o zip.o
//   c++ -std=c++17 -Iexamples -Iinclude -Ithirdparty -Isrc test/profile_test.cpp examples/common/common.cpp \
//       examples/common/log.cpp examples/common/media_io.cpp examples/common/profile.cpp zip.o \
//       -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o profile_test
// Run: ./profile_test ; exit code 0 = all checks passed.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/profile.h"

using sd_profile::Level;

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

static constexpr int64_t GiB = 1024ll * 1024ll * 1024ll;

static sd_profile::Resources resources(double model_gib, double ram_gib, double gpu_gib) {
    sd_profile::Resources r;
    r.model_bytes   = static_cast<int64_t>(model_gib * GiB);
    r.ram_available = ram_gib < 0 ? -1 : static_cast<int64_t>(ram_gib * GiB);
    r.gpu_free      = static_cast<int64_t>(gpu_gib * GiB);
    return r;
}

static void check_choice(const char* label, const sd_profile::Resources& r, Level expected) {
    std::string reason;
    const Level got = sd_profile::choose(r, &reason);
    char text[160];
    snprintf(text, sizeof(text), "choose: %s -> %s", label, sd_profile::level_name(expected));
    CHECK(got == expected && !reason.empty(), text);
}

// Parses a command line into fresh parameters, the way sd-cli and sd-server do.
struct Parsed {
    SDContextParams ctx;
    SDGenerationParams gen;
    bool ok = false;

    explicit Parsed(std::vector<const char*> args) {
        args.insert(args.begin(), "test");
        std::vector<ArgOptions> options = {ctx.get_options(), gen.get_options()};
        ok                              = parse_options(static_cast<int>(args.size()), args.data(), options);
    }
};

int main() {
    CHECK(strcmp(sd_profile::level_name(Level::MAX), "max") == 0 &&
              strcmp(sd_profile::level_name(Level::BALANCED), "balanced") == 0 &&
              strcmp(sd_profile::level_name(Level::LITE), "lite") == 0 &&
              strcmp(sd_profile::level_name(Level::LITE_DISK), "lite-disk") == 0 &&
              strcmp(sd_profile::level_name(Level::LITE_CPU), "lite-cpu") == 0,
          "level names");

    Level level    = Level::MAX;
    bool automatic = true;
    CHECK(sd_profile::parse("lite", level, automatic) && level == Level::LITE && !automatic, "parse lite");
    CHECK(sd_profile::parse("balanced", level, automatic) && level == Level::BALANCED && !automatic, "parse balanced");
    CHECK(sd_profile::parse("max", level, automatic) && level == Level::MAX && !automatic, "parse max");
    CHECK(sd_profile::parse("auto", level, automatic) && automatic, "parse auto");
    CHECK(!sd_profile::parse("lite-disk", level, automatic) && !sd_profile::parse("", level, automatic) &&
              !sd_profile::parse("LITE", level, automatic) && !sd_profile::parse("fast", level, automatic),
          "the steps below lite and unknown names are not accepted");

    CHECK(sd_profile::lighter(Level::MAX) == Level::BALANCED && sd_profile::lighter(Level::BALANCED) == Level::LITE &&
              sd_profile::lighter(Level::LITE) == Level::LITE_DISK && sd_profile::lighter(Level::LITE_DISK) == Level::LITE_CPU &&
              sd_profile::lighter(Level::LITE_CPU) == Level::LITE_CPU,
          "lighter walks down the steps and stops at the last one");

    check_choice("unknown model size", resources(0, 8, 0), Level::MAX);
    check_choice("unknown memory", resources(2, -1, 0), Level::MAX);
    check_choice("SD 1.5 f16, 8 GiB RAM", resources(2, 8, 0), Level::MAX);
    check_choice("SD 1.5 f16, 3.6 GiB RAM", resources(2, 3.6, 0), Level::BALANCED);
    check_choice("SD 1.5 f16, 3 GiB RAM", resources(2, 3, 0), Level::LITE);
    check_choice("SDXL f16, 16 GiB RAM", resources(6.9, 16, 0), Level::MAX);
    check_choice("SDXL f16, 10 GiB RAM", resources(6.9, 10, 0), Level::BALANCED);
    check_choice("SDXL f16, 5 GiB RAM, 5.5 GiB VRAM", resources(6.9, 5, 5.5), Level::LITE);
    check_choice("SDXL Q8, 5 GiB RAM, 6 GiB VRAM", resources(3.7, 5, 6), Level::BALANCED);
    check_choice("SDXL f16, 8 GiB RAM, 24 GiB VRAM", resources(6.9, 8, 24), Level::MAX);
    check_choice("fits the GPU but RAM is almost gone", resources(2, 0.5, 12), Level::BALANCED);

    {
        Parsed p({});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ok && p.ctx.memory_guard == 85 && p.ctx.clip_on_cpu && !p.ctx.batched_cfg && p.gen.vae_tiling_params.enabled,
              "lite sets the guard, the text encoder on the CPU, unbatched CFG and VAE tiling");
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 85 && p.ctx.params_backend.empty() && p.ctx.backend.empty(),
              "applying a level twice changes nothing more, and lite leaves the weights placement to auto-fit");
    }
    {
        Parsed p({});
        sd_profile::apply(Level::BALANCED, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 90 && !p.ctx.clip_on_cpu && p.ctx.batched_cfg && !p.gen.vae_tiling_params.enabled,
              "balanced only turns the memory guard on");
    }
    {
        Parsed p({});
        sd_profile::apply(Level::MAX, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 0 && !p.ctx.clip_on_cpu && p.ctx.batched_cfg, "max changes nothing");
    }
    {
        Parsed p({"--memory-guard", "70", "--batched-cfg", "on"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ok && p.ctx.memory_guard == 70 && p.ctx.batched_cfg, "options given on the command line win");
        sd_profile::apply(Level::LITE, p.ctx, p.gen, false);
        CHECK(p.ctx.memory_guard == 70 && !p.ctx.batched_cfg, "a recovery step overrides them, but never raises the guard");
    }
    {
        Parsed p({"--memory-guard", "0"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(p.ctx.memory_guard == 0, "an explicit --memory-guard 0 is respected");
        sd_profile::apply(Level::LITE, p.ctx, p.gen, false);
        CHECK(p.ctx.memory_guard == 85, "a recovery step turns the guard on anyway");
    }
    {
        Parsed p({"--memory-guard", "95"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, false);
        CHECK(p.ctx.memory_guard == 85, "a recovery step lowers a higher guard");
    }
    {
        Parsed p({"--backend", "te=cuda0"});
        sd_profile::apply(Level::LITE, p.ctx, p.gen, true);
        CHECK(!p.ctx.clip_on_cpu, "an explicit text encoder placement is not overridden by lite");
    }
    {
        Parsed p({"--params-backend", "cpu", "--offload-to-cpu", "--eager-load"});
        sd_profile::apply(Level::LITE_DISK, p.ctx, p.gen, true);
        CHECK(p.ctx.params_backend == "cpu" && p.ctx.offload_params_to_cpu && p.ctx.eager_load,
              "an explicit --params-backend is kept at startup");
        sd_profile::apply(Level::LITE_DISK, p.ctx, p.gen, false);
        CHECK(p.ctx.params_backend == "disk" && !p.ctx.offload_params_to_cpu, "lite-disk streams the weights from disk in a recovery");
        CHECK(!p.ctx.eager_load, "and does not read all of them into memory first");
    }
    {
        Parsed p({"--backend", "cuda0"});
        sd_profile::apply(Level::LITE_CPU, p.ctx, p.gen, true);
        CHECK(p.ctx.backend == "cuda0", "an explicit --backend is kept at startup");
        sd_profile::apply(Level::LITE_CPU, p.ctx, p.gen, false);
        CHECK(p.ctx.backend == "cpu" && !p.ctx.clip_on_cpu && p.ctx.params_backend == "disk",
              "lite-cpu computes everything on the CPU in a recovery");
    }
    {
        Parsed p({});
        p.ctx.profile = "bogus";
        Level applied = Level::LITE;
        CHECK(!sd_profile::apply_from_options(p.ctx, p.gen, applied), "an invalid profile name is refused");
        p.ctx.profile.clear();
        CHECK(sd_profile::apply_from_options(p.ctx, p.gen, applied) && applied == Level::MAX && p.ctx.memory_guard == 0,
              "no --profile leaves the options alone");
        p.ctx.profile = "lite";
        CHECK(sd_profile::apply_from_options(p.ctx, p.gen, applied) && applied == Level::LITE && p.ctx.memory_guard == 85,
              "--profile lite is applied");
    }

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
