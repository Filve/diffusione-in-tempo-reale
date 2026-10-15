// Test of the multi-GPU placement of --auto-fit (src/core/backend_fit.cpp) with simulated devices, so no GPU
// is needed: which module goes to which GPU for --multi-gpu auto|on|off, when the diffusion model is split,
// and that a plan never promises more memory than a device has.
// The planner functions are internal to the library, so the test includes the source file itself.
// Build (from the repository root, after building the library into build/):
//   c++ -std=c++17 -DGGML_MAX_NAME=160 -Isrc -Isrc/core -Iinclude -Ithirdparty -Iggml/include \
//       test/backend_fit_plan_test.cpp -Lbuild/bin -lstable-diffusion -Wl,-rpath,build/bin -pthread -o backend_fit_plan_test
// Run: ./backend_fit_plan_test ; exit code 0 = all checks passed.
#include "../src/core/backend_fit.cpp"

#include <cstdio>
#include <random>

using namespace sd::backend_fit;

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

static constexpr int64_t GiB = 1024 * MiB;

static Device make_gpu(const std::string& name, const char* registry, double free_gib, bool discrete = true) {
    Device device;
    device.name         = name;
    device.description  = name;
    device.registry     = registry;
    device.discrete     = discrete;
    device.free_bytes   = (int64_t)(free_gib * GiB);
    device.budget_bytes = std::max<int64_t>(device.free_bytes - 512 * MiB, 0);
    return device;
}

// staging is the size of the largest tensor of a module, which the compute device must be able to hold at once.
static std::vector<Component> make_components(double dit_gib, double te_gib, double vae_gib, int64_t te_staging = 256 * MiB, int64_t vae_staging = 128 * MiB) {
    return {
        {ComponentKind::DIT, "DiT", (int64_t)(dit_gib * GiB), 2048 * MiB, 256 * MiB},
        {ComponentKind::CONDITIONER, "Conditioner", (int64_t)(te_gib * GiB), 2048 * MiB, te_staging},
        {ComponentKind::VAE, "VAE", (int64_t)(vae_gib * GiB), 1024 * MiB, vae_staging},
    };
}

struct Outcome {
    bool changed = false;
    std::vector<Runtime> start;
    std::vector<Runtime> runtimes;
    std::vector<std::string> notes;
    Plan before;
    Plan plan;
};

// Starts the way derive_backend_specs does unless told otherwise: every module on the device with the largest budget.
static Outcome run(const std::vector<Component>& components,
                   const std::vector<Device>& devices,
                   MultiGpuMode mode,
                   int64_t ram_gib                         = 0,
                   std::vector<bool> explicit_runtime      = {},
                   std::vector<bool> allow_split           = {},
                   const std::vector<Runtime>& start_given = {}) {
    Outcome out;
    if (start_given.empty()) {
        const size_t main_device = select_main_device(devices);
        out.start                = std::vector<Runtime>(components.size(), Runtime{devices[main_device].name, {main_device}});
    } else {
        out.start = start_given;
    }
    explicit_runtime.resize(components.size(), false);
    allow_split.resize(components.size(), true);
    out.runtimes = out.start;
    out.before   = compute_plan(components, devices, ram_gib * GiB, out.start);
    out.changed  = plan_multi_gpu_runtimes(components, devices, ram_gib * GiB, mode, explicit_runtime, allow_split, out.runtimes, out.notes);
    out.plan     = compute_plan(components, devices, ram_gib * GiB, out.runtimes);
    return out;
}

static std::string names(const Outcome& out) {
    std::string text;
    for (const Runtime& runtime : out.runtimes) {
        text += (text.empty() ? "" : " | ") + runtime.name;
    }
    return text;
}

static bool same_names(const Outcome& out, const char* a, const char* b, const char* c) {
    return out.runtimes.size() == 3 && out.runtimes[0].name == a && out.runtimes[1].name == b && out.runtimes[2].name == c;
}

static bool unchanged(const Outcome& out) {
    if (out.changed || out.runtimes.size() != out.start.size()) {
        return false;
    }
    for (size_t i = 0; i < out.runtimes.size(); ++i) {
        if (out.runtimes[i].name != out.start[i].name || out.runtimes[i].devices != out.start[i].devices) {
            return false;
        }
    }
    return true;
}

static bool resident(const Plan& plan, size_t ci) {
    return plan.decisions[ci].params_location == ParamsLocation::MAIN_GPU;
}

static bool mentions(const Outcome& out, const char* text) {
    for (const std::string& note : out.notes) {
        if (note.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// What the plan puts on each device: the weights of the resident modules, split ones by their shares.
static std::vector<int64_t> weights_per_device(const std::vector<Component>& components, const std::vector<Device>& devices, const Outcome& out) {
    std::vector<int64_t> used(devices.size(), 0);
    for (size_t ci = 0; ci < components.size(); ++ci) {
        if (components[ci].params_bytes == 0 || !resident(out.plan, ci)) {
            continue;
        }
        const auto& shares = out.plan.decisions[ci].shares;
        for (size_t i = 0; i < out.plan.runtimes[ci].devices.size(); ++i) {
            used[out.plan.runtimes[ci].devices[i]] += shares.empty() ? components[ci].params_bytes : shares[i];
        }
    }
    return used;
}

// The largest compute reserve among the resident modules that run on the device.
static int64_t reserve_per_device(const std::vector<Component>& components, size_t device, const Outcome& out) {
    int64_t reserve = 0;
    for (size_t ci = 0; ci < components.size(); ++ci) {
        const auto& used_devices = out.plan.runtimes[ci].devices;
        if (components[ci].params_bytes > 0 && resident(out.plan, ci) &&
            std::find(used_devices.begin(), used_devices.end(), device) != used_devices.end()) {
            reserve = std::max(reserve, components[ci].reserve_bytes);
        }
    }
    return reserve;
}

static void check_random_machines() {
    std::mt19937 rng(12345);
    auto pick = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };

    int plans = 0, moved = 0, splits = 0;
    int over_budget = 0, bad_shares = 0, not_resident = 0, evicted = 0, legacy_changed = 0, explicit_changed = 0, bad_group = 0;
    std::string first_problem;

    for (int round = 0; round < 6000; ++round) {
        std::vector<Device> devices;
        const int gpu_count = 1 + (int)(rng() % 4);
        for (int i = 0; i < gpu_count; ++i) {
            const bool vulkan     = rng() % 4 == 0;
            const bool integrated = rng() % 6 == 0;
            devices.push_back(make_gpu((vulkan ? "Vulkan" : "CUDA") + std::to_string(i), vulkan ? "Vulkan" : "CUDA", pick(1, 48), !integrated));
        }
        const std::vector<Component> components = make_components(pick(0.5, 70), rng() % 5 == 0 ? 0 : pick(0.2, 20), pick(0.1, 3),
                                                                  (int64_t)pick(64, 1500) * MiB, (int64_t)pick(64, 1500) * MiB);
        const int64_t ram_gib                   = (int64_t)pick(0, 64);

        const size_t main_device = select_main_device(devices);
        std::vector<Runtime> start(components.size(), Runtime{devices[main_device].name, {main_device}});
        std::vector<bool> explicit_runtime(components.size(), false);
        std::vector<bool> allow_split(components.size(), true);
        for (size_t ci = 0; ci < components.size(); ++ci) {
            if (rng() % 7 == 0) {
                const size_t di      = rng() % devices.size();
                start[ci]            = Runtime{devices[di].name, {di}};
                explicit_runtime[ci] = true;
            }
            allow_split[ci] = rng() % 5 != 0;
        }

        for (MultiGpuMode mode : {MultiGpuMode::AUTO, MultiGpuMode::ON, MultiGpuMode::OFF}) {
            const Outcome out = run(components, devices, mode, ram_gib, explicit_runtime, allow_split, start);
            ++plans;
            moved += out.changed ? 1 : 0;
            auto problem = [&](const char* what) {
                if (first_problem.empty()) {
                    first_problem = std::string(what) + ": " + names(out) + " (round " + std::to_string(round) + ")";
                }
            };

            const std::vector<int64_t> used = weights_per_device(components, devices, out);
            for (size_t di = 0; di < devices.size(); ++di) {
                if (used[di] + reserve_per_device(components, di, out) > devices[di].budget_bytes) {
                    ++over_budget;
                    problem("over budget");
                }
            }

            bool higher_changed = false;
            for (size_t ci = 0; ci < components.size(); ++ci) {
                const Runtime& runtime = out.runtimes[ci];
                const bool is_split    = runtime.devices.size() > 1 && runtime.estimate_shares;
                if (is_split && resident(out.plan, ci)) {
                    ++splits;
                    int64_t sum = 0;
                    for (int64_t share : out.plan.decisions[ci].shares) {
                        sum += share;
                    }
                    if (out.plan.decisions[ci].shares.size() != runtime.devices.size() || sum != components[ci].params_bytes) {
                        ++bad_shares;
                        problem("shares do not add up");
                    }
                }
                if (is_split) {
                    bool same_group = components[ci].kind != ComponentKind::VAE && allow_split[ci] && !explicit_runtime[ci];
                    for (size_t di : runtime.devices) {
                        same_group = same_group && devices[di].discrete && devices[di].registry == devices[runtime.devices.front()].registry;
                    }
                    if (!same_group) {
                        ++bad_group;
                        problem("split over GPUs that cannot share a module");
                    }
                }
                const bool changed_runtime = runtime.name != out.start[ci].name;
                if (changed_runtime && !resident(out.plan, ci)) {
                    ++not_resident;
                    problem("a moved module is not resident");
                }
                if (explicit_runtime[ci] && changed_runtime) {
                    ++explicit_changed;
                    problem("a module assigned by the user was moved");
                }
                if (resident(out.before, ci) && !resident(out.plan, ci) && (ci == 0 || !higher_changed)) {
                    ++evicted;
                    problem("a resident module was pushed out");
                }
                higher_changed = higher_changed || changed_runtime;
            }

            size_t discrete_best = 0;
            for (const Device& device : devices) {
                discrete_best += device.discrete && device.budget_bytes > 0 ? 1 : 0;
            }
            bool all_resident = true;
            for (size_t ci = 0; ci < components.size(); ++ci) {
                all_resident = all_resident && (components[ci].params_bytes == 0 || resident(out.before, ci));
            }
            if ((mode == MultiGpuMode::OFF || discrete_best < 2 || (mode == MultiGpuMode::AUTO && all_resident)) && !unchanged(out)) {
                ++legacy_changed;
                problem("a case that must keep the old plan changed");
            }
        }
    }

    printf("      %d plans, %d with a move, %d splits\n", plans, moved, splits);
    if (!first_problem.empty()) {
        printf("      first problem: %s\n", first_problem.c_str());
    }
    CHECK(moved > 0 && splits > 0, "random machines: the cases cover moves and splits");
    CHECK(over_budget == 0, "random machines: no device is given more weights than its budget less the compute reserve");
    CHECK(bad_shares == 0, "random machines: the shares of every split add up to the weights");
    CHECK(bad_group == 0, "random machines: only discrete GPUs of one backend, and only a splittable module, are ever split");
    CHECK(not_resident == 0, "random machines: a module that was moved ends up with its weights resident");
    CHECK(evicted == 0, "random machines: a module is never pushed out of its GPU by a change that does not concern a module of higher priority");
    CHECK(explicit_changed == 0, "random machines: a module assigned by the user is never moved");
    CHECK(legacy_changed == 0, "random machines: off, a single GPU, and a plan that already fits change nothing");
}

int main() {
    MultiGpuMode mode = MultiGpuMode::OFF;
    CHECK(parse_multi_gpu_mode("", mode) && mode == MultiGpuMode::AUTO, "an empty --multi-gpu means auto");
    CHECK(parse_multi_gpu_mode("auto", mode) && mode == MultiGpuMode::AUTO && parse_multi_gpu_mode("ON", mode) && mode == MultiGpuMode::ON &&
              parse_multi_gpu_mode(" Off ", mode) && mode == MultiGpuMode::OFF,
          "--multi-gpu accepts auto, on and off in any case");
    CHECK(!parse_multi_gpu_mode("maybe", mode) && !parse_multi_gpu_mode("1", mode), "--multi-gpu refuses anything else");

    {
        std::string spec = "te=cpu,vae=CUDA0";
        set_assignment(spec, "vae", "CUDA1");
        set_assignment(spec, "diffusion", "CUDA0&CUDA1");
        CHECK(spec == "te=cpu,vae=CUDA1,diffusion=CUDA0&CUDA1", "set_assignment replaces a module and appends a missing one");
    }

    {
        std::vector<int64_t> shares;
        CHECK(estimate_split_shares(10 * GiB, {8 * GiB, 8 * GiB}, shares) && shares[0] == 8 * GiB && shares[1] == 2 * GiB,
              "a split fills the first device before the next");
        CHECK(!estimate_split_shares(16 * GiB, {8 * GiB, 8 * GiB}, shares), "a split needs room for the blocks that do not cut evenly");
        CHECK(estimate_split_shares(14 * GiB, {8 * GiB, 8 * GiB}, shares) && shares[0] + shares[1] == 14 * GiB,
              "the shares add up to the size of the weights");
    }

    const std::vector<Device> one_gpu  = {make_gpu("CUDA0", "CUDA", 12)};
    const std::vector<Device> two_gpus = {make_gpu("CUDA0", "CUDA", 12), make_gpu("CUDA1", "CUDA", 12)};

    {
        const std::vector<Runtime> on_cuda0(3, Runtime{"CUDA0", {0}});
        const Plan fits = compute_plan(make_components(6, 2, 0.3), one_gpu, 0, on_cuda0);
        CHECK(resident(fits, 0) && resident(fits, 1) && resident(fits, 2), "unchanged planner: weights that fit stay on the GPU");
        const Plan in_ram = compute_plan(make_components(16, 2, 0.3), one_gpu, 64 * GiB, on_cuda0);
        CHECK(in_ram.decisions[0].params_location == ParamsLocation::CPU && in_ram.decisions[0].shares.empty(),
              "unchanged planner: weights that do not fit wait in RAM");
        const Plan on_disk = compute_plan(make_components(16, 2, 0.3), one_gpu, 0, on_cuda0);
        CHECK(on_disk.decisions[0].params_location == ParamsLocation::DISK, "unchanged planner: without RAM they stay on disk");
        const std::vector<Runtime> listed(3, Runtime{"CUDA0&CUDA1", {0, 1}});
        const Plan both = compute_plan(make_components(9, 0, 0), two_gpus, 0, listed);
        CHECK(resident(both, 0) && both.decisions[0].shares.empty(), "unchanged planner: a device list given by the user counts the full size on each device");
        const Plan too_big = compute_plan(make_components(12, 0, 0), two_gpus, 0, listed);
        CHECK(!resident(too_big, 0), "and so a list is refused when the weights do not fit on every device, even if they would fit split");
    }

    for (MultiGpuMode m : {MultiGpuMode::AUTO, MultiGpuMode::ON, MultiGpuMode::OFF}) {
        const Outcome out = run(make_components(16, 2, 0.3), one_gpu, m);
        CHECK(unchanged(out) && out.notes.empty(), "one GPU: the plan is the same as before in every mode");
    }
    {
        const Outcome out = run(make_components(6, 2, 0.3), {}, MultiGpuMode::AUTO, 0, {}, {}, std::vector<Runtime>(3, Runtime{"CPU", {}}));
        CHECK(unchanged(out), "no GPU: nothing to place");
    }

    {
        const Outcome out = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::AUTO);
        CHECK(unchanged(out) && same_names(out, "CUDA0", "CUDA0", "CUDA0") && resident(out.plan, 0) && resident(out.plan, 1) && resident(out.plan, 2),
              "two GPUs, everything fits on one: auto leaves it there");
        CHECK(mentions(out, "2 GPUs found (CUDA0, CUDA1)") && mentions(out, "everything fits on CUDA0, so CUDA1 will not be used"),
              "and says why the second GPU stays idle");
    }
    {
        const Outcome out = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::OFF);
        CHECK(unchanged(out) && out.notes.empty(), "--multi-gpu off never touches the plan");
    }
    {
        const Outcome out = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::ON);
        CHECK(out.changed && same_names(out, "CUDA0", "CUDA1", "CUDA1") && resident(out.plan, 0) && resident(out.plan, 1) && resident(out.plan, 2),
              "--multi-gpu on moves the text encoder and the VAE to the second GPU and keeps the diffusion model on the first");
        CHECK(mentions(out, "te runs on CUDA1") && mentions(out, "vae runs on CUDA1"), "and reports both moves");
    }

    {
        const std::vector<Component> components = make_components(16, 2, 0.3);
        const Outcome out                       = run(components, two_gpus, MultiGpuMode::AUTO);
        CHECK(!resident(out.before, 0), "the diffusion model alone does not fit on one 12 GiB GPU");
        CHECK(out.changed && out.runtimes[0].name == "CUDA0&CUDA1" && out.runtimes[0].devices.size() == 2 && out.runtimes[0].estimate_shares,
              "16 GiB of diffusion weights are split across both GPUs");
        CHECK(resident(out.plan, 0) && out.plan.decisions[0].shares.size() == 2 &&
                  out.plan.decisions[0].shares[0] + out.plan.decisions[0].shares[1] == 16 * GiB &&
                  out.plan.decisions[0].shares[0] > out.plan.decisions[0].shares[1],
              "the shares add up to the weights, the first GPU taking the most");
        CHECK(same_names(out, "CUDA0&CUDA1", "CUDA1", "CUDA1") && resident(out.plan, 1) && resident(out.plan, 2),
              "the text encoder and the VAE use what is left on the second GPU");
        CHECK(mentions(out, "diffusion is split across CUDA0&CUDA1"), "and the split is reported");
        const std::vector<int64_t> used = weights_per_device(components, two_gpus, out);
        CHECK(used[0] + 2 * GiB <= two_gpus[0].budget_bytes && used[1] + 2 * GiB <= two_gpus[1].budget_bytes,
              "no GPU is given more than its budget less the compute reserve");
    }
    {
        const Outcome out = run(make_components(16, 2, 0.3), two_gpus, MultiGpuMode::ON);
        CHECK(same_names(out, "CUDA0&CUDA1", "CUDA1", "CUDA1"), "on and auto agree when the diffusion model needs both GPUs");
    }
    {
        const Outcome out = run(make_components(16, 2, 0.3), two_gpus, MultiGpuMode::OFF, 64);
        CHECK(unchanged(out) && out.plan.decisions[0].params_location == ParamsLocation::CPU,
              "off keeps the old plan: the weights wait in RAM and stream to the first GPU");
    }
    {
        const Outcome out = run(make_components(16, 2, 0.3), two_gpus, MultiGpuMode::AUTO, 64, {}, {false, true, true});
        CHECK(out.runtimes[0].name == "CUDA0" && !resident(out.plan, 0), "a module whose --split-mode is row is not split by auto-fit");
    }
    {
        const Outcome out = run(make_components(40, 2, 0.3), two_gpus, MultiGpuMode::AUTO, 64);
        CHECK(out.runtimes[0].name == "CUDA0" && !resident(out.plan, 0), "weights too big for both GPUs together are not split");
    }
    {
        const std::vector<Device> small_second = {make_gpu("CUDA0", "CUDA", 12), make_gpu("CUDA1", "CUDA", 3)};
        const Outcome out                      = run(make_components(16, 2, 0.3), small_second, MultiGpuMode::AUTO, 64);
        CHECK(out.runtimes[0].name == "CUDA0" && !resident(out.plan, 0), "a second GPU too small to add anything does not make a split");
    }
    {
        const std::vector<Device> three_gpus = {make_gpu("CUDA0", "CUDA", 12), make_gpu("CUDA1", "CUDA", 12), make_gpu("CUDA2", "CUDA", 12)};
        const Outcome out                    = run(make_components(24, 2, 0.3), three_gpus, MultiGpuMode::AUTO);
        CHECK(out.runtimes[0].name == "CUDA0&CUDA1&CUDA2" && resident(out.plan, 0) && out.plan.decisions[0].shares.size() == 3,
              "three GPUs: the diffusion model is split across all three when two are not enough");
    }
    {
        const std::vector<Device> uneven = {make_gpu("CUDA0", "CUDA", 8), make_gpu("CUDA1", "CUDA", 24)};
        const Outcome out                = run(make_components(16, 2, 0.3), uneven, MultiGpuMode::AUTO);
        CHECK(unchanged(out) && same_names(out, "CUDA1", "CUDA1", "CUDA1"), "uneven GPUs: everything stays on the larger one when it fits");
        const Outcome on = run(make_components(16, 2, 0.3), uneven, MultiGpuMode::ON);
        CHECK(same_names(on, "CUDA1", "CUDA0", "CUDA0"), "and --multi-gpu on moves the text encoder and the VAE to the smaller one");
    }
    {
        const std::vector<Device> mixed = {make_gpu("CUDA0", "CUDA", 12), make_gpu("Vulkan0", "Vulkan", 12)};
        const Outcome out               = run(make_components(16, 2, 0.3), mixed, MultiGpuMode::AUTO, 64);
        CHECK(unchanged(out) && mentions(out, "different backends"), "GPUs of different backends are not combined, and the user is told");
    }
    {
        const std::vector<Device> with_igpu = {make_gpu("CUDA0", "CUDA", 8), make_gpu("CUDA1", "CUDA", 8), make_gpu("Vulkan0", "Vulkan", 30, false)};
        const Outcome out                   = run(make_components(6, 2, 0.3), with_igpu, MultiGpuMode::AUTO);
        CHECK(unchanged(out) && same_names(out, "Vulkan0", "Vulkan0", "Vulkan0"), "an integrated GPU keeps the modules it hosts when they fit");
        CHECK(mentions(out, "everything fits on Vulkan0, so CUDA0, CUDA1 will not be used"), "and the note names the device that hosts them");
        const Outcome big = run(make_components(40, 2, 0.3), with_igpu, MultiGpuMode::AUTO, 64);
        CHECK(big.runtimes[0].name == "Vulkan0" && !resident(big.plan, 0), "and is never part of a split");
    }
    {
        const std::vector<Runtime> start = {Runtime{"CUDA1", {1}}, Runtime{"CUDA0", {0}}, Runtime{"CUDA0", {0}}};
        const Outcome out                = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::ON, 0, {true, false, false}, {}, start);
        CHECK(same_names(out, "CUDA1", "CUDA0", "CUDA0") && unchanged(out),
              "--multi-gpu on keeps the text encoder and the VAE away from a diffusion model the user put on another GPU");
        const Outcome same_gpu = run(make_components(6, 2, 0.3), two_gpus, MultiGpuMode::ON, 0, {true, false, false},
                                     {}, {Runtime{"CUDA0", {0}}, Runtime{"CUDA0", {0}}, Runtime{"CUDA0", {0}}});
        CHECK(same_names(same_gpu, "CUDA0", "CUDA1", "CUDA1"), "and moves them off the GPU the user gave to the diffusion model");
    }
    {
        const Outcome out = run(make_components(16, 2, 0.3), two_gpus, MultiGpuMode::ON, 0, {true, true, true});
        CHECK(unchanged(out), "when every module is assigned by the user there is nothing to plan");
    }
    {
        const Outcome out = run(make_components(16, 0, 0.3), two_gpus, MultiGpuMode::AUTO);
        CHECK(out.runtimes[0].name == "CUDA0&CUDA1" && resident(out.plan, 0) && resident(out.plan, 2),
              "a model without a text encoder still splits and places the VAE");
    }
    {
        // The split just fits while the text encoder and the VAE are on the first GPU. Neither can follow onto the
        // second one without pushing the diffusion model off the GPUs, so they stay where they are.
        const std::vector<Component> components = {
            {ComponentKind::DIT, "DiT", 17150 * MiB, 2048 * MiB, 256 * MiB},
            {ComponentKind::CONDITIONER, "Conditioner", 2048 * MiB, 2048 * MiB, 256 * MiB},
            {ComponentKind::VAE, "VAE", 300 * MiB, 1024 * MiB, 1200 * MiB},
        };
        const Outcome out = run(components, two_gpus, MultiGpuMode::AUTO, 64);
        CHECK(same_names(out, "CUDA0&CUDA1", "CUDA0", "CUDA0") && resident(out.plan, 0) && !resident(out.plan, 1) && !resident(out.plan, 2),
              "the split diffusion model keeps its place when the text encoder and the VAE cannot move next to it");
    }

    check_random_machines();

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
