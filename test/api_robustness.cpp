// Robustness test for the public C API (device enumeration, param defaults).
// Build (from the repository root, after building the static library):
//   clang++ -std=c++17 -g -fsanitize=address,undefined -Iinclude test/api_robustness.cpp \
//       build/libstable-diffusion.a build/ggml/src/libggml.a build/ggml/src/libggml-cpu.a \
//       build/ggml/src/libggml-base.a <platform libs> -o api_robustness
// Run: ./api_robustness ; exit code 0 = all checks passed.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <vector>

#include "stable-diffusion.h"

static int failures = 0;
#define CHECK(cond, msg)                  \
    do {                                  \
        if (!(cond)) {                    \
            printf("FAIL: %s\n", msg);    \
            failures++;                   \
        } else {                          \
            printf("ok:   %s\n", msg);    \
        }                                 \
    } while (0)

int main() {
    CHECK(!sd_get_device_info(0, nullptr), "info(null out) rejected");
    sd_device_info_t info;
    CHECK(!sd_get_device_info((size_t)-1, &info), "info(SIZE_MAX index) rejected");
    CHECK(!sd_get_device_info(sd_get_device_count(), &info), "info(index == count) rejected");

    const size_t n = sd_get_device_count();
    CHECK(n >= 1, "at least one device (CPU)");
    bool has_cpu = false, terminated = true;
    for (size_t i = 0; i < n; i++) {
        memset(&info, 0xAB, sizeof(info));
        CHECK(sd_get_device_info(i, &info), "info(i) valid");
        terminated = terminated && memchr(info.name, 0, sizeof(info.name)) &&
                     memchr(info.description, 0, sizeof(info.description));
        has_cpu = has_cpu || info.type == SD_DEVICE_TYPE_CPU;
        CHECK(info.free_memory_bytes <= info.total_memory_bytes || info.total_memory_bytes == 0,
              "free <= total");
    }
    CHECK(terminated, "strings always null-terminated");
    CHECK(has_cpu, "CPU device always present (fallback base)");
    CHECK(strcmp(sd_device_type_name((sd_device_type_t)9999), "unknown") == 0,
          "type_name(invalid) = unknown");

    const size_t need = sd_list_devices(nullptr, 0);
    char tiny[4];
    CHECK(sd_list_devices(tiny, sizeof(tiny)) == need && tiny[3] == '\0',
          "list_devices tiny buffer: no overflow");

    std::vector<std::thread> threads;
    std::vector<int> bad(8, 0);
    for (int t = 0; t < 8; t++) {
        threads.emplace_back([&, t] {
            for (int k = 0; k < 2000; k++) {
                sd_device_info_t x;
                const size_t c = sd_get_device_count();
                for (size_t i = 0; i < c; i++) {
                    if (!sd_get_device_info(i, &x)) {
                        bad[t]++;
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    int total_bad = 0;
    for (int b : bad) {
        total_bad += b;
    }
    CHECK(total_bad == 0, "8-thread enumeration stress without errors");

    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    CHECK(p.memory_guard == 0 && !p.disable_backend_fallback,
          "safe defaults: guard off, fallback on");
    char* s = sd_ctx_params_to_str(&p);
    CHECK(s != nullptr && strstr(s, "memory_guard") && strstr(s, "disable_backend_fallback"),
          "to_str includes new fields");
    free(s);

    printf("\n%s (%d failures)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
