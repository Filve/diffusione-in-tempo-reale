// Test for the pool of open model file handles (src/model_io/file_handle_pool.*).
// Build and run from the repository root; only the headers of this repository are needed:
//   c++ -std=c++17 -g -fsanitize=address,undefined -Isrc -Iinclude -Iggml/include -pthread test/file_handle_pool_test.cpp src/model_io/file_handle_pool.cpp -o file_handle_pool_test
//   ./file_handle_pool_test
// Use -fsanitize=thread instead of address,undefined to check the multi-thread cases.
// Exit code 0 = all checks passed.
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/util.h"
#include "model_io/file_handle_pool.h"

namespace fs = std::filesystem;

static std::atomic<int> warnings{0};
static std::atomic<int> errors{0};

// The pool logs through the library's logger; the test supplies its own.
void log_printf(sd_log_level_t level, const char*, int, const char*, ...) {
    if (level == SD_LOG_WARN) {
        ++warnings;
    } else if (level == SD_LOG_ERROR) {
        ++errors;
    }
}

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

static constexpr size_t kFileSize = 1 << 20;

static std::string content(uint8_t seed) {
    std::string data(kFileSize, '\0');
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<char>((i * 31 + seed) & 0xff);
    }
    return data;
}

static void write_file(const fs::path& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

// Number of descriptors of this process that refer to path; -1 where /proc is unavailable.
static int open_descriptors(const fs::path& path) {
    std::error_code error;
    if (!fs::exists("/proc/self/fd", error)) {
        return -1;
    }
    const fs::path target = fs::canonical(path, error);
    int count             = 0;
    for (const auto& entry : fs::directory_iterator("/proc/self/fd", error)) {
        std::error_code link_error;
        if (fs::read_symlink(entry.path(), link_error) == target) {
            ++count;
        }
    }
    return count;
}

// True when the stream, positioned at offset, yields the bytes the file holds there.
static bool reads_expected(std::ifstream& stream, const std::string& expected, size_t offset, size_t length) {
    std::string buffer(length, '\0');
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(&buffer[0], static_cast<std::streamsize>(length));
    return stream && buffer.compare(0, length, expected, offset, length) == 0;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("file_handle_pool_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const std::string file_a = (dir / "a.bin").string();
    const std::string file_b = (dir / "b.bin").string();
    const std::string data_a = content(7);
    const std::string data_b = content(99);
    write_file(file_a, data_a);
    write_file(file_b, data_b);
    const bool have_proc = open_descriptors(file_a) >= 0;

    {
        sd::FileHandlePool pool;
        {
            auto lease = pool.acquire(file_a);
            CHECK(static_cast<bool>(lease), "opens an existing file");
            CHECK(reads_expected(lease.stream(), data_a, 1000, 64), "reads the right bytes");
        }
        CHECK(pool.opened_total() == 1 && pool.open_count() == 1, "the stream stays open after the lease ends");
        {
            auto lease = pool.acquire(file_a);
            CHECK(reads_expected(lease.stream(), data_a, 500000, 128), "reused stream reads the right bytes");
        }
        CHECK(pool.opened_total() == 1 && pool.reused_total() == 1, "second acquisition reuses the stream");
        if (have_proc) {
            CHECK(open_descriptors(file_a) == 1, "one descriptor in use after two acquisitions");
        }

        // A read past the end leaves the stream failed: it must not be handed out again.
        {
            auto lease = pool.acquire(file_a);
            lease.stream().seekg(static_cast<std::streamoff>(kFileSize - 4));
            char buffer[64];
            lease.stream().read(buffer, sizeof(buffer));
            CHECK(!lease.stream(), "reading past the end fails");
        }
        CHECK(pool.open_count() == 0, "a failed stream is closed, not pooled");
        {
            auto lease = pool.acquire(file_a);
            CHECK(reads_expected(lease.stream(), data_a, 0, 16), "next acquisition opens a healthy stream");
        }

        // Reaching exactly the end of the file is not an error.
        {
            auto lease = pool.acquire(file_a);
            CHECK(reads_expected(lease.stream(), data_a, kFileSize - 16, 16), "reads the last bytes");
        }
        CHECK(pool.open_count() == 1, "a stream that read up to the end is still pooled");
        {
            auto lease = pool.acquire(file_a);
            CHECK(reads_expected(lease.stream(), data_a, 4096, 16), "pooled stream works after reading the end");
        }

        {
            auto lease = pool.acquire(file_a);
            lease.discard();
            CHECK(!lease, "a discarded lease is empty");
        }
        CHECK(pool.open_count() == 0, "discard closes the stream");
        if (have_proc) {
            CHECK(open_descriptors(file_a) == 0, "no descriptor left after discard");
        }

        {
            auto first = pool.acquire(file_a);
        }
        const size_t opened_before = pool.opened_total();
        {
            auto lease = pool.acquire(file_a, true);
            CHECK(static_cast<bool>(lease), "fresh acquisition succeeds");
        }
        CHECK(pool.opened_total() == opened_before + 1 && pool.open_count() == 1, "fresh acquisition replaces the idle stream");

        // The file changes while its handle sits idle.
        const auto old_time = fs::last_write_time(file_a);
        write_file(file_a, data_b);
        fs::last_write_time(file_a, old_time + std::chrono::seconds(5));
        {
            auto lease = pool.acquire(file_a);
            CHECK(reads_expected(lease.stream(), data_b, 2000, 64), "a changed file is read from a new stream");
        }
        CHECK(pool.open_count() == 1, "the stale stream was closed");
        if (have_proc) {
            CHECK(open_descriptors(file_a) == 1, "only the new descriptor remains");
        }
        write_file(file_a, data_a);
        fs::last_write_time(file_a, old_time);
        pool.close_idle();

#ifndef _WIN32
        // A file replaced by rename: the idle stream still points at the old content.
        {
            auto lease = pool.acquire(file_a);
            CHECK(reads_expected(lease.stream(), data_a, 10, 32), "reads before the replacement");
        }
        write_file(dir / "replacement.bin", data_b);
        fs::rename(dir / "replacement.bin", file_a);
        {
            auto lease = pool.acquire(file_a);
            CHECK(reads_expected(lease.stream(), data_b, 10, 32), "a replaced file is read from a new stream");
        }
        write_file(file_a, data_a);
        pool.close_idle();
#endif
    }

    {
        // Missing files fail fast: ENOENT is not retried.
        sd::FileHandlePool pool;
        const int warnings_before = warnings;
        const int errors_before   = errors;
        const auto start          = std::chrono::steady_clock::now();
        auto lease                = pool.acquire((dir / "missing.bin").string());
        const auto elapsed        = std::chrono::steady_clock::now() - start;
        CHECK(!lease, "a missing file gives an empty lease");
        CHECK(elapsed < std::chrono::milliseconds(150), "a missing file is not retried");
        CHECK(warnings - warnings_before == 1 && errors - errors_before == 1, "the failure is logged once");
        CHECK(pool.open_count() == 0, "nothing is counted for a failed open");
    }

    {
        sd::FileHandlePool pool;
        pool.set_max_idle(4);
        {
            std::vector<sd::FileHandlePool::Lease> leases;
            for (int i = 0; i < 6; ++i) {
                leases.push_back(pool.acquire(file_a));
            }
            CHECK(pool.open_count() == 6 && pool.opened_total() == 6, "concurrent leases get separate streams");
            for (auto& lease : leases) {
                CHECK(reads_expected(lease.stream(), data_a, 777, 32), "each stream reads independently");
            }
        }
        CHECK(pool.open_count() == 4, "no more than the limit is kept idle");
        pool.set_max_idle(2);
        CHECK(pool.open_count() == 2, "lowering the limit closes the extra streams");
        if (have_proc) {
            CHECK(open_descriptors(file_a) == 2, "descriptors follow the limit");
        }
        {
            auto on_b = pool.acquire(file_b);
        }
        CHECK(pool.open_count() == 3, "limits apply per file");
        pool.close_idle(file_a);
        CHECK(pool.open_count() == 1, "closing the idle streams of one file leaves the others");
        {
            auto lease = pool.acquire(file_b);
            pool.close_idle();
            CHECK(pool.open_count() == 1, "a leased stream survives close_idle");
        }
        CHECK(pool.open_count() == 0, "a stream leased during close_idle is closed when returned");

        sd::FileHandlePool::Lease a = pool.acquire(file_a);
        sd::FileHandlePool::Lease b = std::move(a);
        CHECK(!a && b, "moving a lease empties the source");
        b = pool.acquire(file_b);
        CHECK(pool.open_count() == 2, "assigning over a lease returns its stream to the pool");
        sd::FileHandlePool::Lease c;
        c = std::move(b);
        CHECK(!b && c && reads_expected(c.stream(), data_b, 5, 8), "move assignment transfers the stream");
    }

    {
        sd::FileHandlePool pool;
        sd::FileHandlePool copy = pool;
        {
            auto lease = pool.acquire(file_a);
        }
        {
            auto lease = copy.acquire(file_a);
        }
        CHECK(pool.opened_total() == 1 && copy.reused_total() == 1, "copies of a pool share their streams");
        sd::FileHandlePool moved = std::move(pool);
        {
            auto lease = pool.acquire(file_a);
            CHECK(static_cast<bool>(lease), "a pool that was moved from is still usable");
        }
        (void)moved;
    }

    {
        sd::FileHandlePool::Lease survivor;
        {
            sd::FileHandlePool pool;
            survivor = pool.acquire(file_a);
        }
        CHECK(reads_expected(survivor.stream(), data_a, 123, 45), "a lease can outlive its pool object");
    }

    {
        // Many threads sharing one pool, some of them discarding, refreshing or closing.
        sd::FileHandlePool pool;
        pool.set_max_idle(8);
        std::atomic<int> bad{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&, t]() {
                std::mt19937 random(1234 + t);
                for (int i = 0; i < 300; ++i) {
                    const int roll   = static_cast<int>(random() % 100);
                    const bool other = random() % 2 == 0;
                    auto lease       = pool.acquire(other ? file_b : file_a, roll < 5);
                    if (!lease) {
                        ++bad;
                        continue;
                    }
                    const size_t offset = random() % (kFileSize - 256);
                    if (!reads_expected(lease.stream(), other ? data_b : data_a, offset, 256)) {
                        ++bad;
                    }
                    if (roll >= 90) {
                        lease.discard();
                    } else if (roll == 50) {
                        pool.close_idle();
                    }
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        CHECK(bad == 0, "concurrent use reads correct data every time");
        if (have_proc) {
            CHECK(static_cast<int>(pool.open_count()) == open_descriptors(file_a) + open_descriptors(file_b),
                  "the pool's count matches the open descriptors");
        }
        pool.close_idle();
        CHECK(pool.open_count() == 0, "everything is closed in the end");
        if (have_proc) {
            CHECK(open_descriptors(file_a) + open_descriptors(file_b) == 0, "no descriptor leaks");
        }
    }

    std::error_code ignored;
    fs::remove_all(dir, ignored);
    printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
