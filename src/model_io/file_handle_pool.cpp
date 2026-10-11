#include "model_io/file_handle_pool.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "core/util.h"

namespace sd {

    bool open_with_retry(const std::string& path, const std::function<bool()>& try_open) {
        constexpr int kAttempts = 5;
        for (int attempt = 1; attempt <= kAttempts; ++attempt) {
            errno = 0;
            if (try_open()) {
                return true;
            }
            const int saved_errno = errno;
#ifdef _WIN32
            unsigned long win_err = 0;
            _get_doserrno(&win_err);
#endif
            const std::string reason = std::generic_category().message(saved_errno);
#ifdef _WIN32
            LOG_WARN("open '%s' failed (attempt %d/%d): errno=%d (%s), doserrno=%lu",
                     path.c_str(), attempt, kAttempts, saved_errno, reason.c_str(), win_err);
#else
            LOG_WARN("open '%s' failed (attempt %d/%d): errno=%d (%s)",
                     path.c_str(), attempt, kAttempts, saved_errno, reason.c_str());
#endif
            if (saved_errno == ENOENT || attempt == kAttempts) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200 * attempt));
        }
        return false;
    }

    struct FileHandlePool::State {
        struct Idle {
            std::unique_ptr<std::ifstream> stream;
            Stamp stamp;
        };

        std::mutex mutex;
        std::map<std::string, std::vector<Idle>> idle;
        size_t max_idle     = 8;
        uint64_t generation = 0;
        size_t open_streams = 0;
        size_t opened       = 0;
        size_t reused       = 0;
    };

    FileHandlePool::FileHandlePool()
        : state_(std::make_shared<State>()) {
    }

    FileHandlePool::Stamp FileHandlePool::read_stamp(const std::string& path) {
        Stamp stamp;
        std::error_code error;
        const auto file_path = std::filesystem::u8path(path);
        stamp.time           = std::filesystem::last_write_time(file_path, error);
        if (error) {
            return stamp;
        }
        stamp.size  = std::filesystem::file_size(file_path, error);
        stamp.valid = !error;
        return stamp;
    }

    FileHandlePool::Lease FileHandlePool::acquire(const std::string& path, bool fresh) {
        const Stamp stamp = read_stamp(path);

        // Streams are closed after the lock is released; closing can block on slow storage.
        std::vector<std::unique_ptr<std::ifstream>> to_close;
        std::unique_ptr<std::ifstream> stream;
        uint64_t generation = 0;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            generation = state_->generation;
            auto found = state_->idle.find(path);
            if (found != state_->idle.end()) {
                auto& list = found->second;
                // A handle opened before the file changed would keep serving the old content.
                // When the file cannot be inspected right now (this can happen under resource
                // pressure) there is nothing to compare, so an idle handle is kept and reused.
                auto stale = std::stable_partition(list.begin(), list.end(), [&](const State::Idle& entry) {
                    return !fresh && (!stamp.valid || entry.stamp.same_as(stamp));
                });
                for (auto it = stale; it != list.end(); ++it) {
                    to_close.push_back(std::move(it->stream));
                    --state_->open_streams;
                }
                list.erase(stale, list.end());
                if (!list.empty()) {
                    stream = std::move(list.back().stream);
                    list.pop_back();
                    ++state_->reused;
                }
                if (list.empty()) {
                    state_->idle.erase(found);
                }
            }
        }
        to_close.clear();

        if (!stream) {
            auto opened   = std::make_unique<std::ifstream>();
            const bool ok = open_with_retry(path, [&]() {
                opened->clear();
                opened->open(path, std::ios::binary);
                return opened->is_open();
            });
            if (!ok) {
                LOG_ERROR("failed to open '%s'", path.c_str());
                return Lease();
            }
            stream = std::move(opened);
            std::lock_guard<std::mutex> lock(state_->mutex);
            ++state_->opened;
            ++state_->open_streams;
        }

        Lease lease;
        lease.state_      = state_;
        lease.stream_     = std::move(stream);
        lease.path_       = path;
        lease.stamp_      = stamp;
        lease.generation_ = generation;
        return lease;
    }

    void FileHandlePool::set_max_idle(size_t max_idle_per_file) {
        std::vector<std::unique_ptr<std::ifstream>> to_close;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->max_idle = std::max<size_t>(max_idle_per_file, 1);
            for (auto it = state_->idle.begin(); it != state_->idle.end();) {
                auto& list = it->second;
                while (list.size() > state_->max_idle) {
                    to_close.push_back(std::move(list.back().stream));
                    list.pop_back();
                    --state_->open_streams;
                }
                it = list.empty() ? state_->idle.erase(it) : std::next(it);
            }
        }
    }

    void FileHandlePool::close_idle() {
        std::vector<std::unique_ptr<std::ifstream>> to_close;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            ++state_->generation;
            for (auto& entry : state_->idle) {
                for (auto& idle : entry.second) {
                    to_close.push_back(std::move(idle.stream));
                    --state_->open_streams;
                }
            }
            state_->idle.clear();
        }
    }

    void FileHandlePool::close_idle(const std::string& path) {
        std::vector<std::unique_ptr<std::ifstream>> to_close;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            auto found = state_->idle.find(path);
            if (found == state_->idle.end()) {
                return;
            }
            for (auto& idle : found->second) {
                to_close.push_back(std::move(idle.stream));
                --state_->open_streams;
            }
            state_->idle.erase(found);
        }
    }

    size_t FileHandlePool::open_count() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->open_streams;
    }

    size_t FileHandlePool::opened_total() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->opened;
    }

    size_t FileHandlePool::reused_total() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->reused;
    }

    FileHandlePool::Lease::Lease(Lease&& other) noexcept
        : state_(std::move(other.state_)),
          stream_(std::move(other.stream_)),
          path_(std::move(other.path_)),
          stamp_(other.stamp_),
          generation_(other.generation_),
          discarded_(other.discarded_) {
    }

    FileHandlePool::Lease& FileHandlePool::Lease::operator=(Lease&& other) noexcept {
        if (this != &other) {
            release();
            state_      = std::move(other.state_);
            stream_     = std::move(other.stream_);
            path_       = std::move(other.path_);
            stamp_      = other.stamp_;
            generation_ = other.generation_;
            discarded_  = other.discarded_;
        }
        return *this;
    }

    FileHandlePool::Lease::~Lease() {
        release();
    }

    void FileHandlePool::Lease::discard() {
        discarded_ = true;
        release();
    }

    void FileHandlePool::Lease::release() {
        if (!stream_) {
            return;
        }
        std::unique_ptr<std::ifstream> stream = std::move(stream_);
        std::shared_ptr<State> state          = std::move(state_);
        const bool discarded                  = discarded_;
        discarded_                            = false;

        std::unique_ptr<std::ifstream> to_close;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            // Reaching the end of the file is normal; any other error leaves the stream unusable.
            const bool healthy = !discarded && stream->is_open() && !stream->fail();
            auto& list         = state->idle[path_];
            if (healthy && generation_ == state->generation && list.size() < state->max_idle) {
                stream->clear();
                list.push_back({std::move(stream), stamp_});
            } else {
                to_close = std::move(stream);
                --state->open_streams;
                if (list.empty()) {
                    state->idle.erase(path_);
                }
            }
        }
    }

}  // namespace sd
