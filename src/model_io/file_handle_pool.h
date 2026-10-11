#ifndef __SD_FILE_HANDLE_POOL_H__
#define __SD_FILE_HANDLE_POOL_H__

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>

namespace sd {

    // Calls try_open until it succeeds, a few times at most, and logs the OS error after each
    // failure. Opening a file can fail for a moment on Windows (sharing violations, scarce
    // system resources); a short wait usually turns that into a success.
    bool open_with_retry(const std::string& path, const std::function<bool()>& try_open);

    // Keeps read-only streams of model files open between tensor loads. Weights that live on
    // disk are read again for every segment and every request; opening the file each time
    // is a chance to fail exactly when the system is short of memory or handles.
    //
    // Copies of a pool share the same handles. All members are thread-safe; a stream is used
    // by one lease at a time.
    class FileHandlePool {
        struct State;
        struct Stamp {
            bool valid                           = false;
            uintmax_t size                       = 0;
            std::filesystem::file_time_type time = {};

            bool same_as(const Stamp& other) const {
                return valid && other.valid && size == other.size && time == other.time;
            }
        };

    public:
        class Lease {
        public:
            Lease() = default;
            Lease(Lease&& other) noexcept;
            Lease& operator=(Lease&& other) noexcept;
            Lease(const Lease&)            = delete;
            Lease& operator=(const Lease&) = delete;
            ~Lease();

            explicit operator bool() const { return stream_ != nullptr; }
            std::ifstream& stream() { return *stream_; }

            // Closes the stream instead of giving it back, for a stream that failed or
            // that may be stale.
            void discard();

        private:
            friend class FileHandlePool;

            void release();

            std::shared_ptr<State> state_;
            std::unique_ptr<std::ifstream> stream_;
            std::string path_;
            Stamp stamp_;
            uint64_t generation_ = 0;
            bool discarded_      = false;
        };

        FileHandlePool();
        // Copying shares the open streams. There is deliberately no move: a moved-from
        // ModelLoader must stay usable.
        FileHandlePool(const FileHandlePool&)            = default;
        FileHandlePool& operator=(const FileHandlePool&) = default;

        // Returns an open stream for path, reusing an idle one when the file has not changed
        // since it was opened. With fresh set, idle streams of that path are closed and the
        // file is opened again. The lease is empty when the file cannot be opened.
        Lease acquire(const std::string& path, bool fresh = false);

        // At most this many idle streams are kept per file.
        void set_max_idle(size_t max_idle_per_file);

        // Closes every idle stream; streams that are leased now are closed when returned.
        void close_idle();

        // Closes the idle streams of one file.
        void close_idle(const std::string& path);

        size_t open_count() const;    // streams that are open, idle or leased
        size_t opened_total() const;  // files opened since the pool was created
        size_t reused_total() const;  // acquisitions served by an idle stream

    private:
        static Stamp read_stamp(const std::string& path);

        std::shared_ptr<State> state_;
    };

}  // namespace sd

#endif  // __SD_FILE_HANDLE_POOL_H__
