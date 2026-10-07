#ifndef __MEMORY_WATCHDOG_H__
#define __MEMORY_WATCHDOG_H__

#include <cstddef>
#include <map>
#include <set>

#include "ggml-backend.h"

namespace sd {

    // Tracks per-device memory pressure against a user-selected threshold and
    // decides when capacity management should act before an allocation failure.
    class MemoryWatchdog {
    public:
        enum class Pressure {
            None,
            High,
            Critical,
        };

        struct Sample {
            size_t used_bytes  = 0;
            size_t limit_bytes = 0;

            bool valid() const { return limit_bytes > 0; }
        };

        // 0 disables the guard; enabled values are clamped to [50, 99].
        void set_threshold_percent(int percent);
        bool enabled() const { return threshold_percent_ > 0; }
        int threshold_percent() const { return threshold_percent_; }
        int critical_percent() const;

        Pressure classify(const Sample& sample) const;

        // Records the backend's pressure state and logs transitions.
        Pressure observe(ggml_backend_t backend, const Sample& sample);

        // True while any observed backend is at High pressure or above.
        bool prefetch_suspended() const;

        // Rate limit for "guard target unreachable" warnings: true at most once
        // per backend until its pressure recovers.
        bool should_log_unreachable(ggml_backend_t backend);

    private:
        int threshold_percent_ = 0;
        std::map<ggml_backend_t, Pressure> states_;
        std::set<ggml_backend_t> unreachable_logged_;
    };

}  // namespace sd

#endif  // __MEMORY_WATCHDOG_H__
