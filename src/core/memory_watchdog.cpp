#include "core/memory_watchdog.h"

#include <algorithm>

#include "core/util.h"

namespace sd {

    void MemoryWatchdog::set_threshold_percent(int percent) {
        if (percent <= 0) {
            threshold_percent_ = 0;
            states_.clear();
            unreachable_logged_.clear();
            return;
        }
        const int clamped = std::min(std::max(percent, 50), 99);
        if (clamped != percent) {
            LOG_WARN("memory guard threshold %d%% is out of range, using %d%%", percent, clamped);
        }
        threshold_percent_ = clamped;
    }

    int MemoryWatchdog::critical_percent() const {
        return std::min(threshold_percent_ + 5, 99);
    }

    MemoryWatchdog::Pressure MemoryWatchdog::classify(const Sample& sample) const {
        if (!enabled() || !sample.valid()) {
            return Pressure::None;
        }
        // used/limit*100 without intermediate overflow on large byte counts.
        const size_t percent_unit = std::max<size_t>(sample.limit_bytes / 100, 1);
        if (sample.used_bytes / percent_unit >= static_cast<size_t>(critical_percent())) {
            return Pressure::Critical;
        }
        if (sample.used_bytes / percent_unit >= static_cast<size_t>(threshold_percent_)) {
            return Pressure::High;
        }
        return Pressure::None;
    }

    MemoryWatchdog::Pressure MemoryWatchdog::observe(ggml_backend_t backend, const Sample& sample) {
        const Pressure pressure = classify(sample);
        if (!enabled() || backend == nullptr) {
            return pressure;
        }
        Pressure& state = states_[backend];
        if (pressure != state) {
            const double used_mb  = sample.used_bytes / (1024.0 * 1024.0);
            const double limit_mb = sample.limit_bytes / (1024.0 * 1024.0);
            if (pressure == Pressure::None) {
                LOG_INFO("memory guard: pressure on %s recovered below %d%% (%.2f MB / %.2f MB)",
                         ggml_backend_name(backend), threshold_percent_, used_mb, limit_mb);
                unreachable_logged_.erase(backend);
            } else {
                LOG_WARN("memory guard: %s pressure on %s at %d%% threshold (%.2f MB / %.2f MB), reducing residency",
                         pressure == Pressure::Critical ? "critical" : "high",
                         ggml_backend_name(backend),
                         pressure == Pressure::Critical ? critical_percent() : threshold_percent_,
                         used_mb, limit_mb);
            }
            state = pressure;
        }
        return pressure;
    }

    bool MemoryWatchdog::prefetch_suspended() const {
        if (!enabled()) {
            return false;
        }
        for (const auto& entry : states_) {
            if (entry.second != Pressure::None) {
                return true;
            }
        }
        return false;
    }

    bool MemoryWatchdog::should_log_unreachable(ggml_backend_t backend) {
        return unreachable_logged_.insert(backend).second;
    }

}  // namespace sd
