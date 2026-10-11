#include "core/system_memory.h"

#include <cstddef>
#include <fstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#include "core/backend_fit.h"
#include "stable-diffusion.h"

namespace sd::system_memory {

    Info query() {
        Info info;
        info.available_bytes = backend_fit::available_ram_bytes();
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            info.total_bytes = static_cast<int64_t>(status.ullTotalPhys);
        }
        PROCESS_MEMORY_COUNTERS counters{};
        counters.cb = sizeof(counters);
        if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
            info.process_bytes = static_cast<int64_t>(counters.WorkingSetSize);
        }
#elif defined(__APPLE__)
        uint64_t memory_size = 0;
        size_t length        = sizeof(memory_size);
        if (sysctlbyname("hw.memsize", &memory_size, &length, nullptr, 0) == 0) {
            info.total_bytes = static_cast<int64_t>(memory_size);
        }
        mach_task_basic_info_data_t task{};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&task), &count) == KERN_SUCCESS) {
            info.process_bytes = static_cast<int64_t>(task.resident_size);
        }
#elif defined(__linux__)
        {
            std::ifstream meminfo("/proc/meminfo");
            std::string key, unit;
            int64_t kib = 0;
            while (meminfo >> key >> kib >> unit) {
                if (key == "MemTotal:" && unit == "kB" && kib >= 0) {
                    info.total_bytes = kib * 1024;
                    break;
                }
            }
        }
        {
            std::ifstream statm("/proc/self/statm");
            int64_t pages_total = 0, pages_resident = 0;
            const long page_size = sysconf(_SC_PAGESIZE);
            if (statm >> pages_total >> pages_resident && page_size > 0 && pages_resident >= 0) {
                info.process_bytes = pages_resident * static_cast<int64_t>(page_size);
            }
        }
#endif
        return info;
    }

}  // namespace sd::system_memory

bool sd_get_system_memory(sd_system_memory_t* info) {
    if (info == nullptr) {
        return false;
    }
    *info                 = sd_system_memory_t{};
    const auto memory     = sd::system_memory::query();
    info->total_bytes     = memory.total_bytes > 0 ? static_cast<uint64_t>(memory.total_bytes) : 0;
    info->available_bytes = memory.available_bytes >= 0 ? static_cast<uint64_t>(memory.available_bytes) : 0;
    info->process_bytes   = memory.process_bytes >= 0 ? static_cast<uint64_t>(memory.process_bytes) : 0;
    return info->total_bytes > 0 || info->available_bytes > 0 || info->process_bytes > 0;
}
