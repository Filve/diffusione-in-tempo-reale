#ifndef __SD_CORE_SYSTEM_MEMORY_H__
#define __SD_CORE_SYSTEM_MEMORY_H__

#include <cstdint>

namespace sd::system_memory {

    // Every field is -1 when the platform cannot report it.
    struct Info {
        int64_t total_bytes     = -1;  // physical memory
        int64_t available_bytes = -1;  // usable without swapping
        int64_t process_bytes   = -1;  // resident memory of this process
    };

    Info query();

}  // namespace sd::system_memory

#endif  // __SD_CORE_SYSTEM_MEMORY_H__
