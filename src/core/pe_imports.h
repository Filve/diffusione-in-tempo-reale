#ifndef __SD_CORE_PE_IMPORTS_H__
#define __SD_CORE_PE_IMPORTS_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sd::pe {

    // Reads `len` bytes at absolute file offset `offset` into `dst`; false when the range
    // is not inside the file.
    using ReadAt = std::function<bool(uint64_t offset, void* dst, size_t len)>;

    // Names of the DLLs a Windows PE image (exe/dll) imports through its import
    // directory, in file order. Delay-load imports are not listed. The result is empty
    // when the data is not a PE image or is truncated or inconsistent: the parser
    // validates every offset it follows, so it is safe on untrusted files.
    std::vector<std::string> imported_dlls(const ReadAt& read_at);

}  // namespace sd::pe

#endif  // __SD_CORE_PE_IMPORTS_H__
