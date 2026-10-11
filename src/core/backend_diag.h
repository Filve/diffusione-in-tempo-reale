#ifndef __SD_CORE_BACKEND_DIAG_H__
#define __SD_CORE_BACKEND_DIAG_H__

namespace sd::backend_diag {

    // Call once, right after the dynamic ggml backend modules were loaded.
    //
    // ggml loads them silently in Release builds and, on Windows, never reports the OS
    // error, so a GPU module that is on disk but fails to load simply makes the GPU
    // disappear. This logs, for each GPU module that was found next to the executable or
    // in the working directory but did not provide a device, why it could not be used, and
    // a note when no GPU device is available at all.
    void report_backend_modules();

}  // namespace sd::backend_diag

#endif  // __SD_CORE_BACKEND_DIAG_H__
