#include "core/backend_diag.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "core/pe_imports.h"
#include "core/util.h"
#include "ggml-backend.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#endif

namespace sd::backend_diag {

    namespace {

        namespace fs = std::filesystem;

        struct GpuModule {
            const char* stem;      // module file: [lib]ggml-<stem>.<dll|so>
            const char* reg_name;  // name the backend registers under in ggml
            const char* label;
        };

        const GpuModule kGpuModules[] = {
            {"cuda", "CUDA", "CUDA"},
            {"hip", "ROCm", "ROCm"},
            {"musa", "MUSA", "MUSA"},
            {"vulkan", "Vulkan", "Vulkan"},
            {"metal", "Metal", "Metal"},
            {"sycl", "SYCL", "SYCL"},
            {"opencl", "OpenCL", "OpenCL"},
            {"cann", "CANN", "CANN"},
        };

        std::string module_file_name(const GpuModule& module) {
#ifdef _WIN32
            return std::string("ggml-") + module.stem + ".dll";
#else
            return std::string("libggml-") + module.stem + ".so";
#endif
        }

        // Same places ggml searches for its modules.
        fs::path executable_dir() {
#if defined(_WIN32)
            std::vector<wchar_t> buffer(MAX_PATH);
            for (;;) {
                const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
                if (length == 0) {
                    return {};
                }
                if (length < buffer.size()) {
                    return fs::path(std::wstring(buffer.data(), length)).parent_path();
                }
                buffer.resize(buffer.size() * 2);
            }
#elif defined(__APPLE__)
            uint32_t size = 0;
            _NSGetExecutablePath(nullptr, &size);
            std::vector<char> buffer(size + 1, '\0');
            if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
                return {};
            }
            return fs::path(buffer.data()).parent_path();
#elif defined(__linux__)
            std::error_code ec;
            const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
            return ec ? fs::path() : exe.parent_path();
#else
            return {};
#endif
        }

        std::vector<fs::path> search_dirs() {
            std::vector<fs::path> dirs;
            auto add = [&dirs](const fs::path& dir) {
                std::error_code ec;
                if (dir.empty() || !fs::is_directory(dir, ec)) {
                    return;
                }
                for (const fs::path& known : dirs) {
                    if (fs::equivalent(known, dir, ec)) {
                        return;
                    }
                }
                dirs.push_back(dir);
            };
            add(executable_dir());
            std::error_code ec;
            add(fs::current_path(ec));
            return dirs;
        }

        const char* score_or_init_problem(bool (*has_symbol)(void*, const char*), void* handle, int (*score)()) {
            if (score != nullptr && score() == 0) {
                return "the module reports that it is not supported on this machine";
            }
            if (!has_symbol(handle, "ggml_backend_init")) {
                return "the module has no ggml_backend_init entry point (not a ggml backend, or built for another ggml version)";
            }
            return "the module loaded but provides no device; the backend logs the cause (usually a missing or outdated GPU driver, or no compatible GPU)";
        }

#ifdef _WIN32

        std::string wide_to_utf8(const std::wstring& wide) {
            if (wide.empty()) {
                return {};
            }
            const int size = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
            if (size <= 0) {
                return {};
            }
            std::string utf8(static_cast<size_t>(size), '\0');
            WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), &utf8[0], size, nullptr, nullptr);
            return utf8;
        }

        std::string windows_error_text(DWORD error) {
            std::string text   = "error " + std::to_string(static_cast<unsigned long>(error));
            wchar_t* message   = nullptr;
            const DWORD length = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                                nullptr, error, 0, reinterpret_cast<LPWSTR>(&message), 0, nullptr);
            if (length > 0 && message != nullptr) {
                std::string detail = wide_to_utf8(std::wstring(message, length));
                while (!detail.empty() && (detail.back() == '\n' || detail.back() == '\r' || detail.back() == ' ')) {
                    detail.pop_back();
                }
                if (!detail.empty()) {
                    text += ": " + detail;
                }
            }
            if (message != nullptr) {
                LocalFree(message);
            }
            switch (error) {
                case ERROR_MOD_NOT_FOUND:
                    text += " (the module or a DLL it depends on was not found)";
                    break;
                case ERROR_PROC_NOT_FOUND:
                    text += " (a DLL it depends on has a different version than expected)";
                    break;
                case ERROR_DLL_INIT_FAILED:
                    text += " (the initialization of the module or of a DLL it depends on failed)";
                    break;
                case ERROR_BAD_EXE_FORMAT:
                    text += " (not valid for this process: wrong architecture or damaged file)";
                    break;
                default:
                    break;
            }
            return text;
        }

        bool is_api_set(const std::string& dll) {
            auto starts_with = [&dll](const char* prefix) {
                const size_t n = std::strlen(prefix);
                if (dll.size() < n) {
                    return false;
                }
                for (size_t i = 0; i < n; ++i) {
                    if (std::tolower(static_cast<unsigned char>(dll[i])) != prefix[i]) {
                        return false;
                    }
                }
                return true;
            };
            return starts_with("api-ms-win-") || starts_with("ext-ms-");
        }

        // DLLs the module imports that the loader cannot find. SearchPathW follows the
        // search order the loader uses for dependencies: the executable directory first.
        std::vector<std::string> missing_imports(const fs::path& module) {
            std::ifstream file(module, std::ios::binary);
            std::vector<std::string> missing;
            if (!file) {
                return missing;
            }
            const std::vector<std::string> imports = sd::pe::imported_dlls([&file](uint64_t offset, void* dst, size_t len) {
                file.clear();
                file.seekg(static_cast<std::streamoff>(offset));
                file.read(static_cast<char*>(dst), static_cast<std::streamsize>(len));
                return static_cast<bool>(file) && static_cast<size_t>(file.gcount()) == len;
            });
            for (const std::string& dll : imports) {
                if (is_api_set(dll)) {
                    continue;
                }
                const std::wstring wide(dll.begin(), dll.end());
                wchar_t found[MAX_PATH];
                if (SearchPathW(nullptr, wide.c_str(), nullptr, MAX_PATH, found, nullptr) == 0) {
                    missing.push_back(dll);
                }
            }
            return missing;
        }

        void* find_symbol(HMODULE module, const char* name) {
            return reinterpret_cast<void*>(GetProcAddress(module, name));
        }

        // Why the module cannot provide a device, found by loading it the way ggml does.
        std::string probe_module(const fs::path& path) {
            const DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
            SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
            HMODULE handle    = LoadLibraryW(path.wstring().c_str());
            const DWORD error = GetLastError();
            SetErrorMode(old_mode);

            if (handle == nullptr) {
                std::string reason = windows_error_text(error);
                if (error == ERROR_MOD_NOT_FOUND) {
                    const std::vector<std::string> missing = missing_imports(path);
                    if (!missing.empty()) {
                        reason += "; missing DLLs:";
                        for (const std::string& dll : missing) {
                            reason += " " + dll;
                        }
                    }
                }
                return reason;
            }
            auto score      = reinterpret_cast<int (*)()>(find_symbol(handle, "ggml_backend_score"));
            auto has_symbol = [](void* module, const char* name) {
                return find_symbol(static_cast<HMODULE>(module), name) != nullptr;
            };
            std::string reason = score_or_init_problem(has_symbol, handle, score);
            FreeLibrary(handle);
            return reason;
        }

#else

        // Why the module cannot provide a device, found by loading it the way ggml does.
        std::string probe_module(const fs::path& path) {
            void* handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle == nullptr) {
                const char* error = dlerror();
                return error != nullptr ? error : "dlopen failed";
            }
            auto score      = reinterpret_cast<int (*)()>(dlsym(handle, "ggml_backend_score"));
            auto has_symbol = [](void* module, const char* name) {
                return dlsym(module, name) != nullptr;
            };
            std::string reason = score_or_init_problem(has_symbol, handle, score);
            dlclose(handle);
            return reason;
        }

#endif

        const char* hint_for(const GpuModule& module) {
            const std::string stem = module.stem;
#ifdef _WIN32
            if (stem == "cuda") {
                return "check that the NVIDIA driver is installed and that cudart64_*.dll, cublas64_*.dll and cublasLt64_*.dll are in the same folder as the executable (the cuda12 package includes them)";
            }
            if (stem == "vulkan") {
                return "Vulkan comes with the graphics driver (vulkan-1.dll): install or update the GPU driver";
            }
#else
            if (stem == "cuda") {
                return "check that the NVIDIA driver and the CUDA 12 runtime libraries (libcudart, libcublas) are installed";
            }
            if (stem == "vulkan") {
                return "install the Vulkan loader (libvulkan1 or vulkan-icd-loader) and the Vulkan driver of the GPU";
            }
#endif
            if (stem == "hip") {
                return "check that the AMD ROCm/HIP runtime and driver are installed";
            }
            if (stem == "sycl") {
                return "check that the Intel oneAPI runtime is installed";
            }
            if (stem == "opencl") {
                return "check that an OpenCL driver (ICD) is installed";
            }
            return "check the driver and runtime libraries of this backend";
        }

        size_t gpu_device_count() {
            size_t count = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    ++count;
                }
            }
            return count;
        }

        void report() {
            const std::vector<fs::path> dirs = search_dirs();
            bool module_found                = false;

            for (const GpuModule& module : kGpuModules) {
                const std::string file_name = module_file_name(module);
                fs::path module_path;
                for (const fs::path& dir : dirs) {
                    std::error_code ec;
                    const fs::path candidate = dir / file_name;
                    if (fs::is_regular_file(candidate, ec)) {
                        module_path = candidate;
                        break;
                    }
                }
                if (module_path.empty()) {
                    continue;
                }
                module_found = true;

                ggml_backend_reg_t reg = ggml_backend_reg_by_name(module.reg_name);
                if (reg != nullptr && ggml_backend_reg_dev_count(reg) > 0) {
                    LOG_DEBUG("%s backend: %zu device(s) from %s", module.label, ggml_backend_reg_dev_count(reg), module_path.string().c_str());
                    continue;
                }
                LOG_WARN("%s backend: %s found in %s was not usable: %s. Hint: %s",
                         module.label, file_name.c_str(), module_path.parent_path().string().c_str(),
                         probe_module(module_path).c_str(), hint_for(module));
            }

            if (gpu_device_count() == 0) {
                std::string searched;
                for (const fs::path& dir : dirs) {
                    searched += (searched.empty() ? "" : ", ") + dir.string();
                }
                if (module_found) {
                    LOG_INFO("no GPU device is available: only the CPU backend will be used");
                } else {
                    LOG_INFO("no GPU backend module (ggml-cuda, ggml-vulkan, ...) found in: %s; only the CPU backend will be used (a CPU-only build or package, or the module files are not next to the executable)",
                             searched.empty() ? "(no readable directory)" : searched.c_str());
                }
            }
        }

    }  // namespace

    void report_backend_modules() {
        try {
            report();
        } catch (const std::exception& e) {
            LOG_DEBUG("backend diagnostics failed: %s", e.what());
        }
    }

}  // namespace sd::backend_diag
