#include "core/cuda_driver.h"

#ifdef SD_USE_CUDA
#include <cuda.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#endif
#else
#include <dlfcn.h>
#endif
#endif

namespace sd::cuda_driver {

#ifdef SD_USE_CUDA

    namespace {

        // The driver is looked up when it is needed instead of being linked: a library that imports it does
        // not load on a machine without an NVIDIA driver, and the CUDA module of a portable package is optional.
        struct Driver {
            CUresult(CUDAAPI* init)(unsigned int)                                   = nullptr;
            CUresult(CUDAAPI* device_by_pci_bus_id)(CUdevice*, const char*)         = nullptr;
            CUresult(CUDAAPI* device_attribute)(int*, CUdevice_attribute, CUdevice) = nullptr;
        };

#ifdef _WIN32
        using Handle = HMODULE;

        Handle open_driver() {
            return LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        }

        template <typename Function>
        Function find_function(Handle handle, const char* name) {
            return reinterpret_cast<Function>(reinterpret_cast<void*>(GetProcAddress(handle, name)));
        }
#else
        using Handle = void*;

        Handle open_driver() {
            return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        }

        template <typename Function>
        Function find_function(Handle handle, const char* name) {
            return reinterpret_cast<Function>(dlsym(handle, name));
        }
#endif

        const Driver& driver() {
            static const Driver loaded = [] {
                Driver found;
                if (Handle handle = open_driver()) {
                    found.init                 = find_function<decltype(found.init)>(handle, "cuInit");
                    found.device_by_pci_bus_id = find_function<decltype(found.device_by_pci_bus_id)>(handle, "cuDeviceGetByPCIBusId");
                    found.device_attribute     = find_function<decltype(found.device_attribute)>(handle, "cuDeviceGetAttribute");
                }
                return found;
            }();
            return loaded;
        }

    }  // namespace

    bool compute_capability(const char* pci_bus_id, int& major, int& minor) {
        const Driver& cuda = driver();
        if (pci_bus_id == nullptr || cuda.init == nullptr || cuda.device_by_pci_bus_id == nullptr || cuda.device_attribute == nullptr) {
            return false;
        }
        CUdevice device;
        return cuda.init(0) == CUDA_SUCCESS &&
               cuda.device_by_pci_bus_id(&device, pci_bus_id) == CUDA_SUCCESS &&
               cuda.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == CUDA_SUCCESS &&
               cuda.device_attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) == CUDA_SUCCESS;
    }

#else

    bool compute_capability(const char*, int&, int&) {
        return false;
    }

#endif

}  // namespace sd::cuda_driver
