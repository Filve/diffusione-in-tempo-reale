#ifndef __SD_CORE_CUDA_DRIVER_H__
#define __SD_CORE_CUDA_DRIVER_H__

namespace sd::cuda_driver {

    // Compute capability of the CUDA device with this PCI bus id, read through the NVIDIA driver.
    // False when the driver is not installed or does not know the device.
    bool compute_capability(const char* pci_bus_id, int& major, int& minor);

}  // namespace sd::cuda_driver

#endif  // __SD_CORE_CUDA_DRIVER_H__
