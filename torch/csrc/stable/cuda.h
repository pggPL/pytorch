#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/util/shim_utils.h>

#include <utility>

// Stable ABI wrappers for CUDA-specific device queries needed to compute kernel
// launch configuration (SM count, compute capability) and to adopt an
// externally-provided cudaStream_t. These require the extension to be built
// with USE_CUDA defined, matching the CUDA shims in shim.h.

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0 && defined(USE_CUDA)

HIDDEN_NAMESPACE_BEGIN(torch, stable, cuda)

using torch::stable::accelerator::DeviceIndex;
using torch::stable::accelerator::Stream;

/**
 * @brief Number of streaming multiprocessors (SMs) on a CUDA device.
 *
 * @param device_index The CUDA device index to query.
 * @return cudaDeviceProp::multiProcessorCount for that device.
 *
 * Minimum compatible version: PyTorch 2.14.
 */
inline int32_t getDeviceMultiProcessorCount(DeviceIndex device_index) {
  int32_t sm_count = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_cuda_get_device_multiprocessor_count(device_index, &sm_count));
  return sm_count;
}

/**
 * @brief CUDA compute capability of a device as (major, minor).
 *
 * @param device_index The CUDA device index to query.
 * @return A pair {major, minor}; the "sm_arch" is major * 10 + minor.
 *
 * Minimum compatible version: PyTorch 2.14.
 */
inline std::pair<int32_t, int32_t> getDeviceComputeCapability(
    DeviceIndex device_index) {
  int32_t major = 0;
  int32_t minor = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cuda_get_device_compute_capability(
      device_index, &major, &minor));
  return {major, minor};
}

/**
 * @brief Wrap an externally-owned cudaStream_t in a stable Stream.
 *
 * The returned Stream does not take ownership of the underlying cudaStream_t;
 * its lifetime remains the caller's responsibility. Only the stable wrapper is
 * freed when the Stream is destroyed.
 *
 * @param ext_stream A cudaStream_t passed as void*.
 * @param device_index The device the stream is associated with.
 *
 * Minimum compatible version: PyTorch 2.14.
 */
inline Stream getStreamFromExternal(void* ext_stream, DeviceIndex device_index) {
  StreamHandle stream = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_get_cuda_stream_from_external(ext_stream, device_index, &stream));
  return Stream(stream);
}

HIDDEN_NAMESPACE_END(torch, stable, cuda)

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0 && defined(USE_CUDA)
