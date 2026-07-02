#pragma once

#include <torch/csrc/stable/c10d.h>
#include <torch/csrc/stable/python/c/shim.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

// Header-only helpers for converting between a Python torch.Tensor (passed
// as a raw PyObject* / void*) and torch::stable::Tensor. Binding-framework
// specific casters (pybind11, nanobind, ...) live in separate opt-in
// headers that build on top of these. The GIL must be held by the caller.

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

// Wrap a Python torch.Tensor (PyObject* passed as void*) as a stable Tensor
// that shares its underlying TensorImpl.
inline Tensor from_pyobject(void* py_obj) {
  AtenTensorHandle ath{};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_tensor_from_pyobject(py_obj, &ath));
  return Tensor(ath);
}

// Wrap a stable Tensor as a new-reference Python torch.Tensor. py_type is
// an optional PyTypeObject* (passed as void*) used as the result's exact
// Python type; nullptr means default torch.Tensor.
inline void* to_pyobject(const Tensor& t, void* py_type = nullptr) {
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_tensor_to_pyobject(t.get(), py_type, &raw));
  return raw;
}

// Wrap a Python torch.distributed.ProcessGroup (PyObject* passed as void*) as a
// stable ProcessGroup that shares the underlying c10d::ProcessGroup. The GIL
// must be held.
inline ProcessGroup processgroup_from_pyobject(void* py_obj) {
  ProcessGroupHandle handle = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_processgroup_from_pyobject(py_obj, &handle));
  return ProcessGroup(handle);
}

// Read the ScalarType from a Python torch.dtype (PyObject* passed as void*).
inline torch::headeronly::ScalarType scalartype_from_pyobject(void* py_obj) {
  int32_t code = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_scalartype_from_pyobject(py_obj, &code));
  return torch::stable::detail::to<torch::headeronly::ScalarType>(
      torch::stable::detail::from(code));
}

// Produce a new-reference Python torch.dtype (PyObject* as void*) for a
// ScalarType.
inline void* scalartype_to_pyobject(torch::headeronly::ScalarType dtype) {
  int32_t code = torch::stable::detail::to<int32_t>(
      torch::stable::detail::from(dtype));
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_scalartype_to_pyobject(code, &raw));
  return raw;
}

// Read a Device from a Python torch.device (PyObject* passed as void*).
inline Device device_from_pyobject(void* py_obj) {
  int32_t device_type = 0;
  int32_t device_index = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_device_from_pyobject(py_obj, &device_type, &device_index));
  DeviceType dt = torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
  return Device(dt, static_cast<DeviceIndex>(device_index));
}

// Produce a new-reference Python torch.device (PyObject* as void*) for a
// Device.
inline void* device_to_pyobject(const Device& device) {
  int32_t device_type = torch::stable::detail::to<int32_t>(
      torch::stable::detail::from(device.type()));
  void* raw = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_device_to_pyobject(
      device_type, static_cast<int32_t>(device.index()), &raw));
  return raw;
}

#ifdef USE_CUDA

// Raw Philox RNG state read from a torch.Generator. Mirrors the fields of
// at::PhiloxCudaState across the stable boundary; `seed` and `offset` are the
// raw 64-bit Payload bits (literal values when !captured, device int64_t*
// pointers reinterpreted as uint64_t when captured). The consumer reconstructs
// an at::PhiloxCudaState (raw struct in
// ATen/cuda/detail/PhiloxCudaStateRaw.cuh) from these fields.
struct PhiloxCudaState {
  uint64_t seed;
  uint64_t offset;
  uint64_t offset_intragraph;
  bool captured;
};

// Equivalent to gen->philox_cuda_state(increment) under the generator lock,
// where gen is the CUDA generator wrapped by the Python torch.Generator
// (PyObject* passed as void*). Advances the generator offset by `increment`
// (elements per thread).
inline PhiloxCudaState philox_cuda_state_from_pyobject(
    void* py_generator,
    uint64_t increment) {
  PhiloxCudaState out{};
  int32_t captured = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_cuda_generator_get_philox_state(
      py_generator,
      increment,
      &out.seed,
      &out.offset,
      &out.offset_intragraph,
      &captured));
  out.captured = captured != 0;
  return out;
}

#endif // USE_CUDA

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

HIDDEN_NAMESPACE_END(torch, stable)
