#ifndef STABLE_TORCH_PYTHON_SHIM
#define STABLE_TORCH_PYTHON_SHIM

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/version.h>

// Stable C entry points for converting between a Python torch.Tensor
// (PyObject*) and an AtenTensorHandle. PyObject* values cross the ABI as
// opaque void* so this header does not depend on Python.h. The
// implementation lives in libtorch_python.so since it uses THPVariable_*,
// so consumers must also link against libtorch_python. The GIL must be held
// by the caller.

#ifdef __cplusplus
extern "C" {
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

// Wrap a Python torch.Tensor as a new AtenTensorHandle that shares the
// underlying TensorImpl with the input.
AOTI_TORCH_EXPORT AOTITorchError torch_tensor_from_pyobject(
    void* py_obj,
    AtenTensorHandle* ret); // returns new reference

// Wrap an AtenTensorHandle as a Python torch.Tensor. If py_type is
// non-null, it is used as the result's exact PyTypeObject* (e.g.
// torch.nn.Parameter); nullptr means the default torch.Tensor type. On
// failure the Python error indicator is left set.
AOTI_TORCH_EXPORT AOTITorchError torch_tensor_to_pyobject(
    AtenTensorHandle ath,
    void* py_type,
    void** ret); // returns new reference

// Reads the ScalarType from a Python torch.dtype object. The result is
// returned as an aoti_torch dtype code (same encoding as the
// aoti_torch_dtype_*() getters).
AOTI_TORCH_EXPORT AOTITorchError torch_scalartype_from_pyobject(
    void* py_obj,
    int32_t* ret_dtype);

// Produces a new-reference Python torch.dtype object for the given aoti_torch
// dtype code.
AOTI_TORCH_EXPORT AOTITorchError torch_scalartype_to_pyobject(
    int32_t dtype,
    void** ret); // returns new reference

// Reads the device type and index from a Python torch.device object.
// device_type uses the same encoding as the aoti_torch_device_type_*()
// getters.
AOTI_TORCH_EXPORT AOTITorchError torch_device_from_pyobject(
    void* py_obj,
    int32_t* ret_device_type,
    int32_t* ret_device_index);

// Produces a new-reference Python torch.device object for the given device
// type (aoti_torch_device_type_*() encoding) and index.
AOTI_TORCH_EXPORT AOTITorchError torch_device_to_pyobject(
    int32_t device_type,
    int32_t device_index,
    void** ret); // returns new reference

// Wraps a Python torch.distributed.ProcessGroup (PyObject* passed as void*) as
// a new owning ProcessGroupHandle that shares the underlying
// c10d::ProcessGroup. The caller must free it with torch_delete_processgroup.
// The GIL must be held. Requires a libtorch built with USE_DISTRIBUTED.
AOTI_TORCH_EXPORT AOTITorchError torch_processgroup_from_pyobject(
    void* py_obj,
    ProcessGroupHandle* ret); // returns new owning handle

#ifdef USE_CUDA

// Obtains the Philox RNG state from a Python torch.Generator (which must wrap a
// CUDA generator), equivalent to gen->philox_cuda_state(increment) under the
// generator lock. This advances the generator's offset by `increment`
// (elements per thread), exactly like the native call site.
//
// The seed and offset are returned as the raw 64-bit bits of
// at::PhiloxCudaState's Payload union. When ret_captured is 0 (no CUDA graph
// capture underway) they are literal seed/offset values. When ret_captured is
// 1, they are device int64_t* pointers (reinterpreted as uint64_t) and
// ret_offset_intragraph carries the intra-graph offset. The consumer
// reconstructs an at::PhiloxCudaState from these fields (the raw struct in
// ATen/cuda/detail/PhiloxCudaStateRaw.cuh is designed to be copied into
// consumer device code):
//   captured==0: PhiloxCudaState(seed, offset)
//   captured==1: PhiloxCudaState((int64_t*)seed, (int64_t*)offset,
//                                offset_intragraph)
AOTI_TORCH_EXPORT AOTITorchError torch_cuda_generator_get_philox_state(
    void* py_generator,
    uint64_t increment,
    uint64_t* ret_seed,
    uint64_t* ret_offset,
    uint64_t* ret_offset_intragraph,
    int32_t* ret_captured);

#endif // USE_CUDA

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_TORCH_PYTHON_SHIM
