#include <torch/csrc/stable/python/c/shim.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/python_headers.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#ifdef USE_DISTRIBUTED
#include <torch/csrc/shim_processgroup_internal.h>
#include <torch/csrc/utils/pybind.h>
#endif

using torch::aot_inductor::new_tensor_handle;
using torch::aot_inductor::tensor_handle_to_tensor_pointer;

extern "C" {

AOTITorchError torch_tensor_from_pyobject(
    void* py_obj,
    AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_tensor_from_pyobject requires the GIL to be held");
    TORCH_CHECK(py_obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(ret != nullptr, "ret must not be null");

    PyObject* obj = static_cast<PyObject*>(py_obj);
    TORCH_CHECK(
        THPVariable_Check(obj),
        "torch_tensor_from_pyobject: expected torch.Tensor, got ",
        Py_TYPE(obj)->tp_name);

    *ret = new_tensor_handle(at::Tensor(THPVariable_Unpack(obj)));
  });
}

AOTITorchError torch_tensor_to_pyobject(
    AtenTensorHandle ath,
    void* py_type,
    void** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_tensor_to_pyobject requires the GIL to be held");
    TORCH_CHECK(ath != nullptr, "ath must not be null");
    TORCH_CHECK(ret != nullptr, "ret must not be null");

    at::Tensor* t = tensor_handle_to_tensor_pointer(ath);

    PyObject* py = (py_type != nullptr)
        ? THPVariable_Wrap(*t, static_cast<PyTypeObject*>(py_type))
        : THPVariable_Wrap(*t);
    if (py == nullptr) {
      // Forward the Python error (left set by THPVariable_Wrap) through
      // AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE.
      throw python_error();
    }
    *ret = py;
  });
}

AOTITorchError torch_scalartype_from_pyobject(void* py_obj, int32_t* ret_dtype) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_scalartype_from_pyobject requires the GIL to be held");
    TORCH_CHECK(py_obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(ret_dtype != nullptr, "ret_dtype must not be null");

    PyObject* obj = static_cast<PyObject*>(py_obj);
    TORCH_CHECK(
        THPDtype_Check(obj),
        "torch_scalartype_from_pyobject: expected torch.dtype, got ",
        Py_TYPE(obj)->tp_name);

    // The aoti_torch dtype code is just the c10::ScalarType enum value.
    *ret_dtype =
        static_cast<int32_t>(reinterpret_cast<THPDtype*>(obj)->scalar_type);
  });
}

AOTITorchError torch_scalartype_to_pyobject(int32_t dtype, void** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_scalartype_to_pyobject requires the GIL to be held");
    TORCH_CHECK(ret != nullptr, "ret must not be null");

    // getTHPDtype returns a borrowed reference to the registered singleton.
    PyObject* py = reinterpret_cast<PyObject*>(
        torch::getTHPDtype(static_cast<at::ScalarType>(dtype)));
    Py_INCREF(py);
    *ret = py;
  });
}

AOTITorchError torch_device_from_pyobject(
    void* py_obj,
    int32_t* ret_device_type,
    int32_t* ret_device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_device_from_pyobject requires the GIL to be held");
    TORCH_CHECK(py_obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(ret_device_type != nullptr, "ret_device_type must not be null");
    TORCH_CHECK(
        ret_device_index != nullptr, "ret_device_index must not be null");

    PyObject* obj = static_cast<PyObject*>(py_obj);
    TORCH_CHECK(
        THPDevice_Check(obj),
        "torch_device_from_pyobject: expected torch.device, got ",
        Py_TYPE(obj)->tp_name);

    const at::Device& device = reinterpret_cast<THPDevice*>(obj)->device;
    *ret_device_type = static_cast<int32_t>(device.type());
    *ret_device_index = static_cast<int32_t>(device.index());
  });
}

AOTITorchError torch_device_to_pyobject(
    int32_t device_type,
    int32_t device_index,
    void** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_device_to_pyobject requires the GIL to be held");
    TORCH_CHECK(ret != nullptr, "ret must not be null");

    at::Device device(
        static_cast<c10::DeviceType>(device_type),
        static_cast<c10::DeviceIndex>(device_index));
    PyObject* py = THPDevice_New(device);
    if (py == nullptr) {
      throw python_error();
    }
    *ret = py;
  });
}

AOTITorchError torch_processgroup_from_pyobject(
    void* py_obj,
    ProcessGroupHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_processgroup_from_pyobject requires the GIL to be held");
    TORCH_CHECK(py_obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(ret != nullptr, "ret must not be null");
#ifdef USE_DISTRIBUTED
    py::handle handle(static_cast<PyObject*>(py_obj));
    auto pg = py::cast<c10::intrusive_ptr<c10d::ProcessGroup>>(handle);
    TORCH_CHECK(
        pg, "torch_processgroup_from_pyobject: got a null ProcessGroup");
    *ret = processgroup_pointer_to_handle(new ProcessGroupOpaque{std::move(pg)});
#else
    TORCH_CHECK(
        false,
        "torch_processgroup_from_pyobject: this libtorch was built without "
        "USE_DISTRIBUTED");
#endif
  });
}

#ifdef USE_CUDA

AOTITorchError torch_cuda_generator_get_philox_state(
    void* py_generator,
    uint64_t increment,
    uint64_t* ret_seed,
    uint64_t* ret_offset,
    uint64_t* ret_offset_intragraph,
    int32_t* ret_captured) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_cuda_generator_get_philox_state requires the GIL to be held");
    TORCH_CHECK(py_generator != nullptr, "py_generator must not be null");
    TORCH_CHECK(ret_seed != nullptr, "ret_seed must not be null");
    TORCH_CHECK(ret_offset != nullptr, "ret_offset must not be null");
    TORCH_CHECK(
        ret_offset_intragraph != nullptr,
        "ret_offset_intragraph must not be null");
    TORCH_CHECK(ret_captured != nullptr, "ret_captured must not be null");

    PyObject* obj = static_cast<PyObject*>(py_generator);
    TORCH_CHECK(
        THPGenerator_Check(obj),
        "torch_cuda_generator_get_philox_state: expected torch.Generator");

    at::Generator gen = THPGenerator_Unwrap(obj);
    auto* impl = at::check_generator<at::CUDAGeneratorImpl>(gen);

    at::PhiloxCudaState state;
    {
      // See Note [Acquire lock when using random generators].
      std::lock_guard<std::mutex> lock(gen.mutex());
      state = impl->philox_cuda_state(increment);
    }

    // Payload is a union of uint64_t val and int64_t* ptr; reading .val yields
    // the raw 64-bit bits for either case, which the consumer reinterprets
    // based on ret_captured.
    *ret_seed = state.seed_.val;
    *ret_offset = state.offset_.val;
    *ret_offset_intragraph = state.offset_intragraph_;
    *ret_captured = state.captured_ ? 1 : 0;
  });
}

#endif // USE_CUDA

} // extern "C"
