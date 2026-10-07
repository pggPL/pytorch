#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>
#include <memory>
#include <vector>

HIDDEN_NAMESPACE_BEGIN(torch, stable, c10d)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

enum class ReduceOp : int32_t {
  SUM = 0,
  AVG = 1,
  PRODUCT = 2,
  MIN = 3,
  MAX = 4,
  BAND = 5,
  BOR = 6,
  BXOR = 7
};

class Work {
 public:
  explicit Work(TorchWorkHandle work)
      : work_(work, [](TorchWorkHandle value) { torch_delete_work(value); }) {}

  bool wait(int64_t timeout_ms = 0) const {
    bool result = false;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_work_wait(work_.get(), timeout_ms, &result));
    return result;
  }

  bool is_completed() const {
    bool result = false;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_work_is_completed(work_.get(), &result));
    return result;
  }

 private:
  std::shared_ptr<TorchWorkOpaque> work_;
};

class ProcessGroup {
 public:
  explicit ProcessGroup(TorchProcessGroupHandle group)
      : group_(group, [](TorchProcessGroupHandle value) {
          torch_delete_process_group(value);
        }) {}

  static ProcessGroup from_pyobject(void* obj) {
    TorchProcessGroupHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_from_pyobject(obj, &result));
    return ProcessGroup(result);
  }

  int64_t rank() const {
    int64_t result = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_rank(group_.get(), &result));
    return result;
  }

  int64_t size() const {
    int64_t result = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_size(group_.get(), &result));
    return result;
  }

  std::string backend() const {
    StringHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_backend(group_.get(), &result));
    return detail::to<std::string>(detail::from(result));
  }

  void* nccl_comm(int32_t device_index) const {
    void* result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_process_group_get_nccl_comm(group_.get(), device_index, &result));
    return result;
  }

  Work allreduce(
      const std::vector<Tensor>& tensors,
      ReduceOp op = ReduceOp::SUM) const {
    auto handles = tensor_handles(tensors);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_allreduce(
        group_.get(),
        handles.data(),
        handles.size(),
        static_cast<int32_t>(op),
        &result));
    return Work(result);
  }

  Work allreduce_coalesced(
      const std::vector<Tensor>& tensors,
      ReduceOp op = ReduceOp::SUM) const {
    auto handles = tensor_handles(tensors);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_allreduce_coalesced(
        group_.get(),
        handles.data(),
        handles.size(),
        static_cast<int32_t>(op),
        &result));
    return Work(result);
  }

  Work broadcast(
      const std::vector<Tensor>& tensors,
      int64_t root_rank,
      int64_t root_tensor = 0) const {
    auto handles = tensor_handles(tensors);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_broadcast(
        group_.get(),
        handles.data(),
        handles.size(),
        root_rank,
        root_tensor,
        &result));
    return Work(result);
  }

  Work allgather(const Tensor& input, const std::vector<Tensor>& outputs)
      const {
    auto handles = tensor_handles(outputs);
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_allgather(
        group_.get(), input.get(), handles.data(), handles.size(), &result));
    return Work(result);
  }

  Work barrier(const std::vector<int64_t>& device_ids = {}) const {
    TorchWorkHandle result = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_process_group_barrier(
        group_.get(), device_ids.data(), device_ids.size(), &result));
    return Work(result);
  }

 private:
  static std::vector<AtenTensorHandle> tensor_handles(
      const std::vector<Tensor>& tensors) {
    std::vector<AtenTensorHandle> result;
    result.reserve(tensors.size());
    for (const auto& tensor : tensors)
      result.push_back(tensor.get());
    return result;
  }

  std::shared_ptr<TorchProcessGroupOpaque> group_;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable, c10d)
