#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cstdint>
#include <memory>
#include <vector>

// Header-only, stable-ABI wrappers around the c10d ProcessGroup collectives
// that a stable extension needs. These build entirely on the torch_processgroup_*
// / torch_work_* C shims and never include non-stable <torch/csrc/distributed/*>
// headers, so they compile under TORCH_TARGET_VERSION.

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

// Mirrors c10d::ReduceOp::RedOpType (ordinals are part of the ABI). PREMUL_SUM
// is intentionally omitted (it requires a scaling factor).
enum class ReduceOp : int32_t {
  SUM = 0,
  AVG = 1,
  PRODUCT = 2,
  MIN = 3,
  MAX = 4,
  BAND = 5,
  BOR = 6,
  BXOR = 7,
};

// RAII wrapper over a WorkHandle representing an in-flight collective.
class Work {
 public:
  explicit Work() = delete;

  // Steals ownership of the WorkHandle.
  explicit Work(WorkHandle handle)
      : work_(handle, [](WorkHandle w) {
          STABLE_TORCH_ERROR_CODE_CHECK(torch_delete_work(w));
        }) {}

  // Blocks until the collective completes.
  bool wait() {
    bool completed = false;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_work_wait(work_.get(), &completed));
    return completed;
  }

 private:
  std::shared_ptr<WorkOpaque> work_;
};

// RAII wrapper over a ProcessGroupHandle. Construct one from a Python
// torch.distributed.ProcessGroup via
// torch::stable::processgroup_from_pyobject (torch/csrc/stable/python/interop.h).
class ProcessGroup {
 public:
  explicit ProcessGroup() = delete;

  // Steals ownership of the ProcessGroupHandle.
  explicit ProcessGroup(ProcessGroupHandle handle)
      : pg_(handle, [](ProcessGroupHandle p) {
          STABLE_TORCH_ERROR_CODE_CHECK(torch_delete_processgroup(p));
        }) {}

  int32_t rank() const {
    int32_t r = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_processgroup_rank(pg_.get(), &r));
    return r;
  }

  int32_t size() const {
    int32_t s = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_processgroup_size(pg_.get(), &s));
    return s;
  }

  bool backend_is_nccl() const {
    bool b = false;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_processgroup_backend_is_nccl(pg_.get(), &b));
    return b;
  }

  Work allreduce(std::vector<Tensor>& tensors, ReduceOp op = ReduceOp::SUM) {
    auto handles = to_handles(tensors);
    WorkHandle work = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_processgroup_allreduce(
        pg_.get(),
        handles.data(),
        handles.size(),
        static_cast<int32_t>(op),
        &work));
    return Work(work);
  }

  Work allreduce_coalesced(
      std::vector<Tensor>& tensors,
      ReduceOp op = ReduceOp::SUM) {
    auto handles = to_handles(tensors);
    WorkHandle work = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_processgroup_allreduce_coalesced(
        pg_.get(),
        handles.data(),
        handles.size(),
        static_cast<int32_t>(op),
        &work));
    return Work(work);
  }

  Work broadcast(std::vector<Tensor>& tensors, int64_t root_rank = 0) {
    auto handles = to_handles(tensors);
    WorkHandle work = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_processgroup_broadcast(
        pg_.get(), handles.data(), handles.size(), root_rank, &work));
    return Work(work);
  }

  // Gathers `input` from every rank into `output` (one tensor per rank).
  Work allgather(std::vector<Tensor>& output, Tensor& input) {
    auto handles = to_handles(output);
    WorkHandle work = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_processgroup_allgather(
        pg_.get(), handles.data(), handles.size(), input.get(), &work));
    return Work(work);
  }

  Work barrier() {
    WorkHandle work = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_processgroup_barrier(pg_.get(), &work));
    return Work(work);
  }

  ProcessGroupHandle get() const {
    return pg_.get();
  }

 private:
  static std::vector<AtenTensorHandle> to_handles(
      std::vector<Tensor>& tensors) {
    std::vector<AtenTensorHandle> handles;
    handles.reserve(tensors.size());
    for (auto& t : tensors) {
      handles.push_back(t.get());
    }
    return handles;
  }

  std::shared_ptr<ProcessGroupOpaque> pg_;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

HIDDEN_NAMESPACE_END(torch, stable)
