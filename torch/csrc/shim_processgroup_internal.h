#pragma once

// Internal (non-stable) definitions backing the stable c10d ProcessGroup shim.
// This header is included by both torch/csrc/shim_common.cpp (libtorch, where
// the collectives are implemented) and torch/csrc/stable/python/shim.cpp
// (libtorch_python, where a ProcessGroupHandle is created from a Python
// object). It must NOT be part of the public stable ABI surface: it exposes the
// c10::intrusive_ptr members that the opaque handles wrap.

#include <torch/csrc/stable/c/shim.h>

#ifdef USE_DISTRIBUTED

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

struct ProcessGroupOpaque {
  c10::intrusive_ptr<c10d::ProcessGroup> pg;
};

struct WorkOpaque {
  c10::intrusive_ptr<c10d::Work> work;
};

inline ProcessGroupHandle processgroup_pointer_to_handle(
    ProcessGroupOpaque* ptr) {
  return reinterpret_cast<ProcessGroupHandle>(ptr);
}

inline ProcessGroupOpaque* processgroup_handle_to_pointer(
    ProcessGroupHandle handle) {
  return reinterpret_cast<ProcessGroupOpaque*>(handle);
}

inline WorkHandle work_pointer_to_handle(WorkOpaque* ptr) {
  return reinterpret_cast<WorkHandle>(ptr);
}

inline WorkOpaque* work_handle_to_pointer(WorkHandle handle) {
  return reinterpret_cast<WorkOpaque*>(handle);
}

#endif // USE_DISTRIBUTED
