---
myst:
  html_meta:
    description: Stable ABI process groups and collective work in PyTorch C++.
    keywords: PyTorch, C++, stable ABI, distributed, process group, collectives
---

# Process Groups and Collective Work

`torch/csrc/stable/c10d.h` provides owning wrappers for existing process groups
and collective operations in `torch::stable::c10d`. These APIs require PyTorch
2.16 or later.

## Process Groups

`ProcessGroup::from_pyobject(obj)` acquires an owning reference to an existing
Python ProcessGroup, preserving Python overrides. The conversion requires the
caller to hold the GIL and `libtorch_python` to be loaded at runtime. Other calls
do not require the caller to hold the GIL; callbacks and ownership release
acquire it as needed.

Copies of the stable wrapper share ownership. Release all handles before Python
interpreter shutdown. Group creation, registration, and destruction remain with
the caller's distributed setup.

## Collective Operations

The wrapper exposes rank, size, backend name, `allreduce`,
`allreduce_coalesced`, `broadcast`, single-input `allgather`, and `barrier`.
Reductions include SUM, AVG, PRODUCT, MIN, MAX, BAND, BOR, and BXOR; backend
support is unchanged. All ranks must issue matching collectives in matching
order. `allgather` takes one output per rank, and broadcast roots are ranks
within this group.

For example, given a Python ProcessGroup pointer `py_group` and a
`torch::stable::Tensor` named `tensor`, the following performs an in-place sum
across the group. The caller must hold the GIL for `from_pyobject`.

```cpp
#include <torch/csrc/stable/c10d.h>

auto group = torch::stable::c10d::ProcessGroup::from_pyobject(py_group);
auto work = group.allreduce({tensor}, torch::stable::c10d::ReduceOp::SUM);
work.wait();
```

On CUDA, subsequent use of `tensor` on the calling stream is ordered after the
collective. Use on another stream requires a dependency on that stream as well.
Keep `work` until the operation completes, as described below.

## Work Ownership and Synchronization

Each operation returns an owning `Work`, which retains the native operation,
group, and tensor objects. Copies share ownership. Keep it until the operation
completes. Dropping `Work` neither waits nor cancels.

`wait()` delegates to the backend: for CUDA it establishes dependencies on the
calling stream and need not block the CPU until GPU completion. Polling
`is_completed()` does not establish ordering on a different CUDA stream.
`wait(timeout_ms)` propagates backend errors and returns the backend's boolean
result; zero selects the backend default. A successful CUDA `wait()` alone does
not establish that it is safe to release externally managed resources.

Ownership does not protect against explicit group destruction or external
mutation/freeing of tensor storage. Consumers must retain external buffers and
follow their backend's stream and graph lifetime requirements.

## API Reference

```{doxygenclass} torch::stable::c10d::ProcessGroup
:members:
:undoc-members:
```

```{doxygenclass} torch::stable::c10d::Work
:members:
:undoc-members:
```

```{doxygenenum} torch::stable::c10d::ReduceOp
```

The C entry points are exported even without distributed support and report a
clear error in that configuration. This API does not expose native NCCL
communicators, symmetric-memory windows, or arbitrary TorchBind objects.
