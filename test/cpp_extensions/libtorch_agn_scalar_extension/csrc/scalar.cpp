#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

namespace {
using torch::stable::Device;
using torch::stable::Scalar;
using torch::stable::ScalarType;
using torch::stable::Tensor;

Tensor add(Tensor self, Tensor other, Scalar alpha) {
  return torch::stable::add(self, other, alpha);
}

Tensor arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Device device) {
  return torch::stable::arange(start, end, step, dtype, std::nullopt, device);
}

Scalar echo(Scalar value) {
  return value;
}

std::vector<Scalar> echo_list(std::vector<Scalar> values) {
  return values;
}

STABLE_TORCH_LIBRARY(stable_scalar_extension, m) {
  m.def("add(Tensor self, Tensor other, Scalar alpha=1) -> Tensor");
  m.def("arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Device device) -> Tensor");
  m.def("echo(Scalar value) -> Scalar");
  m.def("echo_list(Scalar[] values) -> Scalar[]");
}

STABLE_TORCH_LIBRARY_IMPL(stable_scalar_extension, CompositeExplicitAutograd, m) {
  m.impl("add", TORCH_BOX(&add));
  m.impl("arange", TORCH_BOX(&arange));
  m.impl("echo", TORCH_BOX(&echo));
  m.impl("echo_list", TORCH_BOX(&echo_list));
}
} // namespace
