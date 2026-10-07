#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Scalar;

namespace {
Scalar echo(Scalar value) {
  return value;
}
std::optional<Scalar> optional_echo(std::optional<Scalar> value) {
  return value;
}
std::vector<Scalar> list_echo(std::vector<Scalar> value) {
  return value;
}

STABLE_TORCH_LIBRARY(stable_scalar_test, m) {
  m.def("echo(Scalar value) -> Scalar");
  m.def("optional_echo(Scalar? value) -> Scalar?");
  m.def("list_echo(Scalar[] value) -> Scalar[]");
}
STABLE_TORCH_LIBRARY_IMPL(stable_scalar_test, CompositeExplicitAutograd, m) {
  m.impl("echo", TORCH_BOX(&echo));
  m.impl("optional_echo", TORCH_BOX(&optional_echo));
  m.impl("list_echo", TORCH_BOX(&list_echo));
}
} // namespace

TEST(TorchStableScalar, RoundTripPreservesTypeAndBits) {
  std::vector<Scalar> values{
      true,
      false,
      int64_t{9007199254740993LL},
      std::numeric_limits<int64_t>::min(),
      1.25,
      std::complex<double>(1.5, -2.5)};
  for (const auto& value : values) {
    auto result =
        torch::stable::detail::to<Scalar>(torch::stable::detail::from(value));
    EXPECT_EQ(result.value(), value.value());
  }
  EXPECT_THROW(
      Scalar(std::numeric_limits<uint64_t>::max()), std::runtime_error);
  auto malformed =
      torch::stable::detail::from(std::vector<StableIValue>{99, 0, 0});
  EXPECT_THROW(
      torch::stable::detail::to<Scalar>(malformed), std::runtime_error);
}

TEST(TorchStableScalar, BoxedNumberOptionalAndList) {
  auto& dispatcher = c10::Dispatcher::singleton();
  auto op = dispatcher.findSchemaOrThrow("stable_scalar_test::echo", "");
  for (const auto& value : std::vector<c10::IValue>{
           true,
           int64_t{9007199254740993LL},
           1.25,
           c10::complex<double>(1.5, -2.5)}) {
    c10::Stack stack{value};
    op.callBoxed(&stack);
    ASSERT_EQ(stack.size(), 1);
    EXPECT_EQ(stack[0].tagKind(), value.tagKind());
    EXPECT_TRUE(stack[0].equals(value).toBool());
  }
  auto optional =
      dispatcher.findSchemaOrThrow("stable_scalar_test::optional_echo", "");
  c10::Stack absent{c10::IValue()};
  optional.callBoxed(&absent);
  EXPECT_TRUE(absent[0].isNone());
  c10::Stack present{int64_t{9007199254740993LL}};
  optional.callBoxed(&present);
  EXPECT_EQ(present[0].toInt(), 9007199254740993LL);
  auto list_op =
      dispatcher.findSchemaOrThrow("stable_scalar_test::list_echo", "");
  c10::impl::GenericList values(c10::NumberType::get());
  values.push_back(true);
  values.push_back(int64_t{9007199254740993LL});
  values.push_back(c10::complex<double>(1, 2));
  c10::Stack list_stack{values};
  list_op.callBoxed(&list_stack);
  EXPECT_TRUE(list_stack[0].equals(c10::IValue(values)).toBool());
}

TEST(TorchStableScalar, AddAndArangeUseScalarSchema) {
  using torch::stable::detail::to;
  auto base = at::ones({2}, at::kLong);
  torch::stable::Tensor tensor(
      torch::aot_inductor::new_tensor_handle(at::Tensor(base)));
  auto added =
      torch::stable::add(tensor, tensor, Scalar(int64_t{9007199254740993LL}));
  auto* actual =
      torch::aot_inductor::tensor_handle_to_tensor_pointer(added.get());
  EXPECT_TRUE(actual->equal(base + 9007199254740993LL));
  auto range = torch::stable::arange(5, -1, -2);
  auto* range_tensor =
      torch::aot_inductor::tensor_handle_to_tensor_pointer(range.get());
  EXPECT_TRUE(range_tensor->equal(at::arange(5, -1, -2)));
  EXPECT_EQ(torch::stable::arange(0, 0).numel(), 0);
  EXPECT_THROW(torch::stable::arange(0, 5, 0), std::runtime_error);
}

TEST(TorchStableScalar, RejectsLegacyWireFormatBeforeDereferencing) {
  StableIValue value = 0;
  EXPECT_EQ(
      torch_call_dispatcher(
          "stable_scalar_test::echo", "", &value, TORCH_VERSION_2_14_0),
      AOTI_TORCH_FAILURE);
}

TEST(TorchStableScalar, InvalidPayloadAndAllocationErrors) {
  for (const auto& words :
       std::vector<std::vector<StableIValue>>{{}, {1, 0}, {0, 2, 0}}) {
    auto wire = torch::stable::detail::from(words);
    EXPECT_THROW(torch::stable::detail::to<Scalar>(wire), std::runtime_error);
  }
  StableListHandle list = nullptr;
  EXPECT_EQ(
      torch_new_list_reserve_size(std::numeric_limits<size_t>::max(), &list),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(list, nullptr);
  EXPECT_EQ(torch_new_list_reserve_size(3, nullptr), AOTI_TORCH_FAILURE);
}
