#include <ATen/Context.h>
#include <gtest/gtest.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/generator.h>
#include <torch/csrc/stable/tensor.h>

TEST(TorchStableGenerator, DefaultSharesState) {
  auto stable = torch::stable::get_default_generator(
      torch::stable::Device(torch::stable::DeviceType::CPU));
  auto* native =
      torch::aot_inductor::generator_handle_to_generator_pointer(stable.get());
  EXPECT_EQ(*native, at::globalContext().defaultGenerator(at::kCPU));
  auto copy = stable;
  EXPECT_EQ(copy.get(), stable.get());
  EXPECT_EQ(copy.device().type(), torch::stable::DeviceType::CPU);
}

TEST(TorchStableGenerator, PhiloxStateOwnsTensors) {
  if (!at::hasCUDA() || at::getNumGPUs() == 0) {
    GTEST_SKIP() << "requires CUDA";
  }
  const auto [first, second] = [] {
    auto gen = torch::stable::get_default_generator(
        torch::stable::Device(torch::stable::DeviceType::CUDA, 0));
    auto first = gen.philox_state(5);
    auto second = gen.philox_state(4);
    return std::make_pair(std::move(first), std::move(second));
  }();
  for (const auto* state : {&first, &second}) {
    const auto& [seed, offset, intra] = *state;
    for (const auto* tensor : {&seed, &offset, &intra}) {
      ASSERT_EQ(tensor->device().type(), torch::stable::DeviceType::CPU);
      ASSERT_EQ(tensor->scalar_type(), torch::stable::ScalarType::Long);
      ASSERT_EQ(tensor->numel(), 1);
    }
    EXPECT_EQ(*intra.const_data_ptr<int64_t>(), 0);
  }
  EXPECT_EQ(
      *std::get<0>(first).const_data_ptr<int64_t>(),
      *std::get<0>(second).const_data_ptr<int64_t>());
  EXPECT_EQ(
      *std::get<1>(second).const_data_ptr<int64_t>(),
      *std::get<1>(first).const_data_ptr<int64_t>() + 8);
}

TEST(TorchStableGenerator, UnsupportedBackendAndInvalidArguments) {
  auto gen = torch::stable::get_default_generator(
      torch::stable::Device(torch::stable::DeviceType::CPU));
  AtenTensorHandle seed = nullptr, offset = nullptr, intragraph = nullptr;
  EXPECT_EQ(
      torch_generator_philox_state(gen.get(), 4, &seed, &offset, &intragraph),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(seed, nullptr);
  EXPECT_EQ(offset, nullptr);
  EXPECT_EQ(intragraph, nullptr);
  EXPECT_EQ(
      torch_generator_philox_state(gen.get(), 4, &seed, &seed, &intragraph),
      AOTI_TORCH_FAILURE);
  EXPECT_EQ(
      torch_generator_philox_state(nullptr, 4, &seed, &offset, &intragraph),
      AOTI_TORCH_FAILURE);
  AtenGeneratorHandle out = nullptr;
  EXPECT_EQ(torch_get_default_generator(0, 128, &out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(torch_get_default_generator(256, -1, &out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(torch_get_default_generator(0, -1, nullptr), AOTI_TORCH_FAILURE);
}

TEST(TorchStableGenerator, PythonBridgeRequiresPython) {
  int dummy = 0;
  AtenGeneratorHandle out = nullptr;
  EXPECT_EQ(torch_generator_from_pyobject(&dummy, &out), AOTI_TORCH_FAILURE);
  EXPECT_EQ(out, nullptr);
}
