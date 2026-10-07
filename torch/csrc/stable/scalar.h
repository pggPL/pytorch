#pragma once

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <variant>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

// A value type for concrete dispatcher Scalar (NumberType) arguments.
// The variant is local to the extension; it never crosses the ABI boundary.
class Scalar {
 public:
  using Value = std::variant<bool, int64_t, double, std::complex<double>>;

  Scalar() : value_(int64_t{0}) {}
  Scalar(bool value) : value_(value) {}
  Scalar(double value) : value_(value) {}
  Scalar(std::complex<double> value) : value_(value) {}

  template <
      typename T,
      std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> =
          0>
  Scalar(T value) : value_(int64_t{0}) {
    if constexpr (std::is_unsigned_v<T>) {
      STD_TORCH_CHECK(
          value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
          "stable Scalar integer must fit in int64");
    }
    value_ = static_cast<int64_t>(value);
  }

  const Value& value() const {
    return value_;
  }

 private:
  Value value_;
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
