#pragma once

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <complex>
#include <memory>
#include <type_traits>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

// The torch::stable::Scalar class is a header-only C++ wrapper around the C shim
// Scalar APIs, modeled after at::Scalar. It exists because at::Scalar itself is
// not header-only, so extensions cannot use it directly.
//
// A Scalar is a tagged variant (bool / int / double / complex<double>). It is
// backed by an owning ScalarHandle managed through a shared_ptr, mirroring
// torch::stable::Generator/Tensor: copies share the same underlying handle and
// the handle is freed with torch_delete_scalar once the last copy is destroyed.

/**
 * @brief An ABI stable wrapper around a PyTorch Scalar.
 *
 * Modeled after at::Scalar. Implicitly constructible from the integral,
 * floating point, boolean, and complex<double> C++ types.
 *
 * Minimum compatible version: PyTorch 2.14.
 */
class Scalar {
 private:
  std::shared_ptr<ScalarOpaque> scalar_;

  static std::shared_ptr<ScalarOpaque> wrap(ScalarHandle handle) {
    return std::shared_ptr<ScalarOpaque>(handle, [](ScalarHandle handle) {
      STABLE_TORCH_ERROR_CODE_CHECK(torch_delete_scalar(handle));
    });
  }

 public:
  /**
   * @brief Constructs a Scalar from an existing ScalarHandle.
   *
   * Steals ownership of the provided ScalarHandle.
   */
  explicit Scalar(ScalarHandle scalar) : scalar_(wrap(scalar)) {}

  // Implicit constructors from the primitive scalar types. Integral (non-bool)
  // and floating point are templated with SFINAE so that e.g. a plain `int`
  // literal unambiguously selects the integer constructor.
  template <
      typename T,
      std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> =
          0>
  /* implicit */ Scalar(T value) {
    ScalarHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_new_scalar_int(static_cast<int64_t>(value), &handle));
    scalar_ = wrap(handle);
  }

  template <
      typename T,
      std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  /* implicit */ Scalar(T value) {
    ScalarHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_new_scalar_double(static_cast<double>(value), &handle));
    scalar_ = wrap(handle);
  }

  /* implicit */ Scalar(bool value) {
    ScalarHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_new_scalar_bool(value, &handle));
    scalar_ = wrap(handle);
  }

  /* implicit */ Scalar(std::complex<double> value) {
    ScalarHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_new_scalar_complex_double(
        value.real(), value.imag(), &handle));
    scalar_ = wrap(handle);
  }

  // Copy and move can be default since the underlying handle is a shared_ptr.
  /// \private
  Scalar(const Scalar& other) = default;
  /// \private
  Scalar(Scalar&& other) noexcept = default;
  /// \private
  Scalar& operator=(const Scalar& other) = default;
  /// \private
  Scalar& operator=(Scalar&& other) noexcept = default;
  /// \private
  ~Scalar() = default;

  /**
   * @brief Returns a borrowed reference to the underlying ScalarHandle.
   */
  ScalarHandle get() const {
    return scalar_.get();
  }

  bool isInt() const {
    bool ret = false;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_is_int(get(), &ret));
    return ret;
  }

  bool isDouble() const {
    bool ret = false;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_is_double(get(), &ret));
    return ret;
  }

  bool isBool() const {
    bool ret = false;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_is_bool(get(), &ret));
    return ret;
  }

  bool isComplex() const {
    bool ret = false;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_is_complex_double(get(), &ret));
    return ret;
  }

  int64_t toInt() const {
    int64_t ret = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_to_int(get(), &ret));
    return ret;
  }

  double toDouble() const {
    double ret = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_to_double(get(), &ret));
    return ret;
  }

  bool toBool() const {
    bool ret = false;
    STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_to_bool(get(), &ret));
    return ret;
  }

  std::complex<double> toComplexDouble() const {
    double real = 0;
    double imag = 0;
    STABLE_TORCH_ERROR_CODE_CHECK(
        torch_scalar_to_complex_double(get(), &real, &imag));
    return std::complex<double>(real, imag);
  }
};

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_14_0

HIDDEN_NAMESPACE_END(torch, stable)
