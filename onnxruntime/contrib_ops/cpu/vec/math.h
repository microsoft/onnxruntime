#pragma once

#include <cmath>
#include <cstring>
#include <type_traits>

namespace onnxruntime::vec {

template <typename T>
inline constexpr T pi() {
  return static_cast<T>(3.141592653589793238462643383279502);
}

#define CENTRAL_RANGE 0.7

template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_erfinv(T y) {
/* Function to calculate inverse error function.  Rational approximation
is used to generate an initial approximation, which is then improved to
full accuracy by two steps of Newton's method.  Code is a direct
translation of the erfinv m file in matlab version 2.0.
Author:  Gary L. Pavlis, Indiana University
Date:  February 1996
*/
  T x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */
  T a[4] = {  T(0.886226899), T(-1.645349621),  T(0.914624893), T(-0.140543331) };
  T b[4] = { T(-2.118377725),  T(1.442710462), T(-0.329097515),  T(0.012229801) };
  T c[4] = { T(-1.970840454), T(-1.624906493),  T(3.429567803),  T(1.641345311) };
  T d[2] = {  T(3.543889200),  T(1.637067800) };
  T y_abs = std::abs(y);
  if(y_abs > 1.0) return std::numeric_limits<T>::quiet_NaN();
#ifdef _WIN32
  // error C2039: '_copysign': is not a member of 'std'
  if(y_abs == 1.0) return copysign(std::numeric_limits<T>::infinity(), y);
#else
  if(y_abs == 1.0) return std::copysign(std::numeric_limits<T>::infinity(), y);
#endif
  if(y_abs <= static_cast<T>(CENTRAL_RANGE)) {
    z = y * y;
    num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
    dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0]) * z + static_cast<T>(1.0));
    x = y * num / dem;
  }
  else{
    z = std::sqrt(-std::log((static_cast<T>(1.0)-y_abs)/static_cast<T>(2.0)));
    num = ((c[3]*z + c[2])*z + c[1]) * z + c[0];
    dem = (d[1]*z + d[0])*z + static_cast<T>(1.0);
#ifdef _WIN32
    // error C2039: '_copysign': is not a member of 'std'
    x = copysign(num, y) / dem;
#else
    x = std::copysign(num, y) / dem;
#endif
  }
  /* Two steps of Newton-Raphson correction */
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(pi<double>())))*std::exp(-x*x));
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(pi<double>())))*std::exp(-x*x));

  return(x);
}

#undef CENTRAL_RANGE

template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible_v<To>,
      "This implementation additionally requires "
      "destination type to be trivially constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}


// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline bool _isnan(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline bool _isnan(T val) {
  return std::isnan(val);
}

// std::isinf isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline  bool _isinf(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline  bool _isinf(T val) {
  return std::isinf(val);
}

template <typename T>
 inline T exp(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
  return ::exp(x);
}

template <>
 inline double exp<double>(double x) {
  return ::exp(x);
}

template <typename T>
 inline T log(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
  return ::log(x);
}

template <>
 inline double log<double>(double x) {
  return ::log(x);
}

template <typename T>
 inline T log1p(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
  return ::log1p(x);
}

template <>
 inline double log1p<double>(double x) {
  return ::log1p(x);
}

template <typename T>
 inline T tan(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
  return ::tan(x);
}

template <>
 inline double tan<double>(double x) {
  return ::tan(x);
}




}
