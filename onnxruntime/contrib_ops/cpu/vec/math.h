#pragma once

#include <cmath>
#include <cstring>
#include <type_traits>

namespace onnxruntime::vec {

template <typename T>
inline constexpr T pi() {
  return static_cast<T>(3.141592653589793238462643383279502);
}

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
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
  return ::exp(x);
}

template <typename T>
 inline T log(T x) {
  return ::log(x);
}

template <typename T>
 inline T tan(T x) {
  return ::tan(x);
}

}
