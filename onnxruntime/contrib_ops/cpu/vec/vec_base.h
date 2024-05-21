#pragma once

#include <array>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <cmath>
#include <type_traits>
#include <climits>
#include <complex>


#include "contrib_ops/cpu/vec/math.h"
#include "contrib_ops/cpu/vec/intrinsics.h"

#if defined(_MSC_VER)
#define __FORCE_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define __FORCE_INLINE __attribute__((__always_inline__)) inline
#else
#define __FORCE_INLINE inline
#endif

#if defined(__clang__)
#define __sanitize_ignore_float_divide_by_zero__ \
  __attribute__((no_sanitize("float-divide-by-zero")))
#else
#define __sanitize_ignore_float_divide_by_zero__
#endif

// These macros helped us unify vec_base.h
#ifdef CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(64)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(64))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 64
#define int_vector __m512i
#else  // CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(32))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 32
#define int_vector __m256i
#endif  // CPU_CAPABILITY_AVX512

namespace onnxruntime::vec {
inline namespace CPU_CAPABILITY {
template <typename T>
struct is_complex : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

// onnxruntime::MLFloat16 and onnxruntime::BFloat16 should be treated as floating point
template <typename T>
struct is_floating_point : std::integral_constant<bool,
                                                  std::is_floating_point_v<T> ||
                                                      std::is_same_v<T, onnxruntime::MLFloat16> ||
                                                      std::is_same_v<T, onnxruntime::BFloat16>> {
};

template <typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

template <typename T>
struct is_reduced_floating_point : std::integral_constant<bool,
                                                          std::is_same_v<T, onnxruntime::MLFloat16> ||
                                                              std::is_same_v<T, onnxruntime::BFloat16>> {
};

template <typename T>
constexpr bool is_reduced_floating_point_v = is_reduced_floating_point<T>::value;

template <size_t n>
struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t)     \
  template <>                         \
  struct int_of_size<sizeof(int_t)> { \
    using type = int_t;               \
  }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

// NOTE: If you specialize on a type, you must define all operations!

// emulates Vectorized types
#if defined(__s390x__)
template <class T, class TEMP = void>
#else
template <class T>
#endif
struct Vectorized {
 private:
  __at_align__ T values[VECTOR_WIDTH / sizeof(T)];

 public:
  using value_type = T;
  using size_type = int;
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(T);
  }
  Vectorized() : values{static_cast<T>(0)} {}
  Vectorized(T val) {
    for (int i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  template <typename... Args,
            typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) : values{vals...} {
  }
  // This also implies const T& operator[](int idx) const
  inline operator const T*() const {
    return values;
  }
  // This also implies T& operator[](int idx)
  inline operator T*() {
    return values;
  }
  // Return the values as char* for type punning
  auto as_bytes() const -> const char* {
    return reinterpret_cast<const char*>(values);
  }
  template <int64_t mask_>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    int64_t mask = mask_;
    Vectorized vector;
    for (int i = 0; i < size(); i++) {
      if (mask & 0x01) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
      mask = mask >> 1;
    }
    return vector;
  }

  static Vectorized<T> blendv(const Vectorized<T>& a, const Vectorized<T>& b,
                              const Vectorized<T>& mask) {
    Vectorized vector;
    int_same_size_t<T> buffer[size()];
    mask.store(buffer);
    for (int i = 0; i < size(); i++) {
      if (buffer[i] & 0x01) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  template <typename step_t>  // step sometimes requires a higher precision type (e.g., T=int, step_t=double)
  static Vectorized<T> arange(T base = static_cast<T>(0), step_t step = static_cast<step_t>(1)) {
    Vectorized vector;
    for (int i = 0; i < size(); i++) {
      vector.values[i] = base + i * step;
    }
    return vector;
  }

  static Vectorized<T> set(const Vectorized<T>& a, const Vectorized<T>& b, int64_t count = size()) {
    Vectorized vector;
    for (int i = 0; i < size(); i++) {
      if (i < count) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  static Vectorized<T> loadu(const void* ptr) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, VECTOR_WIDTH);
    return vector;
  }
  static Vectorized<T> loadu(const void* ptr, int64_t count) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, count * sizeof(T));
    return vector;
  }
  static Vectorized<T> loadu_one_fourth(const void* ptr) {
    static_assert(std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>, "For byte types only");
    return Vectorized::loadu(ptr, 8);
  }

  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (values[i] == static_cast<T>(0)) {
        mask |= (1 << i);
      }
    }
    return mask;
  }

  Vectorized<T> isnan() const {
    Vectorized<T> vector;
    for (int64_t i = 0; i != size(); i++) {
      if (_isnan(values[i])) {
        std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
      }
    }
    return vector;
  }

  bool has_inf_nan() const {
    for (int64_t i = 0; i != size(); i++) {
      if (_isnan(values[i]) || _isinf(values[i])) {
        return true;
      }
    }
    return false;
  }

  Vectorized<T> map(T (*const f)(T)) const {
    Vectorized<T> ret;
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }

  Vectorized<T> map(T (*const f)(const T&)) const {
    Vectorized<T> ret;
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }

  template <typename other_t_abs = T,
            typename std::enable_if_t<!is_floating_point_v<other_t_abs> && !is_complex<other_t_abs>::value, int> = 0>
  Vectorized<T> abs() const {
    // other_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_abs, T>, "other_t_abs must be T");
    return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
  }

  template <typename float_t_abs = T,
            typename std::enable_if_t<is_floating_point_v<float_t_abs>, int> = 0>
  Vectorized<T> abs() const {
    // float_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<float_t_abs, T>, "float_t_abs must be T");
    // Specifically deal with floating-point because the generic code above won't handle -0.0 (which should result in
    // 0.0) properly.
    return map([](T x) -> T { return std::abs(x); });
  }

  template <typename complex_t_abs = T,
            typename std::enable_if_t<is_complex<complex_t_abs>::value, int> = 0>
  Vectorized<T> abs() const {
    // complex_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<complex_t_abs, T>, "complex_t_abs must be T");
    // Specifically map() does not perform the type conversion needed by abs.
    return map([](T x) { return static_cast<T>(std::abs(x)); });
  }


  Vectorized<T> erf() const {
    return map(std::erf);
  }
  Vectorized<T> erfc() const {
    return map(std::erfc);
  }
  Vectorized<T> exp() const {
    return map(std::exp);
  }
  Vectorized<T> expm1() const {
    return map(std::expm1);
  }
  Vectorized<T> exp_u20() const {
    return map(std::exp);
  }
  Vectorized<T> frac() const {
    return *this - this->trunc();
  }
  template <
      typename U = T,
      typename std::enable_if_t<is_floating_point_v<U>, int> = 0>
  Vectorized<T> fmod(const Vectorized<T>& q) const {
    // U is for SFINAE purposes only. Make sure it is not changed.
    static_assert(std::is_same_v<U, T>, "U must be T");
    Vectorized<T> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::fmod(values[i], q[i]);
    }
    return ret;
  }
  Vectorized<T> log() const {
    return map(std::log);
  }
  Vectorized<T> log10() const {
    return map(std::log10);
  }
  Vectorized<T> log1p() const {
    return map(std::log1p);
  }

  template <typename other_t_log2 = T,
            typename std::enable_if_t<!is_complex<other_t_log2>::value, int> = 0>
  Vectorized<T> log2() const {
    // other_t_log2 is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<other_t_log2, T>, "other_t_log2 must be T");
    return map(std::log2);
  }
  template <typename complex_t_log2 = T,
            typename std::enable_if_t<is_complex<complex_t_log2>::value, int> = 0>
  Vectorized<T> log2() const {
    // complex_t_log2 is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same_v<complex_t_log2, T>, "complex_t_log2 must be T");
    const T log_2 = T(std::log(2.0));
    return Vectorized(map(std::log)) / Vectorized(log_2);
  }
  Vectorized<T> ceil() const {
    return map(std::ceil);
  }
  Vectorized<T> cos() const {
    return map(std::cos);
  }
  Vectorized<T> cosh() const {
    return map(std::cosh);
  }
  Vectorized<T> floor() const {
    return map(std::floor);
  }

  Vectorized<T> neg() const {
    // NB: the trailing return type is needed because we need to coerce the
    // return value back to T in the case of unary operator- incuring a
    // promotion
    return map([](T x) -> T { return -x; });
  }
  Vectorized<T> nextafter(const Vectorized<T>& b) const {
    Vectorized<T> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::nextafter(values[i], b[i]);
    }
    return ret;
  }
  //   Vectorized<T> round() const {
  //     // We do not use std::round because we would like to round midway numbers to the nearest even integer.
  //     return map(at::native::round_impl);
  //   }
  Vectorized<T> sin() const {
    return map(std::sin);
  }
  Vectorized<T> sinh() const {
    return map(std::sinh);
  }
  Vectorized<T> tan() const {
    return map(std::tan);
  }
  Vectorized<T> tanh() const {
    return map(std::tanh);
  }
  Vectorized<T> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<T> reciprocal() const {
    return map([](T x) { return (T)(1) / x; });
  }
  Vectorized<T> rsqrt() const {
    return map([](T x) { return (T)1 / std::sqrt(x); });
  }
  Vectorized<T> pow(const Vectorized<T>& exp) const {
    Vectorized<T> ret;
    for (int i = 0; i < size(); i++) {
      ret[i] = std::pow(values[i], exp[i]);
    }
    return ret;
  }

 private:
  template <typename Op>
  inline Vectorized<T> binary_pred(const Vectorized<T>& other, Op op) const {
    // All bits are set to 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (int64_t i = 0; i != size(); i++) {
      if (op(values[i], other.values[i])) {
        std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
      }
    }
    return vector;
  }

 public:
  Vectorized<T> operator==(const Vectorized<T>& other) const { return binary_pred(other, std::equal_to<T>()); }
  Vectorized<T> operator!=(const Vectorized<T>& other) const { return binary_pred(other, std::not_equal_to<T>()); }
  Vectorized<T> operator>=(const Vectorized<T>& other) const { return binary_pred(other, std::greater_equal<T>()); }
  Vectorized<T> operator<=(const Vectorized<T>& other) const { return binary_pred(other, std::less_equal<T>()); }
  Vectorized<T> operator>(const Vectorized<T>& other) const { return binary_pred(other, std::greater<T>()); }
  Vectorized<T> operator<(const Vectorized<T>& other) const { return binary_pred(other, std::less<T>()); }

 private:
  template <typename Op>
  inline Vectorized<T> binary_pred_bool(const Vectorized<T>& other, Op op) const {
    // 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (int i = 0; i != size(); ++i) {
      vector[i] = static_cast<T>(op(values[i], other.values[i]));
    }
    return vector;
  }

 public:
  Vectorized<T> eq(const Vectorized<T>& other) const { return binary_pred_bool(other, std::equal_to<T>()); }
  Vectorized<T> ne(const Vectorized<T>& other) const { return binary_pred_bool(other, std::not_equal_to<T>()); }
  Vectorized<T> gt(const Vectorized<T>& other) const { return binary_pred_bool(other, std::greater<T>()); }
  Vectorized<T> ge(const Vectorized<T>& other) const { return binary_pred_bool(other, std::greater_equal<T>()); }
  Vectorized<T> lt(const Vectorized<T>& other) const { return binary_pred_bool(other, std::less<T>()); }
  Vectorized<T> le(const Vectorized<T>& other) const { return binary_pred_bool(other, std::less_equal<T>()); }
};

template <class T>
Vectorized<T> inline operator+(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator-(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator*(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator/(const Vectorized<T>& a, const Vectorized<T>& b)
    __sanitize_ignore_float_divide_by_zero__ {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <class T,
          typename std::enable_if_t<!is_floating_point_v<T>, int> = 0>
Vectorized<T> inline operator%(const Vectorized<T>& a, const Vectorized<T>& b) __sanitize_ignore_float_divide_by_zero__ {
  return a - a / b * b;
}

template <class T>
Vectorized<T> inline operator||(
    const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] || b[i];
  }
  return c;
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <class T,
          typename std::enable_if_t<!is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T,
          typename std::enable_if_t<is_complex<T>::value, int> = 0>
Vectorized<T> inline maximum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (std::abs(a[i]) > std::abs(b[i])) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <class T,
          typename std::enable_if_t<!is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T,
          typename std::enable_if_t<is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (std::abs(a[i]) < std::abs(b[i])) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T,
          typename std::enable_if_t<!is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp(const Vectorized<T>& a, const Vectorized<T>& min_vec, const Vectorized<T>& max_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = std::min(std::max(a[i], min_vec[i]), max_vec[i]);
  }
  return c;
}

template <class T,
          typename std::enable_if_t<!is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_max(const Vectorized<T>& a, const Vectorized<T>& max_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] > max_vec[i] ? max_vec[i] : a[i];
  }
  return c;
}

template <class T,
          typename std::enable_if_t<!is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_min(const Vectorized<T>& a, const Vectorized<T>& min_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline operator<<(const Vectorized<T>& a, const Vectorized<T>& b) {
  constexpr T max_shift = sizeof(T) * CHAR_BIT;
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
      c[i] = 0;
    } else {
      c[i] = static_cast<std::make_unsigned_t<T>>(a[i]) << shift;
    }
  }
  return c;
}

template <class T>
Vectorized<T> inline operator>>(const Vectorized<T>& a, const Vectorized<T>& b) {
  // right shift value to retain sign bit for signed and no bits for unsigned
  constexpr T max_shift = sizeof(T) * CHAR_BIT - std::is_signed_v<T>;
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
      c[i] = a[i] >> max_shift;
    } else {
      c[i] = a[i] >> shift;
    }
  }
  return c;
}

template <typename T>
inline Vectorized<T>& operator+=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a + b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator-=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a - b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator/=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a / b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator%=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a % b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator*=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a * b;
  return a;
}

template <typename T>
inline Vectorized<T>& operator<<=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a << b;
  return a;
}

template <typename T>
inline Vectorized<T>& operator>>=(Vectorized<T>& a, const Vectorized<T>& b) {
  a = a >> b;
  return a;
}

template <typename T>
inline Vectorized<T> fmadd(const Vectorized<T>& a, const Vectorized<T>& b, const Vectorized<T>& c) {
  return a * b + c;
}

template <typename T>
inline Vectorized<T> fmsub(const Vectorized<T>& a, const Vectorized<T>& b, const Vectorized<T>& c) {
  return a * b - c;
}

template <int64_t scale = 1, typename T = void>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>>
inline gather(T const* base_addr, const Vectorized<int_same_size_t<T>>& vindex) {
  static constexpr int size = Vectorized<T>::size();
  int_same_size_t<T> index_arr[size];
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (int i = 0; i < size; i++) {
    buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
  }
  return Vectorized<T>::loadu(static_cast<void*>(buffer));
}

// Cast a given vector to another type without changing the bits representation.
// So a Vectorized<double> of 512 bits containing all ones can be cast to a
// Vectorized<int64_t> of 512 bits containing all ones (i.e., eight negative 1s).
// A Vec<double> of 256 bits containing all ones can be cast to a
// Vec<int64_t> of 256 bits containing all ones (i.e., four negative 1s).
// There is a struct here because we don't have static_if and I can't
// partially specialize a templated function.
template <typename dst_t, typename src_t>
struct CastImpl {
  static inline Vectorized<dst_t> apply(const Vectorized<src_t>& src) {
    src_t src_arr[Vectorized<src_t>::size()];
    src.store(static_cast<void*>(src_arr));
    return Vectorized<dst_t>::loadu(static_cast<const void*>(src_arr));
  }
};

template <typename scalar_t>
struct CastImpl<scalar_t, scalar_t> {
  static inline Vectorized<scalar_t> apply(const Vectorized<scalar_t>& src) {
    return src;
  }
};

template <typename dst_t, typename src_t>
inline Vectorized<dst_t> cast(const Vectorized<src_t>& src) {
  return CastImpl<dst_t, src_t>::apply(src);
}

template <typename T>
inline Vectorized<T> flip(const Vectorized<T>& data) {
  static constexpr int size = Vectorized<T>::size();
  T output[size];
  T buffer[size];
  data.store(static_cast<void*>(buffer));
  for (int i = 0; i < size; i++) {
    output[i] = buffer[size - i - 1];
  }
  return Vectorized<T>::loadu(static_cast<void*>(output));
}

// Transpose the `src` buffer of type `T` and size (M,N) into the `dst` buffer. `ld_src` is the leading
// dimension of `src` and `ld_dst` is the leading dimension of `dst`.
template <typename T, int M, int N>
inline void transpose_mxn(const T* src, int64_t ld_src, T* dst, int64_t ld_dst) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}

template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

}  // namespace CPU_CAPABILITY

template <typename T>
inline void _store(T* dst, vec::Vectorized<T> src) {
  src.store(dst);
}

}  // namespace onnxruntime::vec
