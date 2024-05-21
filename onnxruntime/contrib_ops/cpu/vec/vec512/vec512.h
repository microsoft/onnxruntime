#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <contrib_ops/cpu/vec/intrinsics.h>

#include <contrib_ops/cpu/vec/vec_base.h>
#include <contrib_ops/cpu/vec/vec512/vec512_float.h>
// #include <contrib_ops/cpu/vec/vec512/vec512_double.h>
// #include <contrib_ops/cpu/vec/vec512/vec512_int.h>
//#include <contrib_ops/cpu/vec/vec512/vec512_mask.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>

namespace onnxruntime {
namespace vec {

// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// inline std::ostream& operator<<(std::ostream& stream, const c10::qint32& val) {
//   stream << val.val_;
//   return stream;
// }
// inline std::ostream& operator<<(std::ostream& stream, const c10::qint8& val) {
//   stream << static_cast<int>(val.val_);
//   return stream;
// }
// inline std::ostream& operator<<(std::ostream& stream, const c10::quint8& val) {
//   stream << static_cast<unsigned int>(val.val_);
//   return stream;
// }

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vectorized<T>& vec) {
  T buf[Vectorized<T>::size()];
  vec.store(buf);
  stream << "vec[";
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    if (i != 0) {
      stream << ", ";
    }
    stream << buf[i];
  }
  stream << "]";
  return stream;
}


#if defined(CPU_CAPABILITY_AVX512)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX512) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// template<>
// inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
//   return _mm512_castpd_ps(src);
// }

template<>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm512_castps_pd(src);
}

template<>
inline Vectorized<float> cast<float, int32_t>(const Vectorized<int32_t>& src) {
  return _mm512_castsi512_ps(src);
}

template<>
inline Vectorized<double> cast<double, int64_t>(const Vectorized<int64_t>& src) {
  return _mm512_castsi512_pd(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef _MSC_VER
// MSVC is not working well on complex function overload.
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline gather(const double* base_addr, const Vectorized<int64_t>& vindex) {
  return _mm512_i64gather_pd(vindex, base_addr, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline gather(const float* base_addr, const Vectorized<int32_t>& vindex) {
  return _mm512_i32gather_ps(vindex, base_addr, scale);
}
#endif

#endif // defined(CPU_CAPABILITY_AVX512)

}}}
