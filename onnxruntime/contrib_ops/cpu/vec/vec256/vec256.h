#pragma once

#include <contrib_ops/cpu/vec/intrinsics.h>
#include <contrib_ops/cpu/vec/vec_base.h>
#include <contrib_ops/cpu/vec/vec256/vec256_float.h>

#include <cstring>
#include <ostream>

namespace onnxruntime::vec {

inline namespace CPU_CAPABILITY {

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

}  // namespace CPU_CAPABILITY
}  // namespace onnxruntime::vec
