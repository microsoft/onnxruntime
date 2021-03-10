// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <iomanip>

namespace onnxruntime {
namespace cuda {
template <typename T>
struct DumpType {
  static std::ostream& dump(std::ostream& os, T v) {
    return os << std::setw(17) << std::setprecision(8) << v;
  }
};

template <>
struct DumpType<MLFloat16> {
  static std::ostream& dump(std::ostream& os, const MLFloat16& v) {
    return DumpType<float>::dump(os, v.ToFloat());
  }
};

template <>
struct DumpType<BFloat16> {
  static std::ostream& dump(std::ostream& os, const BFloat16& v) {
    return DumpType<float>::dump(os, v.ToFloat());
  }
};

template <typename T>
struct DumpArray {
  void operator()(std::ostream& os, const std::string& name, const void* in, size_t len, size_t col_width) const {
    std::unique_ptr<T[]> buf(new T[len]);
    cudaMemcpy(buf.get(), in, len * sizeof(T), cudaMemcpyDeviceToHost);
    const T* src = buf.get();

    os << "Dump array: " << name << std::endl;

    if (col_width == -1) col_width = len;

    for (size_t i = 0; i < len;) {
      for (size_t w = 0; w < col_width && i < len; ++w, ++i) {
        DumpType<T>::dump(os, src[i]);
      }
      os << std::endl;
    }
    os << std::endl;
  }
};

#if 0
#define DUMP_TYPE(T, ...) DumpType<T>()(__VA_ARGS__)
#define DUMP_ARRAY(T, ...) DumpArray<T>()(__VA_ARGS__)
#define DUMP_DISP(var, t, ...) utils::MLTypeCallDispatcher<__VA_ARGS__> var(t)
#define DUMP_INVOKE(var, fn, ...) var.Invoke<fn>(__VA_ARGS__)
#else
#define DUMP_DISP(var, t, ...)
#define DUMP_INVOKE(var, fn, ...)
#define DUMP_TYPE(T, ...)
#define DUMP_ARRAY(T, ...)
#endif
}
}