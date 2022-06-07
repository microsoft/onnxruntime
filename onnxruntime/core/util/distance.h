// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cmath>
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

// Computes the squared Euclidean distance between the vectors.
template <typename T>
class Sqeuclidean {
 public:
  T operator()(const T* a1, const T* b1, size_t n) const {
    // if n is too small, Eigen is much slower than a plain loop
    T sum = 0;
    for (size_t k = 0; k != n; ++k) {
      const T t = a1[k] - b1[k];
      sum += t * t;
    }
    return sum;
  }
};

// Computes the Euclidean distance between the vectors.
template <typename T>
class Euclidean {
 public:
  T operator()(const T* a1, const T* b1, size_t n) const {
    // if n is too small, Eigen is much slower than a plain loop
    T sum = 0;
    for (size_t k = 0; k != n; ++k) {
      const T t = a1[k] - b1[k];
      sum += t * t;
    }
    return std::sqrt(sum);
  }
};

template <typename T>
class SqeuclideanWithEigen {
 public:
  T operator()(const T* a1, const T* b1, size_t n) const {
    return (ConstEigenVectorMap<T>(a1, n) - ConstEigenVectorMap<T>(b1, n)).array().square().sum();
  }
};

template <typename T>
class EuclideanWithEigen {
 public:
  T operator()(const T* a1, const T* b1, size_t n) const {
    return std::sqrt((ConstEigenVectorMap<T>(a1, n) - ConstEigenVectorMap<T>(b1, n)).array().square().sum());
  }
};
}  // namespace onnxruntime