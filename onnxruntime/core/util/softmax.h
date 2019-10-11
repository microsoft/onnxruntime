/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/util/eigen_common_wrapper.h"

// copied from tensorflow/core/kernels/softmax_op.cc
template <bool log, typename Device, typename T1, typename T2>
void ComputeSoftMax(const Device& d, T1& logits, T2& softmax, int batch_size, int num_classes) {
  const int kClassDim = 1;

  // const int num_classes = (int)logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
  Eigen::DSizes<int, 1> along_class(kClassDim);
  Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
  Eigen::DSizes<int, 2> one_by_class(1, num_classes);
#else
  Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
  Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
  batch_by_one.set(0, batch_size);
  Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
  one_by_class.set(1, num_classes);
#endif
  // shifted_logits = logits - max(logits along classes);
  auto shifted_logits = (logits - logits.maximum(along_class).eval().reshape(batch_by_one).broadcast(one_by_class));
  if (log) {
    // Calculate the log of the softmax
    // softmax = logits - max(logits along classes);
    softmax.device(d) = shifted_logits;
    // softmax = softmax - log(sum(exp(softmax along classes)));
    softmax.device(d) =
        (softmax - softmax.exp().sum(along_class).log().eval().reshape(batch_by_one).broadcast(one_by_class));
  } else {
    // NOTE(touts): If you modify this implementation please run
    // the BM_ImageNetSoftmaxFwd benchmark in nn_ops_test.cc.
    //
    // softmax = exp(logits - max(logits along classes));
    softmax.device(d) = shifted_logits.exp();
    // softmax = softmax * (1 / sum(softmax along classes));
    softmax.device(d) =
        (softmax * softmax.sum(along_class).inverse().eval().reshape(batch_by_one).broadcast(one_by_class));
  }
}

//The below functions are not copied from TF.

// this function skips zero values (since exp(0) is non zero)
template <typename T>
void ComputeSoftmaxZero(T* values, size_t values_len, T* out_p) {
  std::vector<T> newscores(values_len);
  // compute exp with negative number to be numerically stable
  T v_max = -std::numeric_limits<T>::max();
  for (size_t i = 0; i != values_len; ++i) {
    auto value = values[i];
    if (value > v_max)
      v_max = value;
  }

  T exp_neg_v_max = std::exp(-v_max);
  T this_sum = 0.f;
  for (size_t i = 0; i != values_len; ++i) {
    auto value = values[i];
    if (value > 0.0000001f || value < -0.0000001f) {
      T val2 = std::exp(value - v_max);
      this_sum += val2;
      newscores[i] = val2;
    } else {
      newscores[i] = value * exp_neg_v_max;
    }
  }
  for (int64_t k = 0; k < static_cast<int64_t>(values_len); k++) {
    out_p[k] = newscores[k] / this_sum;
  }
}

template <typename T>
void ComputeSoftmax(T* values, size_t values_len, T* out_p) {
  std::vector<T> newscores(values_len);
  // compute exp with negative number to be numerically stable
  T v_max = -std::numeric_limits<T>::max();
  for (size_t i = 0; i != values_len; ++i) {
    auto value = values[i];
    if (value > v_max)
      v_max = value;
  }
  T this_sum = 0.f;
  for (size_t i = 0; i != values_len; ++i) {
    auto value = values[i];
    T val2 = std::exp(value - v_max);
    this_sum += val2;
    newscores[i] = val2;
  }
  for (int64_t k = 0; k < static_cast<int64_t>(values_len); k++) {
    out_p[k] = newscores[k] / this_sum;
  }
}
