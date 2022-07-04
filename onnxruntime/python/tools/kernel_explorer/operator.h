// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/rocm/bert/timer.h"

using onnxruntime::contrib::rocm::Timer;

namespace onnxruntime {
namespace rocm {

template<typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

}
}

template <typename T>
class Operator {
 public:
  Operator() : repeats_(100) {}

  virtual void Run() = 0;

  void SetRepeats(int n) {
    repeats_ = n;
  }

  float Profile() {
    // warm up
    for (int i = 0; i < 5; i++) {
      Run();
    }
    Timer timer;
    timer.Start();
    for (int i = 0; i < repeats_; i++) {
      Run();
    }
    timer.End();
    return timer.time()/repeats_;
  }

 private:
  int repeats_;
};
