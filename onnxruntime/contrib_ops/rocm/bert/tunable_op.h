// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include "contrib_ops/rocm/bert/timer.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

struct OpParams {
  virtual std::string signature() const = 0;
};

class Op {
 public:
  Op() : repeats_(100) {}

  virtual void Run(const OpParams*) = 0;

  void SetRepeats(int n) {
    repeats_ = n;
  }

  float Profile(const OpParams* op_params) {
    // warm up
    for (int i = 0; i < 5; i++) {
      Run(op_params);
    }
    timer_.Start();
    for (int i = 0; i < repeats_; i++) {
      Run(op_params);
    }
    timer_.End();
    return timer_.time()/repeats_;
  }

  virtual ~Op() {}

 private:
  Timer timer_;
  int repeats_;
};

class TunableOp {
 public:
  TunableOp() {}

  void Run(const OpParams* op_params_) {
    int id;
    if (kernel_map_.find(op_params_->signature()) == kernel_map_.end()) {
      id = FindFastest(op_params_);
      kernel_map_.insert({op_params_->signature(), id});
    } else {
      id = kernel_map_[op_params_->signature()];
    }
    ops_[id]->Run(op_params_);
  }

  virtual ~TunableOp() {}

 protected:
  std::vector<std::unique_ptr<Op>> ops_;

 private:
  int FindFastest(const OpParams* op_params_) {
    assert(ops_.size() > 0);
    float min_time = ops_[0]->Profile(op_params_);
    int id = 0;
    for (int i = 1; i < ops_.size(); i++) {
      float time = ops_[i]->Profile(op_params_);
      if (time < min_time) {
        min_time = time;
        id = i;
      }
    }
    return id;
  }

  std::map<std::string, int> kernel_map_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
