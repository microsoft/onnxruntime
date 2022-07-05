// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include "contrib_ops/rocm/bert/util.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

struct OpParams {
  explicit OpParams(hipStream_t stream) : stream(stream) {}
  virtual std::string signature() const = 0;
  hipStream_t stream;
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
  explicit TunableOp(int default_id) : default_id_(default_id), tuning_(false) {}

  void Run(const OpParams* op_params) {
    int id;
    if (tuning_ == true && Condition(op_params)) {
      if (kernel_map_.find(op_params->signature()) == kernel_map_.end()) {
        id = FindFastest(op_params);
        kernel_map_.insert({op_params->signature(), id});
      } else {
        id = kernel_map_[op_params->signature()];
      }
    } else {
      id = default_id_;
    }
    ops_[id]->Run(op_params);
  }

  void EnableTuning() {
    tuning_ = true;
  }

  void DisableTuning() {
    tuning_ = false;
  }

  virtual ~TunableOp() {}

 protected:
  std::vector<std::unique_ptr<Op>> ops_;

 private:
  virtual bool Condition(const OpParams* op_params) = 0;

  int FindFastest(const OpParams* op_params) {
    assert(ops_.size() > 0);
    float min_time = ops_[0]->Profile(op_params);
    int id = 0;
    for (int i = 1; i < ops_.size(); i++) {
      float time = ops_[i]->Profile(op_params);
      if (time < min_time) {
        min_time = time;
        id = i;
      }
    }
    return id;
  }

  std::map<std::string, int> kernel_map_;
  int default_id_;
  bool tuning_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
