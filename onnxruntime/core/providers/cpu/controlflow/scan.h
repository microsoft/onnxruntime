// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class Scan final : public OpKernel {
 public:
  Scan(const OpKernelInfo& info) : OpKernel(info) {
    // make sure the attribute was present even though we don't need it here.
    // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
    // and a SessionState instance for executing the subgraph is created by InferenceSession.
    // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
    ONNX_NAMESPACE::GraphProto proto;
    ONNXRUNTIME_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
    (void)proto;

    ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs_).IsOK());

    if (info.GetAttrs<int64_t>("directions", directions_).IsOK()) {
      ONNXRUNTIME_ENFORCE(gsl::narrow_cast<int64_t>(directions_.size()) == num_scan_inputs_,
                          "Number of entries in 'directions' was ", directions_.size(),
                          ". Must match 'num_scan_inputs' of ", num_scan_inputs_);
      ONNXRUNTIME_ENFORCE(std::all_of(directions_.cbegin(), directions_.cend(),
                                      [](int64_t i) { return i == static_cast<int64_t>(Direction::kForward) ||
                                                             i == static_cast<int64_t>(Direction::kReverse); }),
                          "Invalid values in 'directions'. 0 == forward. 1 == reverse.");
    } else {
      // default to forward
      directions_ = std::vector<int64_t>(num_scan_inputs_, static_cast<int64_t>(Direction::kForward));
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

  enum class Direction { kForward = 0,
                         kReverse = 1 };

 private:
  int64_t num_scan_inputs_;
  std::vector<int64_t> directions_;
};
}  // namespace onnxruntime
