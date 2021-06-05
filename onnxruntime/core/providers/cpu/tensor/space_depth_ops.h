// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class SpaceDepthBase : public OpKernel {
 protected:
  SpaceDepthBase(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("blocksize", &blocksize_).IsOK(),
                "Attribute blocksize is not set.");
  }

  Status InputValidationsAndOutputDims(const Tensor& input,
                                       int64_t& batch,
                                       int64_t& input_depth, int64_t& input_height, int64_t& input_width,
                                       int64_t& output_depth, int64_t& output_height, int64_t& output_width,
                                       bool is_space_to_depth) const;

  int64_t blocksize_;
};

class SpaceToDepth final : public SpaceDepthBase {
 public:
  SpaceToDepth(const OpKernelInfo& info) : SpaceDepthBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

class DepthToSpace final : public SpaceDepthBase {
 public:
  DepthToSpace(const OpKernelInfo& info) : SpaceDepthBase(info) {
    std::string mode;
    // if  mode doesn't exist, then it is the default "DCR" mode
    // (or) it is an opset < 11 model for which the only mode is "DCR" mode
    if (info.GetAttr("mode", &mode).IsOK()) {
      if (mode == "CRD")
        is_dcr_ = false;

      else if (mode != "DCR")
        ORT_THROW("DepthToSpace op: only 'DCR' and 'CRD' modes are supported");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool is_dcr_ = true;
};

}  //namespace onnxruntime
