// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class SpaceDepthBase : public OpKernel {
 public:
  SpaceDepthBase(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("blocksize", &blocksize_).IsOK(),
                "Attribute blocksize is not set.");
  }

 protected:
  int64_t blocksize_;
};

template <typename T>
class SpaceToDepth final : public SpaceDepthBase {
 public:
  SpaceToDepth(const OpKernelInfo& info) : SpaceDepthBase(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
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
