// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/space_depth_ops.h"

namespace onnxruntime {
namespace rocm {

class SpaceToDepth final : public RocmKernel, SpaceDepthBase {
 public:
  SpaceToDepth(const OpKernelInfo& info) : RocmKernel(info), SpaceDepthBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

class DepthToSpace final : public RocmKernel, SpaceDepthBase {
 public:
  DepthToSpace(const OpKernelInfo& info) : RocmKernel(info), SpaceDepthBase(info) {
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

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool is_dcr_ = true;
};

}  // namespace rocm
}  //namespace onnxruntime
