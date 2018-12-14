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
  }

  Status Compute(OpKernelContext* context) const override;
};

}  //namespace onnxruntime
