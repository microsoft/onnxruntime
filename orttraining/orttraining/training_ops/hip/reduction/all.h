// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

template <typename T>
class All final : public HipKernel {
 public:
  All(const OpKernelInfo& info) : HipKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template<typename T>
void LaunchAllKernel(const T* data, const int size, bool* output);

}  // namespace hip
}  // namespace onnxruntime
