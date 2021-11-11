// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class Trilu final : public CudaKernel {
 public:
  Trilu(const OpKernelInfo& info) : CudaKernel(info), upper_(info.GetAttrOrDefault<int64_t>("upper", 1) >= 1) {
  }
  ~Trilu() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool upper_;
};

}  // namespace cuda
}  // namespace onnxruntime