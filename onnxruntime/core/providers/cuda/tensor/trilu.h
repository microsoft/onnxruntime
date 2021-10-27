// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

class Trilu final : public CudaKernel {
 public:
  Trilu(const OpKernelInfo& info) : CudaKernel(info), upper_(true) {
    int64_t temp_upper;
    Status status = info.GetAttr<int64_t>("upper", &temp_upper);
    if (status.IsOK()) {
      upper_ = temp_upper >= 1;
    }
  }
  ~Trilu() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool upper_;
};

}  // namespace cuda
}  // namespace onnxruntime