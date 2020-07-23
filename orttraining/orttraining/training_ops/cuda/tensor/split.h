// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/split.h"

namespace onnxruntime {
namespace cuda {

class SplitTraining final : public CudaKernel, public SplitBase {
 public:
  SplitTraining(const OpKernelInfo& info) : CudaKernel(info), SplitBase(info) {}
  Status PrepareForCompute(const TensorShape& input_shape, int num_outputs, int64_t& axis, int& before_dims,
                           int& after_dims_including_split_axis, int& after_dims_excluding_split,
                           std::vector<int64_t>& split_sizes) const override;
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
