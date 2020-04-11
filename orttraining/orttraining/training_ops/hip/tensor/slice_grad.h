// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/tensor/slice.h"

namespace onnxruntime {
namespace hip {

class SliceGrad final : public Slice<true> {
 public:
  SliceGrad(const OpKernelInfo& info) : Slice(info) {}

 private:
  const Tensor* GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const override;
  void FillInputVectors(OpKernelContext* ctx, std::vector<int64_t>& input_starts, std::vector<int64_t>& input_ends,
                        std::vector<int64_t>& input_axes, std::vector<int64_t>& input_steps) const override;

  Status CallSliceImp(size_t element_size, size_t dimension_count, const int64_t* starts_buffer,
                      const int64_t* steps_buffer, const int64_t* input_strides,
                      const fast_divmod* output_strides, OpKernelContext* ctx, TensorShape output_shape)
      const override;
};
}  // namespace hip
}  // namespace onnxruntime
