// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/slice.h"

namespace onnxruntime {
namespace cuda {

template<typename Tind, bool dynamic>
class Slice final : public CudaKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : CudaKernel(info), SliceBase(info, dynamic) {}
  Status ComputeInternal(OpKernelContext* context) const override;
 private:
  void FillVectorsFromInput(const OpKernelContext* context,
                            std::vector<int64_t>&  raw_starts,
                            std::vector<int64_t>&  raw_ends,
                            std::vector<int64_t>&  raw_axes) const;
};

}  // namespace cuda
}  // namespace onnxruntime
