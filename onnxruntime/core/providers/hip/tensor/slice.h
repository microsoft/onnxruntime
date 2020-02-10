// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace hip {

template<typename Tind, bool dynamic>
class Slice final : public HipKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : HipKernel(info), SliceBase(info, dynamic) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace hip
}  // namespace onnxruntime
