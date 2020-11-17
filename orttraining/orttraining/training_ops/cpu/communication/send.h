// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_MPI)
#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class Send final : public OpKernel {
 public:
  Send(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("tag", &tag_).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("element_types", element_types_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  void SendData(
      OpKernelContext* ctx,
      const int dst,
      const int num_tensors,
      size_t aggregated_aligned_tensor_bytes,
      std::vector<size_t> tensor_offsets_in_bytes,
      std::vector<size_t> tensor_sizes_in_bytes) const;

  int64_t tag_;
  std::vector<int64_t> element_types_;
};

}  // namespace contrib
}  // namespace onnxruntime
#endif