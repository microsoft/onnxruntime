// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "decoding_base.h"
#include "decoding_traits.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class DecodingGpt2 final : public DecodingBase<T> {
 public:
  DecodingGpt2(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
  
 private:
  
    int batch_size_;
    int candidate_num_;
    int max_seq_len_;
    float probability_threshold_;
    float temperature_;
    int head_num_;
    int size_per_head_;
    int num_layer_;
    int start_id_;
    int end_id_;
    bool is_fuse_qkv_;

    typedef DecodingTraits<T> traits_;
    typedef typename traits_::DataType DataType_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
