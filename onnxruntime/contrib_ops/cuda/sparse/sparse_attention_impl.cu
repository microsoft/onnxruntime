// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sparse/sparse_attention_impl.h"
//#include "contrib_ops/cuda/sparse/sparse_attention_tunable_op.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& /*device_prop*/,
    cublasHandle_t& /*cublas*/,
    Stream* /*stream*/,
    contrib::SparseAttentionParameters& /*parameters*/,
    SparseAttentionData<T>& /*data*/) {

// TODO: implement the kernel
return Status::OK();
}

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<half>& data);

template Status QkvToContext<BFloat16>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<BFloat16>& data);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
