// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/fastertransformer/utils/common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T> class DecodingTraits;
  
template <>
class DecodingTraits<float>
{
  public:
    typedef float DataType;
    static const fastertransformer::OperationType OpType = fastertransformer::OperationType::FP32;
};

template <>
class DecodingTraits<MLFloat16>
{
  public:
    typedef half DataType;
    static const fastertransformer::OperationType OpType = fastertransformer::OperationType::FP16;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

