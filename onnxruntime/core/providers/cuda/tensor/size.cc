// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/size.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Size,
    kOnnxDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .TypeConstraint("T",
                        std::vector<MLDataType>({DataTypeImpl::GetTensorType<float>(),
                                                 DataTypeImpl::GetTensorType<double>(),
                                                 DataTypeImpl::GetTensorType<int8_t>(),
                                                 DataTypeImpl::GetTensorType<int16_t>(),
                                                 DataTypeImpl::GetTensorType<int32_t>(),
                                                 DataTypeImpl::GetTensorType<int64_t>(),
                                                 DataTypeImpl::GetTensorType<uint8_t>(),
                                                 DataTypeImpl::GetTensorType<uint16_t>(),
                                                 DataTypeImpl::GetTensorType<uint32_t>(),
                                                 DataTypeImpl::GetTensorType<uint64_t>(),
                                                 DataTypeImpl::GetTensorType<std::string>(),
                                                 DataTypeImpl::GetTensorType<bool>()}))
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Size);

}  // namespace cuda
}  // namespace onnxruntime
