// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_TORCH

#include "orttraining/training_ops/cpu/tensor/torch_embedding_grad.h"
#include "core/providers/cuda/cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(TorchEmbeddingGrad, kMSDomain, 1, kCudaExecutionProvider,
                        KernelDefBuilder()
                            .InputMemoryType<OrtMemTypeCPUInput>(2)
                            .InputMemoryType<OrtMemTypeCPUInput>(3)
                            .InputMemoryType<OrtMemTypeCPUInput>(4)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .ExternalOutputs(),
                        onnxruntime::contrib::TorchEmbeddingGrad);

}  // namespace cuda
}  // namespace onnxruntime

#endif
