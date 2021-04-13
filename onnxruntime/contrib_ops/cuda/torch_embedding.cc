// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(USE_TORCH)

#include "contrib_ops/cpu/torch_embedding.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(TorchEmbedding, kMSDomain, 1, kCudaExecutionProvider,
                        KernelDefBuilder()
                            .InputMemoryType<OrtMemTypeCPUInput>(2)
                            .InputMemoryType<OrtMemTypeCPUInput>(3)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .ExternalOutputs(),
                        onnxruntime::contrib::TorchEmbedding);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif
