// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/inplace_to_sequence.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(InPlaceToSequence, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .SequenceAlias(0)
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
                        InPlaceToSequence);

}  // namespace cuda
}  // namespace onnxruntime
