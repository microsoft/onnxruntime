// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/fpA_intB_gemv_dispatcher.h"


namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace fpA_intB_gemv {

INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(
    KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true, 64);

}  // namespace fpA_intB_gemv
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
