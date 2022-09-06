// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.cuh"

namespace onnxruntime {
namespace cuda {

TOPKIMPLE(float);
TOPKIMPLE(MLFloat16);
TOPKIMPLE(double);

}  // namespace cuda
}  // namespace onnxruntime
