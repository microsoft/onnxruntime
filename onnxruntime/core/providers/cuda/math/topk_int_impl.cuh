// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.cuh"

namespace onnxruntime {
namespace cuda {

TOPKIMPLE(int8_t);
TOPKIMPLE(int16_t);
TOPKIMPLE(int32_t);
TOPKIMPLE(int64_t);

}  // namespace cuda
}  // namespace onnxruntime
