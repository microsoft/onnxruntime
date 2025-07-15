// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/bert/flash_attention/flash_debug.h"

#ifdef ENABLE_FLASH_DEBUG
namespace onnxruntime {
namespace flash {

__device__ volatile int flash_debug_block_sync = 0;

}
}
#endif
