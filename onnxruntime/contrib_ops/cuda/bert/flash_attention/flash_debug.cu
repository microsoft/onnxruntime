#if USE_FLASH_ATTENTION

#include "contrib_ops/cuda/bert/flash_attention/flash_debug.h"

#ifdef ENABLE_FLASH_DEBUG
namespace onnxruntime {
namespace flash {

__device__ volatile int flash_debug_block_sync = 0;

}  // namespace flash
}  // namespace onnxruntime

#endif
#endif
