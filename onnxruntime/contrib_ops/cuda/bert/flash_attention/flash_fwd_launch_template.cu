#if USE_FLASH_ATTENTION

#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace onnxruntime {
namespace flash {

#ifdef ENABLE_FLASH_DEBUG
__device__ volatile int flash_debug_block_sync = 0;
#endif

}  // namespace flash
}  // namespace onnxruntime

#endif
