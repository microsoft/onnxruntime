// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.
#if USE_FLASH_ATTENTION

#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace onnxruntime {
namespace flash {

template <>
void run_mha_fwd_<cutlass::half_t, 192>(Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim192<cutlass::half_t>(params, stream);
}

}  // namespace flash
}  // namespace onnxruntime
#endif
