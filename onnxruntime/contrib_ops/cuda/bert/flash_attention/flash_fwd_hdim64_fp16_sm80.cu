// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <>
void run_mha_fwd_<cutlass::half_t, 64>(Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim64<cutlass::half_t>(params, stream);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
