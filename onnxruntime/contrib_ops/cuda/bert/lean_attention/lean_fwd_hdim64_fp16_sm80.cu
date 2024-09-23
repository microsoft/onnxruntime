// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#if USE_FLASH_ATTENTION

#include "contrib_ops/cuda/bert/lean_attention/lean_fwd_launch_template.h"

namespace onnxruntime {
namespace lean {


template void run_mha_fwd_lean_dispatch<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream);


}  // namespace flash
}  // namespace onnxruntime
#endif

