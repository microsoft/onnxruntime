// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_LEAN_ATTENTION

#include "contrib_ops/cuda/bert/lean_attention/lean_fwd_launch_template.h"

namespace onnxruntime {
namespace lean {

template void run_mha_fwd_lean_dispatch<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream);

}  // namespace flash
}  // namespace onnxruntime
#endif
