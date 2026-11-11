/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template <>
void run_mha_fwd_<cutlass::bfloat16_t, 128, true>(Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim128<cutlass::bfloat16_t, true>(params, stream);
}

void run_mha_fwd_hdim128_tile64(Flash_fwd_params& params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    BOOL_SWITCH(params.is_causal, IsCausal, [&] {
      run_flash_fwd<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, elem_type>, IsCausal>(params, stream);
    });
  });
}

}  // namespace FLASH_NAMESPACE
