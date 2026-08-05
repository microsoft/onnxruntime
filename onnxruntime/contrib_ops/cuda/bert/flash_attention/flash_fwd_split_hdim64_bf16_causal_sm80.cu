/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, true>(Flash_fwd_params& params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
