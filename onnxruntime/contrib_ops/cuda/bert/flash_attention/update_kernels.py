import os

base_dir = os.path.dirname(os.path.realpath(__file__))
dims = [32, 64, 96, 128, 192, 256]
types = ["fp16", "bf16"]

copyright_header = """/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
"""

template_standard = (
    copyright_header
    + """
#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {{

template<>
void run_mha_fwd_<{cpp_type}, {dim}, {is_causal}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{dim}<{cpp_type}, {is_causal}>(params, stream);
}}

}} // namespace FLASH_NAMESPACE
"""
)

template_split = (
    copyright_header
    + """
#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {{

template void run_mha_fwd_splitkv_dispatch<{cpp_type}, {dim}, {is_causal}>(Flash_fwd_params &params, cudaStream_t stream);

}} // namespace FLASH_NAMESPACE
"""
)

for dim in dims:
    for t in types:
        cpp_type = "cutlass::half_t" if t == "fp16" else "cutlass::bfloat16_t"

        # Standard - Non-causal
        filename = f"flash_fwd_hdim{dim}_{t}_sm80.cu"
        filepath = os.path.join(base_dir, filename)
        content = template_standard.format(cpp_type=cpp_type, dim=dim, is_causal="false")
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Updated {filename}")

        # Standard - Causal
        filename_causal = f"flash_fwd_hdim{dim}_{t}_causal_sm80.cu"
        filepath_causal = os.path.join(base_dir, filename_causal)
        content_causal = template_standard.format(cpp_type=cpp_type, dim=dim, is_causal="true")
        with open(filepath_causal, "w") as f:
            f.write(content_causal)
        print(f"Updated {filename_causal}")

        # Split - Non-causal
        filename_split = f"flash_fwd_split_hdim{dim}_{t}_sm80.cu"
        filepath_split = os.path.join(base_dir, filename_split)
        content_split = template_split.format(cpp_type=cpp_type, dim=dim, is_causal="false")
        with open(filepath_split, "w") as f:
            f.write(content_split)
        print(f"Updated {filename_split}")

        # Split - Causal
        filename_split_causal = f"flash_fwd_split_hdim{dim}_{t}_causal_sm80.cu"
        filepath_split_causal = os.path.join(base_dir, filename_split_causal)
        content_split_causal = template_split.format(cpp_type=cpp_type, dim=dim, is_causal="true")
        with open(filepath_split_causal, "w") as f:
            f.write(content_split_causal)
        print(f"Updated {filename_split_causal}")
