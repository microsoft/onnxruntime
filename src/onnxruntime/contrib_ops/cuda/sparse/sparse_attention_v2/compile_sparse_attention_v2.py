# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Use triton AoT compiler to convert sparse_attention_v2_triton.py to C source files including cubin and dispatcher.
# Example to use this script (Tested with Python 3.10 and CUDA 12.3 in Ubuntu 20.04):
#    python3 -m pip install numpy==1.26.4 torch==2.3.0 triton==2.3.0
#    python3 compile_sparse_attention_v2.py | sh
#
# Note that sparse_attention_v2_*.cc and sparse_attention_v2_dispatcher_*.h are the generated files.

from itertools import product

import torch
import triton


def generate_triton_compile_shell_script(sm, dtype="fp16"):
    assert dtype in ["fp16", "bf16"]
    print("export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)")

    # Modify the compile.py to use custom template files compile_template_kernel_v2_c/h.txt in current directory.
    print(
        'python -c "'
        "import sys;lines=sys.stdin.read();"
        "lines=lines.replace('template_path = Path(__file__).parent / f\\\"compile.{ext}\\\"','template_path = f\\\"compile_template_kernel_v2_{ext}.txt\\\"');"
        'print(lines)"'
        "< ${TRITON_ROOT}/triton/tools/compile.py > compile.py"
    )

    out_dir = f"trition_cubin_{dtype}"
    print(f"rm -rf {out_dir}")
    print(f"mkdir -p {out_dir}")

    # All combinations of parameters for kernel.
    has_batch_dim_values = [True]
    head_size_values = [128]
    block_size_values = [64]
    is_prompt_values = [True, False]

    # Use triton compiler to compile the kernel of different combinations of constant parameters.
    for has_batch_dim, head_size, block_size, is_prompt in product(
        has_batch_dim_values, head_size_values, block_size_values, is_prompt_values
    ):
        # Constant parameters for triton kernel.
        block_d = triton.next_power_of_2(head_size)
        block_m = block_size
        block_n = block_size
        block_m_loading = 16 if not is_prompt else block_size
        even_d = block_d == head_size
        m_lt_n = block_m < block_n
        num_warps = 1 if not is_prompt else 4

        # Shared memory is 96KB for V100 (sm70), 64KB for T4 (sm75), 164KB for A100 (sm80), 228KB for H100 (sm90).
        # Adjust stages so that shared memory size is within limit, and choose the one with best performance.
        sm_to_stages = {90: 3, 80: 3, 75: 2}

        num_stages = sm_to_stages[sm]

        # There are 4 float and 8 int32 buffer pointers, and they are assumed to be aligned to 16 bytes.
        tensor_params = f"*{dtype}:16," * 4 + "*i32:16," * 8

        # The strides for Q, K, V, and Out are multiples of 16 since head_size is 128.
        scalar_params = ("i32," * 2) + ("i32:16," * 12) + "i32,i32,fp32,"

        constant_params = f"{int(has_batch_dim)},{head_size},{block_m},{block_n},{block_d},{block_m_loading},{int(even_d)},{int(m_lt_n)}"
        signature = f"{tensor_params}{scalar_params}{constant_params}"
        prefix = "python compile.py sparse_attention_v2_triton.py"

        # output filename
        filename = f"sparse_attention_v2_{dtype}_d{head_size}_m{block_m}_{block_m_loading}_n{block_n}_b{int(has_batch_dim)}_sm{sm}"

        # function name
        name = f"sparse_attention_v2_{dtype}_sm{sm}"

        print(
            f"{prefix} -n block_sparse_attention -o {out_dir}/{filename} --out-name {name} "
            f'-w {num_warps} -ns {num_stages} -s "{signature}" -g "query_blocks, num_heads, 1"'
        )

    # Generate the dispatcher.
    dispatcher = f"sparse_attention_v2_dispatcher_{dtype}_sm{sm}"
    print(f"cd {out_dir}")
    print(f"python ${{TRITON_ROOT}}/triton/tools/link.py sparse_attention_v2_*.h -o {dispatcher}")
    print("rm *.h")

    # Remove signature in code.
    suffix = "0d1d2d3d4d5d6d7d8d9d10d11d121314d15d16d17d18d19d20d21d22d23d24d25d262728"
    print(f"for file in *.c; do sed -i 's/_{suffix}//g'  \"$file\"; done")

    # Recover signature in kernel name that is removed in previous step. Kernel name shall not be changed.
    print(f"for file in *.c; do sed -i 's/block_sparse_attention/block_sparse_attention_{suffix}/g'  \"$file\"; done")

    # Remove signature from filename since we use same signature for all kernels except constants.
    # and we have constants in filename so that we can distinguish files without the hash.
    print(f'for file in sparse_attention_v2_{dtype}_*.c; do mv "$file" "$(echo $file | cut -f 1 -d \'.\').c"; done')

    # Change function parameters and return type. If you change the kernel interface, you will need to modify this part.
    source1 = "CUstream stream, CUdeviceptr Out, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr q_batch_starts, CUdeviceptr q_batch_ends, CUdeviceptr k_batch_starts, CUdeviceptr k_batch_ends, CUdeviceptr q_batch_ids, CUdeviceptr q_start_sids, CUdeviceptr layout_crow_ptr, CUdeviceptr layout_col_ptr, int32_t layout_crow_stride_h, int32_t layout_col_stride_h, int32_t stride_qb, int32_t stride_qt, int32_t stride_qh, int32_t stride_kb, int32_t stride_kt, int32_t stride_kh, int32_t stride_vb, int32_t stride_vt, int32_t stride_vh, int32_t stride_ob, int32_t stride_ot, int32_t stride_oh, int32_t q_k_ratio, int32_t num_layout, float softmax_scale"
    target1 = "SparseAttentionParams& params"
    source2 = "stream, Out, Q, K, V, q_batch_starts, q_batch_ends, k_batch_starts, k_batch_ends, q_batch_ids, q_start_sids, layout_crow_ptr, layout_col_ptr, layout_crow_stride_h, layout_col_stride_h, stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh, stride_ob, stride_ot, stride_oh, q_k_ratio, num_layout, softmax_scale"
    target2 = "params"
    print(
        f"python -c \"import sys;lines=sys.stdin.read();lines=lines.replace('{source1}', '{target1}');"
        f'lines=lines.replace(\'{source2}\', \'{target2}\');print(lines)" < "{dispatcher}.c" > "{dispatcher}.h"'
    )
    print(f"sed -i 's/CUresult/Status/g'  \"{dispatcher}.h\"")

    # Remove parameter checking since we moved the validation logic to SparseAttentionParams
    print(f"sed -i '/if /d'  \"{dispatcher}.h\"")
    print(f"sed -i '/CUDA_ERROR_INVALID_VALUE/d'  \"{dispatcher}.h\"")
    print(f"sed -i '/#include/d'  \"{dispatcher}.h\"")

    print(f"rm {dispatcher}.c")

    # Use a template file to add namespace and includes to the dispatcher file.
    print(
        'python -c "'
        "from pathlib import Path;"
        "template=Path('../compile_template_dispatcher_v2_h.txt').read_text();"
        f"code=Path('{dispatcher}.h').read_text();"
        "text=template.replace('PLACEHOLDER', code); print(text)\" "
        f"> ../{dispatcher}.h"
    )
    # rename *.c to *.cc
    print('for file in *.c; do mv -- "$file" "${file%.c}.cc"; done')

    # Move kernel files to parent directory. This might overwrite existing files in repository.
    print("echo Generated files:")
    print("ls sparse_attention_v2_*")
    print(f"mv -f sparse_attention_v2_{dtype}_* ../")

    # Clean up
    print("cd ..")
    print("rm compile.py")
    print(f"rm -rf {out_dir}")

    print(f"echo compiling {dtype} is done")


if __name__ == "__main__":
    major, minor = torch.cuda.get_device_capability()
    print(f"echo Generate sparse attention v2 kernels for compute capability:{major}.{minor}")
    assert major >= 7, "triton only supports compute capability >= 7.0"

    sm = major * 10 + minor
    for dtype in ["fp16", "bf16"] if major >= 8 else ["fp16"]:
        generate_triton_compile_shell_script(sm, dtype)
