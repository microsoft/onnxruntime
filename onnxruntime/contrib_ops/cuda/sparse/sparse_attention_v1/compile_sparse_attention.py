# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Use triton AoT compiler to convert sparse_attention_triton.py to C source files including cubin and dispatcher.
# Example to use this script (Tested with Python 3.10 and CUDA 12.3 in Ubuntu 20.04):
#    python3 -m pip install numpy==1.26.4 torch==2.3.0 triton==2.3.0
#    python3 compile_sparse_attention.py | sh
#
# Note that sparse_attention_v1_*.cc and sparse_attention_dispatcher_*.h are the generated files.

import math
from itertools import product

import torch


def generate_triton_compile_shell_script(sm, dtype="fp16"):
    assert dtype in ["fp16", "bf16"]
    print("export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)")

    # Modify the compile.py to use custom template file template_h.txt and template_c.txt in current directory.
    # Also pass block_m to the template.
    print(
        'python -c "'
        "import sys;lines=sys.stdin.read();"
        "lines=lines.replace('template_path = Path(__file__).parent / f\\\"compile.{ext}\\\"','template_path = f\\\"compile_template_kernel_{ext}.txt\\\"');"
        'lines=lines.replace(\'\\"_placeholder\\": \\"\\",\', \'\\"_placeholder\\": \\"\\",\\n        \\"block_m\\": list(constants.values())[0],\');'
        'print(lines)"'
        "< ${TRITON_ROOT}/triton/tools/compile.py > compile.py"
    )

    out_dir = f"trition_cubin_{dtype}"
    print(f"rm -rf {out_dir}")
    print(f"mkdir -p {out_dir}")

    block_n_values = [64]
    block_d_values = [64]
    num_block_d_values = [2]
    even_n_values = [True, False]
    # Use triton compiler to compile the kernel of different combinations of constant parameters.
    for block_n, block_d, num_blocks_d, even_n in product(
        block_n_values, block_d_values, num_block_d_values, even_n_values
    ):
        head_size = block_d * num_blocks_d
        block_m = block_n
        even_m = even_n
        scalar_params = "i32,i32,i32,fp32,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32,i32,i32"
        sig = f"*{dtype}:16,*{dtype}:16,*{dtype}:16,*{dtype}:16,*i32:16,*i32:16,{scalar_params},{block_m},{int(even_m)},{block_n},{int(even_n)},{block_d},{num_blocks_d}"
        prefix = "python compile.py sparse_attention_triton.py"
        filename = f"sparse_attention_v1_{dtype}_d{head_size}_n{block_n}_e{int(even_n)}_sm{sm}"
        name = f"sparse_attention_{dtype}_sm{sm}"
        num_warps = max(1, 2 ** int(math.log2(min(block_m, block_n, block_d) / 16)))

        # Shared memory is 96KB for V100 (sm70), 64KB for T4 (sm75), 164KB for A100 (sm80), 228KB for H100 (sm90).
        # Adjust stages so that shared memory size is within limit, and choose the one with best performance.
        sm_to_stages = {90: 3, 80: 2, 75: 2}

        num_stages = sm_to_stages[sm]

        # TODO: use different kernel name (change the name in sparse_attention_triton.py before running compile.py)
        print(
            f"{prefix} -n block_sparse_attention_kernel -o {out_dir}/{filename} --out-name {name} "
            f'-w {num_warps} -ns {num_stages} -s "{sig}" -g "(total_seq_len - past_seq_len + {block_m} - 1) / {block_m}, batch_size * num_heads, 1"'
        )

    # Generate the dispatcher.
    dispatcher = f"sparse_attention_dispatcher_{dtype}_sm{sm}"
    print(f"cd {out_dir}")
    print(f"python ${{TRITON_ROOT}}/triton/tools/link.py sparse_attention_v1_*.h -o {dispatcher}")
    print("rm *.h")

    # Remove signature hash in code.
    suffix = "0d1d2d3d4d5d678910d11d12d13d14d15d16d17d18d19d20d21d222324"
    print(f"for file in *.c; do sed -i 's/_{suffix}//g'  \"$file\"; done")

    # Recover signature hash in kernel name that is removed in previous step. Kernel name shall not be changed.
    print(
        f"for file in *.c; do sed -i 's/block_sparse_attention_kernel/block_sparse_attention_kernel_{suffix}/g'  \"$file\"; done"
    )

    # Remove signature hash from filename since we use same signature for all kernels except constants.
    # and we have constants in filename so that we can distinguish files without the hash.
    print('for file in sparse_attention_v1_*.c; do mv -- "$file" "$(echo $file | cut -f 1 -d \'.\').c"; done')

    # Change function parameters and return type. If you change the kernel interface, you will need to modify this part.
    source1 = "CUstream stream, CUdeviceptr out, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr layout_csr_row_indices, CUdeviceptr layout_csr_col_indices, int32_t layout_csr_row_stride_h, int32_t layout_csr_col_stride_h, int32_t num_layout, float softmax_scale, int32_t stride_qb, int32_t stride_qh, int32_t stride_qm, int32_t stride_kb, int32_t stride_kh, int32_t stride_kn, int32_t stride_vb, int32_t stride_vh, int32_t stride_vn, int32_t stride_ob, int32_t stride_oh, int32_t stride_om, int32_t num_heads, int32_t num_kv_heads, int32_t total_seq_len"
    target1 = "SparseAttentionParams& params"
    source2 = "stream, out, Q, K, V, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_ob, stride_oh, stride_om, num_heads, num_kv_heads, total_seq_len"
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
        "template=Path('../compile_template_dispatcher_h.txt').read_text();"
        f"code=Path('{dispatcher}.h').read_text();"
        "text=template.replace('PLACEHOLDER', code); print(text)\" "
        f"> ../{dispatcher}.h"
    )
    # rename *.c to *.cc
    print('for file in *.c; do mv -- "$file" "${file%.c}.cc"; done')

    # Move kernel files to parent directory. This might overwrite existing files in repository.
    print("mv sparse_attention_v1_* ../")

    # Clean up
    print("cd ..")
    print("rm compile.py")
    print(f"rm -rf {out_dir}")

    print("echo Done")


if __name__ == "__main__":
    major, minor = torch.cuda.get_device_capability()
    print(f"echo Generate sparse attention v1 kernels for compute capability:{major}.{minor}")
    assert major >= 7, "triton only supports compute capability >= 7.0"

    sm = major * 10 + minor
    for dtype in ["fp16", "bf16"] if major >= 8 else ["fp16"]:
        generate_triton_compile_shell_script(sm, dtype)
