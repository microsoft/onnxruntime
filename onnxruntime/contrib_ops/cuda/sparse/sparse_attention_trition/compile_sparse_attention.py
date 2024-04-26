# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Use triton AoT compiler to convert sparse_attention_triton.py to C source files including cubin and dispatcher.
# Example to use this script (Tested with CUDA 12.3 in Ubuntu 20.04):
#    python3 -m pip install triton==2.3.0
#    python3 compile_sparse_attention.py --dtype fp16 | sh
#    python3 compile_sparse_attention.py --dtype bf16 | sh
#
# Note that sparse_attention_kernel_*.cc and sparse_attention_dispatcher_*.h are the generated files.

import argparse
import math
from itertools import product


def generate_triton_compile_shell_script(dtype="fp16"):
    assert dtype in ["fp16", "bf16"]
    print("export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)")
    print('export ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)"')
    print("export SM=$(echo $ARCH | sed -e 's/\\.//g')")

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

    # Note that block_n * num_block_d is the head_size. We support head_size = 128 for now.
    block_n_values = [64]
    block_d_values = [64]
    num_block_d_values = [2]
    even_m_values = [True, False]
    even_n_values = [True, False]

    # Use triton compiler to compile the kernel of different combinations of constant parameters.
    for block_n, block_d, num_blocks_d, even_m, even_n in product(
        block_n_values, block_d_values, num_block_d_values, even_m_values, even_n_values
    ):
        block_m_values = [16, block_n] if block_n != 16 else [block_n]
        for block_m in block_m_values:
            scalar_params = "i32,i32,i32,fp32,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32,i32,i32,i32"
            sig = f"*{dtype}:16,*{dtype}:16,*{dtype}:16,*{dtype}:16,*i32,*i32:16,{scalar_params},{block_m},{int(even_m)},{block_n},{int(even_n)},{block_d},{num_blocks_d}"
            prefix = "python compile.py sparse_attention_triton.py"
            filename = f"sparse_attention_kernel_{dtype}_m{block_m}_{int(even_m)}_n{block_n}_{int(even_n)}_d{block_d}_{num_blocks_d}_sm${{SM}}"
            name = f"sparse_attention_{dtype}_sm${{SM}}"
            num_warps = max(1, 2 ** int(math.log2(min(block_m, block_n, block_d) / 16)))
            num_stages = 2
            # TODO: use different kernel name (change the name in sparse_attention_triton.py before running compile.py)
            print(
                f"{prefix} -n block_sparse_attention_kernel -o {out_dir}/{filename} --out-name {name} "
                f'-w {num_warps} -ns {num_stages} -s "{sig}" -g "(total_seq_len - past_seq_len + {block_m} - 1) / {block_m}, batch_size * num_heads, 1"'
            )

    # Generate the dispatcher.
    dispatcher = f"sparse_attention_dispatcher_{dtype}_sm${{SM}}"
    print(f"cd {out_dir}")
    print(f"python ${{TRITON_ROOT}}/triton/tools/link.py sparse_attention_kernel_*.h -o {dispatcher}")
    print("rm *.h")

    # Remove signature hash in code.
    suffix = "0d1d2d3d45d678910d11d12d13d14d15d16d17d18d19d20d21d22232425"
    print(f"for file in *.c; do sed -i 's/_{suffix}//g'  \"$file\"; done")

    # Recover signature hash in kernel name that is removed in previous step. Kernel name shall not be changed.
    print(
        f"for file in *.c; do sed -i 's/block_sparse_attention_kernel/block_sparse_attention_kernel_{suffix}/g'  \"$file\"; done"
    )

    # Remove signature hash from filename since we use same signature for all kernels except constants.
    # and we have constants in filename so that we can distinguish files without the hash.
    print('for file in sparse_attention_kernel_*.c; do mv -- "$file" "$(echo $file | cut -f 1 -d \'.\').c"; done')

    # Change function parameters and return type. If you change the kernel interface, you will need to modify this part.
    source1 = "CUstream stream, CUdeviceptr out, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr layout_csr_row_indices, CUdeviceptr layout_csr_col_indices, int32_t layout_csr_row_stride_h, int32_t layout_csr_col_stride_h, int32_t num_layout, float softmax_scale, int32_t stride_qb, int32_t stride_qh, int32_t stride_qm, int32_t stride_kb, int32_t stride_kh, int32_t stride_kn, int32_t stride_vb, int32_t stride_vh, int32_t stride_vn, int32_t stride_ob, int32_t stride_oh, int32_t stride_om, int32_t num_heads, int32_t num_kv_heads, int32_t total_seq_len, int32_t past_seq_len"
    target1 = "SparseAttentionParams& params"
    source2 = "stream, out, Q, K, V, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_ob, stride_oh, stride_om, num_heads, num_kv_heads, total_seq_len, past_seq_len"
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
    print("mv sparse_attention_kernel_* ../")

    # Clean up
    print("cd ..")
    print("rm compile.py")
    print(f"rm -rf {out_dir}")

    print("echo Done")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compile block sparse attention triton kernel")
    parser.add_argument("--dtype", default="fp16", type=str, choices=["fp16", "bf16"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    generate_triton_compile_shell_script(args.dtype)
