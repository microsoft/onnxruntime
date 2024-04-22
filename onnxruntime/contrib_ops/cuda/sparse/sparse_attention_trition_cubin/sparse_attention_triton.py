# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
from itertools import product

import triton
import triton.language as tl


# yapf: disable
@triton.jit
def block_sparse_attention_kernel(
    out,  # output [B, H, M, D]. Note that B is batch_size, H is num_heads, M is q_seq_len, and D is head_size
    Q,  # query [B, H, M, D]
    K,  # key [B, H, N, D]. Note that N is k_seq_len (i.e. total_seq_len), k_num_heads need to expand to num_heads
    V,  # value [B, H, N, D]
    layout_csr_row_indices,  # block mask CSR format. Shape is [H, num_rows + 1] where num_rows = max_seq_len / BLOCK_M
    layout_csr_col_indices,  # block mask CSR format. Shape is [H, num_rows * num_cols] where num_cols = max_seq_len / BLOCK_N
    layout_csr_row_stride_h,  # stride per head for csr_row_indices, i.e. num_rows + 1
    layout_csr_col_stride_h,  # stride per head for csr_col_indices, i.e. num_rows * num_cols
    num_layout,  # number of sparse layout
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    num_heads,
    total_seq_len,
    past_seq_len,
    BLOCK_M: tl.constexpr,  # block size for q_seq_len
    EVEN_M: tl.constexpr,  # whether q_seq_len % BLOCK_M == 0
    BLOCK_N: tl.constexpr,  # block size for k_seq_len
    EVEN_N: tl.constexpr,  # whether k_seq_len % BLOCK_N == 0
    BLOCK_D: tl.constexpr,  # block size for D
    NUM_D_BLOCKS: tl.constexpr,  # number of data blocks =  D / BLOCK_D
):
    q_seq_len = total_seq_len - past_seq_len

    # Grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads]
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_h = off_bh % num_heads
    off_b = off_bh // num_heads
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :]
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :]

    # Initialize pointers to query, key, value
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    if NUM_D_BLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load q: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(q_ptrs)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
        if NUM_D_BLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_D, mask=offs_m[:, None] < q_seq_len)

    layout_h = off_h % num_layout
    layout_ptr = layout_csr_row_indices + layout_h * layout_csr_row_stride_h + start_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + 1).to(tl.int32)

    # Loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # -- compute qk ----
        if EVEN_N:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_n[None, :] + start_n < total_seq_len)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_D_BLOCKS >= 2:
            if EVEN_N:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_D)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn + BLOCK_D,
                    mask=offs_n[None, :] + start_n < total_seq_len,
                )
            qk += tl.dot(q2, k)

        qk *= softmax_scale
        qk += tl.where(offs_m[:, None] + past_seq_len >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        if NUM_D_BLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)
        # update acc
        if EVEN_N:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < total_seq_len)
        acc += tl.dot(p, v)

        if NUM_D_BLOCKS >= 2:
            if EVEN_N:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_D)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn + BLOCK_D,
                    mask=offs_n[:, None] + start_n < total_seq_len,
                )
            acc2 += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_o = off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    out_ptrs = out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
    if NUM_D_BLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_D, acc2, mask=offs_m[:, None] < q_seq_len)

# Generate the command lines to run in Linux to compile CUDA kernels.
# Example to use this script (Tested with triton 2.3.0 in Ubuntu 20.04):
#    python sparse_attention_triton.py | sh
def generate_triton_compile_shell_script(dtype = "fp16"):
    assert(dtype in ["fp16", "bf16"])
    print("export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)")
    print("export ARCH=\"$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)\"")
    print("export SM=$(echo $ARCH | sed -e 's/\\.//g')")
    out_dir = f"trition_cubin_{dtype}"
    print(f"rm -rf {out_dir}")
    print(f"mkdir -p {out_dir}")

    block_n_values = [64]
    block_d_values = [64]
    num_block_d_values = [2]
    even_m_values = [True, False]
    even_n_values = [True, False]

    for block_n, block_d, num_blocks_d, even_m, even_n in product(
        block_n_values, block_d_values, num_block_d_values, even_m_values, even_n_values
    ):
        block_m_values = [16, block_n] if block_n != 16 else [block_n]
        for block_m in block_m_values:
            scalar_params = "i32,i32,i32,fp32,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32:16,i32,i32"
            sig = f"*{dtype}:16,*{dtype}:16,*{dtype}:16,*{dtype}:16,*i32:16,*i32:16,{scalar_params},{block_m},{int(even_m)},{block_n},{int(even_n)},{block_d},{num_blocks_d}"
            prefix = "python ${TRITON_ROOT}/triton/tools/compile.py sparse_attention_triton.py"
            filename = "fbsa_sm${SM}_" + f"{dtype}_m{block_m}_{int(even_m)}_n{block_n}_{int(even_n)}_d{block_d}_{num_blocks_d}"
            name="fbsa_sm${SM}_" + f"{dtype}"
            num_warps = max(1, 2 ** int(math.log2(min(block_m, block_n, block_d) / 16)))
            num_stages = 2
            print(f"{prefix} -n block_sparse_attention_kernel -o {out_dir}/{filename} --out-name {name} -w {num_warps} -ns {num_stages} -s \"{sig}\" -g \"(total_seq_len - past_seq_len + {block_m} - 1) / {block_m}, batch_size * num_heads, 1\"")

    print(f"cd {out_dir}")
    print("python ${TRITON_ROOT}/triton/tools/link.py " + f"fbsa_*.h -o sparse_attention_api_{dtype}" + "_sm${SM}")

    # add batch_size parameter which is required for launching but not used in kernel.
    print("for file in *.c; do sed -i 's/int32_t past_seq_len/int32_t past_seq_len, int32_t batch_size/g'  \"$file\"; done")
    #print("for file in *.c; do sed -i 's/&past_seq_len/\\&past_seq_len\\, \\&batch_size/g'  \"$file\"; done")
    #print("for file in *.c; do sed -i 's/\\[25\\]/\\[26\\]/g'  \"$file\"; done")
    print(f"sed -i 's/ past_seq_len)/ past_seq_len, batch_size)/g'  \"sparse_attention_api_{dtype}" + "_sm${SM}.c\"")
    print("for file in *.h; do sed -i 's/int32_t past_seq_len/int32_t past_seq_len, int32_t batch_size/g'  \"$file\"; done")
    # remove a line like if(gX * gY * gZ > 0) so that kernel always have return value
    print("for file in *.c; do sed -i '/gX \\* gY \\* gZ/d'  \"$file\"; done")

    # Remove signature hash from filename since we use same signature for all kernels except constants and we add constants to the filename.
    print("for file in *.h; do mv -- \"$file\" \"$(echo $file | cut -f 1 -d '.').h\"; done")
    print("for file in *.c; do mv -- \"$file\" \"$(echo $file | cut -f 1 -d '.').c\"; done")

    # rename *.c to *.cc
    print("for file in *.c; do mv -- \"$file\" \"${file%.c}.cc\"; done")

if __name__ == "__main__":
    generate_triton_compile_shell_script("fp16")

# yapf: enable
