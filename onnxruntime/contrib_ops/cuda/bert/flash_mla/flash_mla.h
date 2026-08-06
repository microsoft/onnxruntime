/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * reference: https://github.com/deepseek-ai/FlashMLA
 */

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_mla_params
{
    using index_t = int64_t;

    int b, seqlen_q, d, d_v;
    int h, h_h_k_ratio, ngroups;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int* __restrict__ cu_seqlens_k;

    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;
    void* __restrict__ o_ptr;
    void* __restrict__ softmax_lse_ptr;

    float* __restrict__ descale_q_ptr = nullptr;
    float* __restrict__ descale_k_ptr = nullptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;

    int* __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    int* __restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int* __restrict__ num_splits_ptr;

    void* __restrict__ softmax_lseaccum_ptr;
    void* __restrict__ oaccum_ptr;
};

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params& params, cudaStream_t stream);

struct Mla_metadata_params
{
    int* __restrict__ seqlens_k_ptr;
    int* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};

void get_mla_metadata_func(Mla_metadata_params& params, cudaStream_t stream);
