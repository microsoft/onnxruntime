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

/**
 * ref to Fa3's SmemTranspose64x64:
 * https://github.com/Dao-AILab/flash-attention/blob/0823cf7b5d96499c1c79a4f64b1e256a035ba4b4/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L26
 */

#pragma once

template <int kBlockN, int kHeadDim, typename SmemLayoutK>
struct SmemTransposeFp8_64x64
{
    static_assert((kBlockN % 64 == 0) && (kHeadDim % 64 == 0));

    using Element = cutlass::float_e4m3_t;
    using TransposeShapeAtomV = Shape<_64, _64>;
    using SmemLayoutAtomV = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // for fp8 in-kernel transpose -- src layout
    using SmemLayoutDivideV = decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
    using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
    using FactoringShapeV
        = decltype(make_shape(SmemShapeLDSM{}, shape<1>(SmemLayoutDivideV{}), shape<2>(SmemLayoutDivideV{})));
    using SmemLayoutTransposeV = decltype(composition(SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

    // For fp8, this is the memory transpose.
    using SmemLayoutAtomVt = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtomVt{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    // for fp8 in-kernel transpose -- dst layout
    using SmemLayoutVtTrans = decltype(composition(
        SmemLayoutVt{}, make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1>{})));
    using SmemLayoutDivideVt = decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
    using FactoringShapeVt
        = decltype(make_shape(SmemShapeSTSM{}, shape<1>(SmemLayoutDivideVt{}), shape<2>(SmemLayoutDivideVt{})));
    using SmemLayoutTransposeVt = decltype(composition(SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));

    using ldsm_thread_shape = Shape<_4, _1, _8, _4>;
    using ldsm_value_shape = Shape<_2, _8, _2, _1>;
    using ldsm_value_stride = Stride<_2, _4, _1, _0>;
    using TiledCopyLDSM = decltype(make_tiled_copy(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<ldsm_thread_shape>{},
        Layout<ldsm_value_shape, ldsm_value_stride>{}));
    TiledCopyLDSM tiled_copy_ldsm;

    using stsm_thread_shape = Shape<_4, _1, _8, _4>;
    // using stsm_thread_stride = Stride<_1, _0, _4, _32>;
    using stsm_value_shape = Shape<_4, _4, _2, _1>;
    using stsm_value_stride = Stride<_1, _8, _4, _0>;

    using TiledCopySTSM = decltype(make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<stsm_thread_shape>{},
        Layout<stsm_value_shape, stsm_value_stride>{}));
    TiledCopySTSM tiled_copy_stsm;

    template <class SmemTensor, class SmemTensorOut>
    CUTLASS_DEVICE void transpose(SmemTensor&& s_in, SmemTensorOut&& s_out)
    {
        using namespace cute;

        auto tid = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        auto thr_copy_ldsm = tiled_copy_ldsm.get_thread_slice(tid);
        auto thr_copy_stsm = tiled_copy_stsm.get_thread_slice(tid);

        auto tXsX = thr_copy_ldsm.partition_S(s_in);
        auto tXrX = make_tensor<Element>(shape(tXsX));
        auto tXsX_out = thr_copy_stsm.partition_D(s_out);

        cute::copy(tiled_copy_ldsm, tXsX, tXrX);

        auto data = tXrX.data();
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size(tXrX); n += 8)
        {
            uint32_t* data_32bit = reinterpret_cast<uint32_t*>(&data[n]);
            auto upper = data_32bit[0];
            auto lower = data_32bit[1];
            data_32bit[0] = __byte_perm(upper, lower, 0x6420);
            data_32bit[1] = __byte_perm(upper, lower, 0x7531);
        }

        cute::copy(tiled_copy_stsm, tXrX, tXsX_out);
    }
};
