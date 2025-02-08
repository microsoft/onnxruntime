/*
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
 */

#pragma once
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/common.h"

namespace onnxruntime::llm
{
namespace kernels
{
namespace weight_only
{
struct FP16DetailsA
{
    using Type = half;
    using Type2 = half2;
    static constexpr int kElemBits = 16;
};

struct BF16DetailsA
{
    using Type = __nv_bfloat16;
    using Type2 = __nv_bfloat162;
    static constexpr int kElemBits = 16;
};

struct Int8DetailsW
{
    static constexpr int kElemBits = 8;
};

struct Int4DetailsW
{
    static constexpr int kElemBits = 4;
};

template <typename TypeDetailsA, typename TypeDetailsW, int TileSizeK>
struct ColumnMajor
{
    using DetailsA = TypeDetailsA;
    using DetailsW = TypeDetailsW;
    using AccessTypeA = float4;
    using AccessTypeW = int;
    static constexpr int kAccessSize = 128;
    static constexpr int kStepK = kAccessSize / TypeDetailsA::kElemBits;
    static constexpr int kTileSize = TileSizeK;
    static constexpr int kInterleave = 1;

    struct Mapper
    {
        __device__ __forceinline__ int operator()(int i)
        {
            return i;
        }
    };
};

template <typename TypeDetailsA, typename TypeDetailsW, int TileSizeK>
struct ColumnMajorInterleavedForHopper
{
    using DetailsA = TypeDetailsA;
    using DetailsW = TypeDetailsW;
    using AccessTypeA = float4;
    using AccessTypeW = int4;
    static constexpr int kAccessSize = 128;
    static constexpr int kStepK = kAccessSize / TypeDetailsW::kElemBits;
    static constexpr int kTileSize = TileSizeK;
    static constexpr int kInterleave = 1;

    static constexpr int kTypeFactor = 128 * 8 / (TileSizeK * TypeDetailsW::kElemBits);

    // constants for mapper
    static constexpr int kElementGroupSizeA = TileSizeK / 32;
    static constexpr int kElementGroupSizeW = kTypeFactor * kElementGroupSizeA;
    static constexpr int kGroupOffsetA = 4 * kElementGroupSizeA;

    struct Mapper
    {
        __device__ __forceinline__ int operator()(int i)
        {
            return i % kElementGroupSizeA + (i % kGroupOffsetA) / kElementGroupSizeA * kElementGroupSizeW
                + i / kGroupOffsetA * kElementGroupSizeA;
        }
    };
};

template <typename TypeDetailsA, typename TypeDetailsW, int TileSizeK>
struct ColumnMajorInterleaved
{
    using DetailsA = TypeDetailsA;
    using DetailsW = TypeDetailsW;
    using AccessTypeA = float4;
    using AccessTypeW = int4;
    static constexpr int kAccessSize = 128;
    static constexpr int kStepK = kAccessSize / TypeDetailsW::kElemBits;
    static constexpr int kTileSize = TileSizeK;
    static constexpr int kInterleave = 128 * 8 / (TileSizeK * TypeDetailsW::kElemBits);

    // constants for mapper
    static constexpr int kElementGroupSizeA = TileSizeK / 32;
    static constexpr int kElementGroupSizeW = kInterleave * kElementGroupSizeA;
    static constexpr int kGroupOffsetA = 4 * kElementGroupSizeA;

    struct Mapper
    {
        __device__ __forceinline__ int operator()(int i)
        {
            return i % kElementGroupSizeA + (i % kGroupOffsetA) / kElementGroupSizeA * kElementGroupSizeW
                + i / kGroupOffsetA * kElementGroupSizeA;
        }
    };
};

template <typename TypeDetailsA_, typename TypeDetailsW_, template <typename, typename, int> class LayoutDetails_,
    bool UseInterleavedConverter, int TileSizeK>
struct KernelDetails
{
    using TypeDetailsA = TypeDetailsA_;
    using TypeDetailsW = TypeDetailsW_;
    using LayoutDetails = LayoutDetails_<TypeDetailsA, TypeDetailsW, TileSizeK>;
    using AccessTypeA = typename LayoutDetails::AccessTypeA;
    using AccessTypeW = typename LayoutDetails::AccessTypeW;
    static constexpr int kWarpSize = 32;
    static constexpr int kStepK = LayoutDetails::kStepK;
    static constexpr int kAccessNumA = kStepK * TypeDetailsA::kElemBits / (sizeof(AccessTypeA) * 8);
    static constexpr int kAccessNumW = kStepK * TypeDetailsW::kElemBits / (sizeof(AccessTypeW) * 8);
    static constexpr int kInterleave = LayoutDetails::kInterleave;
    static constexpr int kThreadsPerInterleavedTile = LayoutDetails::kTileSize / kStepK;
    static constexpr int kElemsPerByteW = 8 / TypeDetailsW::kElemBits;
    static constexpr bool kUseInterleavedConverter = UseInterleavedConverter;
};

} // namespace weight_only
} // namespace kernels
} // namespace onnxruntime::llm
