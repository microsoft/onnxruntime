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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace onnxruntime::llm
{
namespace kernels
{
namespace weight_only
{
enum class KernelType
{
    FP16Int8Groupwise,
    BF16Int8Groupwise,
    FP16Int4Groupwise,
    BF16Int4Groupwise,
    FP16Int8PerChannel,
    BF16Int8PerChannel,
    FP16Int4PerChannel,
    BF16Int4PerChannel
};

template <KernelType KT>
struct kernel_type_traits;
#define KERNEL_TYPE_TRAITS_REGISTRY(KT, _isGroupwise, _isInt4)                                                         \
    template <>                                                                                                        \
    struct kernel_type_traits<KT>                                                                                      \
    {                                                                                                                  \
        static constexpr bool isGroupwise = _isGroupwise;                                                              \
        static constexpr bool isInt4 = _isInt4;                                                                        \
    };
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int8Groupwise, true, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int8Groupwise, true, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int4Groupwise, true, true);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int4Groupwise, true, true);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int8PerChannel, false, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int8PerChannel, false, false);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::FP16Int4PerChannel, false, true);
KERNEL_TYPE_TRAITS_REGISTRY(KernelType::BF16Int4PerChannel, false, true);
#undef KERNEL_TYPE_TRAITS_REGISTRY

struct Params
{
    using Pointer = void*;
    using ConstPointer = void const*;
    Pointer act;
    Pointer act_scale;
    Pointer weight;
    Pointer scales;
    Pointer zeros;
    Pointer bias;
    Pointer out;
    float alpha;
    int m;
    int n;
    int k;
    int groupsize;
    KernelType type;
    bool apply_alpha_in_advance;

    Params(ConstPointer _act, ConstPointer _act_scale, ConstPointer _weight, ConstPointer _scales, ConstPointer _zeros,
        ConstPointer _bias, Pointer _out, float _alpha, int _m, int _n, int _k, int _groupsize, KernelType _type,
        bool _apply_alpha_in_advance = false)
        : act(const_cast<Pointer>(_act))
        , act_scale(const_cast<Pointer>(_act_scale))
        , weight(const_cast<Pointer>(_weight))
        , scales(const_cast<Pointer>(_scales))
        , zeros(const_cast<Pointer>(_zeros))
        , bias(const_cast<Pointer>(_bias))
        , out(_out)
        , alpha(_alpha)
        , m(_m)
        , n(_n)
        , k(_k)
        , groupsize(_groupsize)
        , type(_type)
        , apply_alpha_in_advance(_apply_alpha_in_advance)
    {
    }
};
} // namespace weight_only
} // namespace kernels
} // namespace onnxruntime::llm
