// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace fpA_intB_gemm {

enum class KernelType
{
    FP16Int8Groupwise,
    FP16Int4Groupwise,
    FP16Int8PerChannel,
    FP16Int4PerChannel
    // BF16Int8Groupwise,
    // BF16Int4Groupwise,
    // BF16Int8PerChannel,
    // BF16Int4PerChannel
};

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

void kernel_launcher(int arch, Params& params, cudaStream_t s);

bool is_supported(int arch, KernelType kernel_type);

template<bool is_zero_point_int4_packed, typename T, typename Z>
void launch_scaled_zero_point_kernel(
    cudaStream_t stream,
    const T* scale,
    const Z* zero_point,
    T* transposed_scale,
    T* scaled_zero_point,
    int n, int k, float default_zero_point);


// unpack int4 packed transposed weight of shape (n, k/2) to int8 weight of shape (k, n)
void unpack_int4_packed_transposed_tensor_to_int8_cuda(cudaStream_t stream, void* unpacked_weight, const void* weight, int n, int k);

}  // namespace fpA_intB_gemm
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
