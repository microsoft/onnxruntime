//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <array>
#include <vector>

#include <unordered_map>

#include "mlasi_kleidiai.h"

#include "kai_ukernel_interface.h"

#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.h"


const KaiF32IMatmulKernel& imatmul_conv = GetKleidiAIF32IMatmulUKernel();

// Bound for the per-thread packed LHS scratch buffer used by the chunked IGEMM path.
// This is a heuristic budget (chosen because 2 MiB is a common L2 cache size) which limits
// temporary memory growth while keeping chunks large enough to be worth the KAI packing overhead.
// If one m_step block already exceeds the budget, processing is done one m_step at a time.
constexpr size_t kMaxLhsPackedChunkBytes = 2 * 1024 * 1024;

// Left-hand-side (input indirection) cache key
struct LhsCacheKey {
    size_t ci, ih, iw;
    size_t padding, sh, sw;
    size_t kh, kw;
    size_t dilationh, dilationw;

    bool operator==(const LhsCacheKey& other) const {
        return ci == other.ci && ih == other.ih && iw == other.iw &&
               padding == other.padding && sh == other.sh && sw == other.sw &&
               kh == other.kh && kw == other.kw &&
               dilationh == other.dilationh && dilationw == other.dilationw;
    }
};

namespace std {
    // Specialize hash type for cache keys and do it within namespace std.
    // Doing this allows standard containers like std::unordered_map to find
    // the appropriate hash function via template specialization, as ADL
    // (argument-dependent lookup) does not apply to std::hash.
    template<>
    struct hash<LhsCacheKey> {
        size_t operator()(const LhsCacheKey& k) const {
            return std::hash<size_t>()(k.ci) ^
                (std::hash<size_t>()(k.ih) << 1) ^
                (std::hash<size_t>()(k.iw) << 2) ^
                (std::hash<size_t>()(k.padding) << 3) ^
                (std::hash<size_t>()(k.sh) << 4) ^
                (std::hash<size_t>()(k.sw) << 5) ^
                (std::hash<size_t>()(k.kh) << 6) ^
                (std::hash<size_t>()(k.kw) << 7) ^
                (std::hash<size_t>()(k.dilationh) << 8) ^
                (std::hash<size_t>()(k.dilationw) << 9);
        }
    };

}


static constexpr size_t ComputeKernelSize(const size_t D, const size_t K) {
    // D - dilation size
    // K - kernel size

    // D*S scale 1D kernel dimension by dilation factor
    // (D-1) remove affect of dilation scaling at kernel end
    return (D*K) - (D - 1);
}

static constexpr size_t ComputeConvOutSize(const size_t L, const size_t K, const size_t P, const size_t S) {

    //With start + end padding

    //L - input size
    //K - kernel size
    //P - Padding size
    //S - stride size

    //Does the convolution compute one value or less ?
    if ( S > 0 && (L + 2*P) >= K) {
        // L-(K-1) standard convolution output size is L-(K-1) for a step size of 1 with no padding
        // (2*P) 1D start and end padding
        // (L+2*P)-(K-1) the 1D length of convolution result for a kernel step size of 1
        // /S apply the kernel step
        return (((L - K) + (2 * P)) / S) + 1;
    }
    return 0;
}

static inline void CopyChannelBlock(float* dst, const float* src, size_t channels) {
    if (channels == 1) {
        *dst = *src;
        return;
    }

    std::memcpy(dst, src, channels * sizeof(float));
}

static size_t ComputeMlasWorkingBufferSize(const size_t co,
                                           const size_t ih, const size_t iw,
                                           const size_t kh, const size_t kw,
                                           const size_t dilationh, const size_t dilationw,
                                           const size_t sh, const size_t sw,
                                           const size_t padding) {
    // dimensions of dilated kernel
    const auto d_kh = ComputeKernelSize(dilationh, kh);
    const auto d_kw = ComputeKernelSize(dilationw, kw);

    const auto m = ComputeConvOutSize(ih, d_kh, padding, sh) *
                   ComputeConvOutSize(iw, d_kw, padding, sw);

    return m * co;
}

static bool CheckCapabilitiesSme(const MLAS_CONV_PARAMETERS* Parameters) {
    // Grouped support in this override is only implemented for channels-last
    // layout. The generic grouped path still assumes contiguous per-group CHW.
    if (Parameters->GroupCount > 1 && !Parameters->ChannelsLast) {
        return false;
    }

    if (!MlasConvSupportsDenseChannelsLast2DFloatKernel(
            Parameters->Dimensions,
            Parameters->BatchCount,
            Parameters->GroupCount,
            Parameters->InputShape,
            Parameters->KernelShape,
            Parameters->DilationShape,
            Parameters->Padding,
            Parameters->StrideShape,
            Parameters->FilterCount,
            Parameters->Beta)) {
        KLEIDIAI_DEBUG_LOG("CheckCapabilitiesSme returning false on shared capability checks.");
        return false;
    }

    const auto route_selection = ArmKleidiAI::SelectConvRoute(Parameters);
    const auto route = route_selection.route;

    if (route == ArmKleidiAI::ConvRoute::IGemm) {
        // ensure LHS packed buffer size is non-zero
        const size_t d_kh = route_selection.effective_kernel_h;
        const size_t d_kw = route_selection.effective_kernel_w;
        const size_t m_step = imatmul_conv.ukernel.get_m_step();

        const size_t bytes_per_m_step = kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme(
            m_step, d_kh * d_kw, Parameters->InputChannels);

        if (bytes_per_m_step == 0) {
            KLEIDIAI_DEBUG_LOG("CheckCapabilitiesSME returning false on zero LHS packed size");
            return false;
        }
        return true;
    }

    if (route == ArmKleidiAI::ConvRoute::SGemmFallback) {
        KLEIDIAI_DEBUG_LOG("CheckCapabilitiesSme returning false to prefer SGEMM-backed conv path.");
    } else {
        KLEIDIAI_DEBUG_LOG("CheckCapabilitiesSme returning false on functional or optimization checks.");
    }
    return false;
}

//General purpose axis swapping
static auto Transpose4D(std::array<const size_t,4> shape_in,
                        const float* in,
                        std::array<const size_t,4> permute) {

    std::array<size_t, 4> shape_out{shape_in[permute[0]],
                                   shape_in[permute[1]],
                                   shape_in[permute[2]],
                                   shape_in[permute[3]]};

    assert((shape_in[0] * shape_in[1] * shape_in[2] * shape_in[3]) ==
           (shape_out[0] * shape_out[1] * shape_out[2] * shape_out[3]));
    assert(permute[0] < 4 && permute[1] < 4 && permute[2] < 4 && permute[3] < 4);

    const size_t get_stride[] {shape_in[1] * shape_in[2] * shape_in[3], shape_in[2] * shape_in[3], shape_in[3]};
    auto get = [get_stride,in](const std::array<size_t, 4>& el) {
        return in[el[0] * get_stride[0] +
                  el[1] * get_stride[1] +
                  el[2] * get_stride[2] +
                  el[3]];
    };

    auto out_ = std::make_unique<float[]>(shape_in[0] * shape_in[1] * shape_in[2] * shape_in[3]);
    auto out = out_.get();

    const size_t set_stride[]{shape_out[1] * shape_out[2] * shape_out[3], shape_out[2] * shape_out[3], shape_out[3]};
    auto set = [set_stride,out](const std::array<size_t, 4>& el, float v) {
        out[el[0] * set_stride[0] +
            el[1] * set_stride[1] +
            el[2] * set_stride[2] +
            el[3]] = v;
    };

    std::array<size_t, 4> shape;
    for (shape[0] = 0; shape[0] < shape_in[0]; ++shape[0]) {
        for (shape[1] = 0; shape[1] < shape_in[1]; ++shape[1]) {
            for (shape[2] = 0; shape[2] < shape_in[2]; ++shape[2]) {
                for (shape[3] = 0; shape[3] < shape_in[3]; ++shape[3]) {
                    set({shape[permute[0]], shape[permute[1]], shape[permute[2]], shape[permute[3]]}, get(shape));
                }
            }
        }
    }

    return out_;
}

//nchw to nhwc specific axis swapping
static std::unique_ptr<float[]> NChwToNhwc(const size_t n,
                                           const size_t c,
                                           const size_t h,
                                           const size_t w,
                                           const float* RESTRICT in,
                                           const size_t dilationh=1,
                                           const size_t dilationw=1,
                                           const bool zero_fill=false,
                                           MLAS_THREADPOOL* ThreadPool=nullptr) {

    const auto d_h = ComputeKernelSize(dilationh, h);
    const auto d_w = ComputeKernelSize(dilationw, w);

    auto t = std::make_unique<float[]>(n*d_h*d_w*c);
    if (zero_fill) {
        std::fill(&t.get()[0], &t.get()[n*d_h*d_w*c], 0.f);
    }

    if (dilationh > 1 || dilationw > 1 || n > 1) {
        const size_t get_strides[] {c*h*w,h*w,w};
        auto get = [get_strides,in](const std::array<size_t, 4>& el) {
            return in[el[0]*get_strides[0] +
                    el[1]*get_strides[1] +
                    el[2]*get_strides[2] +
                    el[3]];
        };

        const size_t set_strides[] {d_h*d_w*c,dilationh*d_w*c,dilationw*c};
        auto set = [set_strides](const std::array<size_t, 4>& el, float v, float* out) {
            out[el[0]*set_strides[0] +
                el[1]*set_strides[1] +
                el[2]*set_strides[2] +
                el[3]] = v;
        };

        MLAS_UNREFERENCED_PARAMETER(set);
        MLAS_UNREFERENCED_PARAMETER(get);

        auto out0 = t.get();
        for (size_t s0 = n; s0 > 0; --s0) {
            auto out1 = out0;
            for (size_t s1 = c; s1 > 0; --s1) {
                auto out2 = out1;
                for (size_t s2 = h; s2 > 0; --s2) {
                    float* RESTRICT out3 = out2;
                    size_t s3 = w;
                    for (; s3 > 4; s3 -= 4) {
                        auto vf32 = MlasLoadFloat32x4(in);
                        in += 4;
                        MlasStoreLaneFloat32x4<0>(out3,vf32);
                        out3 += set_strides[2];
                        MlasStoreLaneFloat32x4<1>(out3,vf32);
                        out3 += set_strides[2];
                        MlasStoreLaneFloat32x4<2>(out3,vf32);
                        out3 += set_strides[2];
                        MlasStoreLaneFloat32x4<3>(out3, vf32);
                        out3 += set_strides[2];
                    }
                    for (; s3 > 0; --s3) {
                        //set({s0,s2,s3,s1}, get({s0,s1,s2,s3}),t.get());
                        *out3 = *in++;
                        out3 += set_strides[2];
                    }
                    out2 += set_strides[1];
                }
                out1++;
            }
            out0 += set_strides[0];
        }
    } else {
        MlasTranspose(in, t.get(), c, d_h*d_w, ThreadPool);
    }

    return t;
}

size_t
MLASCALL
ArmKleidiAI::MlasConvSymmetricChannelsLast2DFloatPackWSize(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape) {
    const auto d_kh = ComputeKernelSize(static_cast<size_t>(DilationShape[0]), static_cast<size_t>(KernelShape[0]));
    const auto d_kw = ComputeKernelSize(static_cast<size_t>(DilationShape[1]), static_cast<size_t>(KernelShape[1]));
    return kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(FilterCount, d_kh * d_kw,
                                                                                InputChannels);
}

void
MLASCALL
ArmKleidiAI::MlasConvSymmetricChannelsLast2DFloatPackW(
    size_t FilterCount,
    size_t InputChannels,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    size_t GroupCount,
    const float* Filter,
    const float* Bias,
    void* PackedFilter,
    size_t PackedFilterGroupStride,
    MLAS_THREADPOOL* ThreadPool) {
    const size_t kh = static_cast<size_t>(KernelShape[0]);
    const size_t kw = static_cast<size_t>(KernelShape[1]);
    const size_t dilationh = static_cast<size_t>(DilationShape[0]);
    const size_t dilationw = static_cast<size_t>(DilationShape[1]);
    const auto d_kh = ComputeKernelSize(dilationh, kh);
    const auto d_kw = ComputeKernelSize(dilationw, kw);

    for (size_t group_idx = 0; group_idx < GroupCount; ++group_idx) {
        const float* weights = Filter + group_idx * FilterCount * InputChannels * kh * kw;
        const float* bias = Bias ? Bias + group_idx * FilterCount : nullptr;
        auto* packed_group = reinterpret_cast<std::byte*>(PackedFilter) + group_idx * PackedFilterGroupStride;

        // prepare mlas filter weights for kai rhs packing
        // dilated nhwc format
        auto nhwc = NChwToNhwc(FilterCount, InputChannels, kh, kw, weights, dilationh, dilationw, true, ThreadPool);

        //t_weights[d_kh][d_kw][ci][co] = nhwc[co][d_kh][d_kw][ci]
        auto t_weights = Transpose4D({FilterCount, d_kh, d_kw, InputChannels}, &nhwc[0], {1, 2, 3, 0});

        std::vector<float> bias_copy;
        if (bias) {
            bias_copy.assign(bias, bias + FilterCount);
        } else {
            bias_copy.resize(FilterCount, 0.0f);
        }

        KLEIDIAI_KERNEL_LOG("kai_run_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme"
                            << " N=" << FilterCount << " k_chunk_count=" << (d_kh * d_kw)
                            << " k_chunk_length=" << InputChannels
                            << " rhs_stride_row=" << (FilterCount * sizeof(float)));
        kai_run_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
            FilterCount, d_kh * d_kw, InputChannels, FilterCount * sizeof(float), &t_weights[0], bias_copy.data(),
            packed_group);
    }
}

static std::shared_ptr<const void*[]> LhsPtrFill(const size_t ci, const size_t ih, const size_t iw,
                                                 const size_t kh, const size_t kw, size_t sh, size_t sw,
                                                 const size_t padding,
                                                 const float* pad_ptr) {
    size_t check_filled{0};

    const auto m = ComputeConvOutSize(ih, kh, padding, sh) * ComputeConvOutSize(iw, kw, padding, sw);

    const auto m_step = imatmul_conv.ukernel.get_m_step();

    const auto lhs_ptrs_k = kh * kw;
    const auto lhs_ptrs_m = m_step * MlasDivRoundup(m, m_step);
    auto lhs_ptrs = std::shared_ptr<const void*[]>(new const void*[lhs_ptrs_k * lhs_ptrs_m],
                                                std::default_delete<const void*[]>());

    // Initialize all padding entries. For partial tiles (m < m_step),
    // the kai LHS packing kernel may still read pointer entries beyond the logically
    // filled 'm' positions. Leaving these uninitialized can cause non-deterministic
    // reads and corrupt packed LHS data.
    auto lhs_ptrs_ = lhs_ptrs.get();
    std::fill(lhs_ptrs_, lhs_ptrs_ + (lhs_ptrs_k * lhs_ptrs_m), reinterpret_cast<const void*>(&pad_ptr[0]));

    auto ih_out_size = ComputeConvOutSize(ih, kh, padding, 1);
    auto iw_out_size = ComputeConvOutSize(iw, kw, padding, 1);

    auto ptrs_offset = [lhs_ptrs_m,lhs_ptrs_k, m_step](size_t k, size_t m) {
        //(m/m_step,transpose(m_step,k)
        auto offset {((m/m_step) * lhs_ptrs_k * m_step) + (k*m_step) + (m%m_step)};
        assert(offset < (lhs_ptrs_k * lhs_ptrs_m));

        MLAS_UNREFERENCED_PARAMETER(lhs_ptrs_m);

        return offset;
    };

    auto pixel_offset = [ih, iw, ci, pad_ptr, padding](size_t h, size_t w) {
        if (h < padding) {
            return reinterpret_cast<size_t>(&pad_ptr[0]);
        }
        h -= padding;

        if (w < padding) {
            return reinterpret_cast<size_t>(&pad_ptr[0]);
        }
        w -= padding;

        if ((h >= ih) || (w >= iw)) {
            return reinterpret_cast<size_t>(&pad_ptr[0]);
        }

        auto offset{h * iw * ci + w * ci};
        assert(offset < (ih*iw*ci));
        return offset*sizeof(float);
    };

    size_t m_{0};
    for (size_t ih_ = 0; ih_ < ih_out_size; ih_ += sh) {
        for (size_t iw_ = 0; iw_ < iw_out_size; iw_ += sw, ++m_) {
            size_t k_{0};
            for (size_t kh_ = 0; kh_ < kh; ++kh_) {
                for (size_t kw_ = 0; kw_ < kw; ++kw_) {
                    lhs_ptrs_[ptrs_offset(k_, m_)] = reinterpret_cast<void*>(pixel_offset(ih_+kh_, iw_+kw_));
                    k_++; check_filled++;
                }
            }
        }
    }

    assert(check_filled == (lhs_ptrs_k * m));
    MLAS_UNREFERENCED_PARAMETER(check_filled);

    return lhs_ptrs;
}

static const float* GetOrCreatePadDataSme(const size_t ci) {
    size_t padsize = 256;
    if (ci > padsize) {
        padsize = MlasDivRoundup(ci, padsize) * padsize;
    }

    // pad_ptr must be at least 'ci' floats for padding pixels.
    // The buffer is grow-only and zero-initializes newly-grown elements.
    thread_local std::vector<float> pad_ptr;
    if (pad_ptr.size() < padsize) {
        pad_ptr.resize(padsize, 0.f);
    }

    return pad_ptr.data();
}

static std::shared_ptr<const void*[]> GetOrCreateLhsPtrTableSme(const size_t ci, const size_t ih, const size_t iw,
                                                                const size_t kh, const size_t kw, const size_t sh,
                                                                const size_t sw, const size_t padding, const float* pad_ptr) {
    // LhsPtrFill stores geometry offsets only; the current input base is supplied when packing.
    LhsCacheKey key = {
        ci, ih, iw,
        padding, sh, sw,
        kh, kw,
        1, 1
    };

    // Cache of computed LHS pointer offsets. thread_local to prevent interference from parallel sessions.
    // Entries include pointers to the pad buffer for out-of-bounds pixels; if the grow-only pad buffer
    // reallocates, erase the old group so stale pointer tables cannot reference the previous allocation.
    using LhsPtrsCache = std::unordered_map<LhsCacheKey, std::shared_ptr<const void*[]>>;
    thread_local std::unordered_map<const float*, LhsPtrsCache> lhs_ptrs_cache_by_pad;
    thread_local const float* last_pad_ptr = nullptr;

    const float* cur_pad_ptr = pad_ptr;
    if (last_pad_ptr != nullptr && last_pad_ptr != cur_pad_ptr) {
        lhs_ptrs_cache_by_pad.erase(last_pad_ptr);
    }
    last_pad_ptr = cur_pad_ptr;

    auto& lhs_ptrs_cache = lhs_ptrs_cache_by_pad[cur_pad_ptr];

    std::shared_ptr<const void*[]> lhs_ptrs;
    if (auto found = lhs_ptrs_cache.find(key); found != lhs_ptrs_cache.end()) {
        lhs_ptrs = found->second;
    } else {
        lhs_ptrs = LhsPtrFill(ci, ih, iw, kh, kw, sh, sw, padding, pad_ptr);
        lhs_ptrs_cache[key] = lhs_ptrs;
    }
    return lhs_ptrs;
}

static void ConvolveSme(const size_t co, //channels out
                        const size_t ci,  //channels in
                        const size_t ih,  //image height
                        const size_t iw,  //image width
                        const size_t kh,  //kernel height
                        const size_t kw,  //kernel width
                        const size_t sh,  //kernel stride height
                        const size_t sw,  //kernel stride width
                        const size_t dilationh, //kernel dilation stride
                        const size_t dilationw, //kernel dilation stride
                        const size_t padding,   //padding size
                        const size_t groups,       //number of filter groups
                        const float* weights,      //kernel weights [co,ci,ih,iw]
                        const float* bias,         //kernel biases
                        const std::byte* packed_rhs,
                        const size_t packed_rhs_group_stride,
                        const float* in,           //in image data
                        float* out,                //out image data
                        float* tmp_mlas_aligned,   //intermediate buffer if we need to perform a transpose
                        bool input_is_channels_last,
                        MLAS_THREADPOOL* ThreadPool) {

    //dilation expands the logical kernel shape; RHS packing masks the inserted unused weights.
    //this way, compute corrected dimensions of dilated kernel
    const auto d_kh = ComputeKernelSize(dilationh, kh);
    const auto d_kw = ComputeKernelSize(dilationw, kw);

    //run igemm based convolution
    const auto m = ComputeConvOutSize(ih, d_kh, padding, sh) *
                   ComputeConvOutSize(iw, d_kw, padding, sw);

    size_t n_step = imatmul_conv.ukernel.get_n_step();
    size_t m_step = imatmul_conv.ukernel.get_m_step();

    // Query the packed LHS buffer size for exactly one m_step block.
    const size_t bytes_per_m_step = kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme(
        m_step, d_kh * d_kw, ci);
    // Sanity check to ensure data passed to the function is valid and won't cause a zero division error.
    assert(bytes_per_m_step != 0);

    // tile iteration dimensions
    std::array<size_t,3> dim;
    dim[0] = 1;                          // B
    dim[1] = MlasDivRoundup(m, m_step);  // M
    dim[2] = MlasDivRoundup(co, n_step); // N

    //Minimize the kernel call count for the number of available threads
    auto required_tiles = std::min(static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)), dim[0]*dim[1]*dim[2]);

    //scale required tiles over available tile processors
    dim[1] = MlasDivRoundup(required_tiles * dim[1], dim[1] * dim[2]);
    dim[2] = MlasDivRoundup(required_tiles * dim[2], dim[1] * dim[2]);

    //compute new step sizes
    size_t m_tile_step = m_step * MlasDivRoundup(MlasDivRoundup(m, dim[1]), m_step);
    size_t n_tile_step = n_step * MlasDivRoundup(MlasDivRoundup(co, dim[2]), n_step);

    //update tile iterations
    dim[1] = MlasDivRoundup(m, m_tile_step);
    dim[2] = MlasDivRoundup(co, n_tile_step);

    const bool grouped_channels_last = input_is_channels_last && groups > 1;
    const size_t input_channels_total = ci * groups;
    const size_t output_channels_total = co * groups;
    const float* input_base = in;
    float* output_base = out;
    std::vector<float> input_group_buffer;
    if (grouped_channels_last) {
        input_group_buffer.resize(ih * iw * ci);
    }

    for (size_t g = 0; g < groups; ++g) {
        const float* input_group = in;
        if (grouped_channels_last) {
            for (size_t pixel = 0; pixel < ih * iw; ++pixel) {
                const float* src = input_base + pixel * input_channels_total + g * ci;
                CopyChannelBlock(input_group_buffer.data() + pixel * ci, src, ci);
            }
            input_group = input_group_buffer.data();
        }

        auto result = out;
        const bool need_transpose = (!input_is_channels_last) && (co > 1);
        if (need_transpose || grouped_channels_last) {
            result = tmp_mlas_aligned;
        }

        // LHS packing data
        const float* pad_data = GetOrCreatePadDataSme(ci);
        std::unique_ptr<float[]> nhwc_holder;
        const float* activation_src = input_group;
        if (!input_is_channels_last) {
            nhwc_holder = NChwToNhwc(1, ci, ih, iw, input_group, 1, 1, false, ThreadPool);
            activation_src = nhwc_holder.get();
        }
        auto lhs_ptrs = GetOrCreateLhsPtrTableSme(ci, ih, iw, d_kh, d_kw, sh, sw, padding, pad_data);

        // RHS packing data
        const std::byte* rhs_data = packed_rhs ? packed_rhs + g * packed_rhs_group_stride : nullptr;
        std::unique_ptr<std::byte[]> rhs_storage;
        if (rhs_data == nullptr) {
            const std::array<int64_t, 2> kernel_shape{static_cast<int64_t>(kh), static_cast<int64_t>(kw)};
            const std::array<int64_t, 2> dilation_shape{static_cast<int64_t>(dilationh), static_cast<int64_t>(dilationw)};
            const size_t packed_size =
                ArmKleidiAI::MlasConvSymmetricChannelsLast2DFloatPackWSize(co, ci, kernel_shape.data(),
                                                                           dilation_shape.data());
            rhs_storage = std::make_unique<std::byte[]>(packed_size);
            ArmKleidiAI::MlasConvSymmetricChannelsLast2DFloatPackW(co, ci, kernel_shape.data(), dilation_shape.data(), 1,
                                                                   weights, bias, rhs_storage.get(), packed_size,
                                                                   ThreadPool);
            rhs_data = rhs_storage.get();
        }

        MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(dim[0] * dim[1] * dim[2]), [&](ptrdiff_t tid) {
            //compute B,M,N index from iteration index
            //ptrdiff_t BIdx = tid / (dim[1] * dim[2]);
            const size_t m_idx = static_cast<size_t>((tid % (dim[1] * dim[2])) / dim[2]);
            const size_t n_idx = static_cast<size_t>((tid % (dim[1] * dim[2])) % dim[2]);

            // Get rhs tile, B
            const size_t rhs_packed_offset =
                imatmul_conv.ukernel.get_rhs_packed_offset(n_idx * n_tile_step, d_kh * d_kw, ci);

            auto b_tile = reinterpret_cast<const void*>(
                rhs_data + rhs_packed_offset
            );

            // Calculate lhs tile chunks
            auto tile_size_n = (n_idx + 1) * n_tile_step > co ? (co - n_idx * n_tile_step) : n_tile_step;

            // Compute the starting row index of the current M tile
            size_t tile_m_start = m_idx * m_tile_step;
            // Actual number of rows in this tile (may be smaller for the last tile)
            size_t tile_m_size = std::min(m_tile_step, m - tile_m_start);

            // Determine how many rows we can pack in one chunk.
            const size_t max_m_step_chunks = kMaxLhsPackedChunkBytes / bytes_per_m_step;
            // If bytes_per_m_step exceeds kMaxLhsPackedChunkBytes, process m_step at a time
            size_t m_chunk = max_m_step_chunks == 0 ? m_step : m_step * max_m_step_chunks;

            // Do not exceed the number of rows available in this tile
            m_chunk = std::min<size_t>(tile_m_size, m_chunk);

            // Compute the exact packed buffer size for m_chunk rows.
            const size_t lhs_buffer_bytes = kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme(
                m_chunk, d_kh * d_kw, ci);

            // Thread-local grow-only reusable buffer for LHS packing
            thread_local std::vector<std::byte> lhs_buffer;
            if (lhs_buffer.size() < lhs_buffer_bytes) {
                lhs_buffer.resize(lhs_buffer_bytes);
            }
            auto* lhs = lhs_buffer.data();

            // Interpret the packed LHS buffer as the input A tile
            // for the matrix multiplication kernel.
            auto a_tile = reinterpret_cast<const float*>(lhs);

            for (size_t m_base = 0; m_base < tile_m_size; m_base += m_chunk) {
                // Actual number of rows processed in this iteration.
                // The last chunk may be smaller than m_chunk.
                const size_t tile_size_m = std::min(m_chunk, tile_m_size - m_base);

                // Pack tile_size_m rows of the LHS matrix into a temporary buffer.
                kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme(tile_size_m,
                                                            d_kh * d_kw,
                                                            ci,
                                                            lhs_ptrs.get() + (tile_m_start + m_base) * d_kh * d_kw,
                                                            reinterpret_cast<size_t>(activation_src),
                                                            reinterpret_cast<const void*>(pad_data),
                                                            lhs);

                // Get result tile, C
                auto c_tile = &reinterpret_cast<std::byte*>(result)[
                    (tile_m_start + m_base) * co * sizeof(float) +
                    n_idx * n_tile_step * sizeof(float)];

                KLEIDIAI_KERNEL_LOG(imatmul_conv.name
                    << " M=" << tile_size_m << " N=" << tile_size_n
                    << " k_chunk_count=" << (d_kh * d_kw) << " k_chunk_length=" << ci);

                imatmul_conv.ukernel.run_imatmul(tile_size_m, tile_size_n, d_kh * d_kw, ci, a_tile, b_tile,
                                                 c_tile, co * sizeof(float), -std::numeric_limits<float>::max(),
                                                 std::numeric_limits<float>::max());
            }
        });

        if (grouped_channels_last) {
            for (size_t pixel = 0; pixel < m; ++pixel) {
                float* dst = output_base + pixel * output_channels_total + g * co;
                const float* src = result + pixel * co;
                CopyChannelBlock(dst, src, co);
            }
        }

        if (need_transpose) {
            //Note: this could be absorbed into post conv activation
            MlasTranspose(tmp_mlas_aligned, out, m, co, ThreadPool);
        }

        if (!grouped_channels_last) {
            in += ci * ih * iw;
            out += m * co;
        }
        weights += co * ci * kh * kw;
        if(bias){
            bias += co;
        }
    }
}

bool MLASCALL
ArmKleidiAI::MlasConvPrepare(MLAS_CONV_PARAMETERS* Parameters,
                size_t Dimensions,
                size_t BatchCount,
                size_t GroupCount,
                size_t InputChannels,
                const int64_t* InputShape,
                const int64_t* KernelShape,
                const int64_t* DilationShape,
                const int64_t* Padding,
                const int64_t* StrideShape,
                const int64_t* OutputShape,
                size_t FilterCount,
                const MLAS_ACTIVATION* Activation,
                size_t* WorkingBufferSize,
                bool ChannelsLast,
                float Beta,
                MLAS_THREADPOOL* ThreadPool)
{
    // Check if the user wants to use KleidiAI
    if (Parameters->BackendKernelSelectorConfig && !Parameters->BackendKernelSelectorConfig->use_kleidiai) {
        KLEIDIAI_DEBUG_LOG("User explicitly disabled KleidiAI, returning false from MlasConvPrepare.");
        return false;
    }

    //Check dimensions before accessing
    if (Dimensions < 2) {
        return false;
    }

    Parameters->Activation = Activation;
    Parameters->Dimensions = Dimensions;
    Parameters->BatchCount = BatchCount;
    Parameters->GroupCount = GroupCount;
    Parameters->InputChannels = InputChannels;
    Parameters->ChannelsLast = ChannelsLast;
    Parameters->FilterCount = FilterCount;
    Parameters->Beta = Beta;

    size_t InputSize = 1;
    size_t OutputSize = 1;
    size_t K = InputChannels;

    for (size_t dim = 0; dim < Dimensions; dim++) {

        Parameters->InputShape[dim] = size_t(InputShape[dim]);
        Parameters->OutputShape[dim] = size_t(OutputShape[dim]);
        Parameters->KernelShape[dim] = size_t(KernelShape[dim]);
        Parameters->DilationShape[dim] = size_t(DilationShape[dim]);
        Parameters->Padding[dim] = size_t(Padding[dim]);
        Parameters->Padding[dim + Dimensions] = size_t(Padding[dim + Dimensions]);
        Parameters->StrideShape[dim] = size_t(StrideShape[dim]);

        InputSize *= Parameters->InputShape[dim];
        OutputSize *= Parameters->OutputShape[dim];
        K *= Parameters->KernelShape[dim];
    }

    Parameters->InputSize = InputSize;
    Parameters->OutputSize = OutputSize;
    Parameters->K = K;

    Parameters->ThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if(!CheckCapabilitiesSme(Parameters)){
        return false;
    }

    //Allocate an aligned buffer for MlasTranspose()
    *WorkingBufferSize = ComputeMlasWorkingBufferSize(Parameters->FilterCount,
                                                      Parameters->InputShape[0], Parameters->InputShape[1],
                                                      Parameters->KernelShape[0], Parameters->KernelShape[1],
                                                      Parameters->DilationShape[0], Parameters->DilationShape[1],
                                                      Parameters->StrideShape[0], Parameters->StrideShape[1],
                                                      Parameters->Padding[0]);
    return true;
}

bool
MLASCALL
ArmKleidiAI::MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
{
    // Check if the user wants to use KleidiAI
    if (Parameters->BackendKernelSelectorConfig && !Parameters->BackendKernelSelectorConfig->use_kleidiai) {
        KLEIDIAI_DEBUG_LOG("User explicitly disabled KleidiAI, returning false from MlasConv.");
        return false;
    }

    if(!CheckCapabilitiesSme(Parameters)){
        // Fallback to Default Mlas
        return false;
    };
    ConvolveSme(Parameters->FilterCount, Parameters->InputChannels,          // channel out, in
                Parameters->InputShape[0], Parameters->InputShape[1],         // image dimensions
                Parameters->KernelShape[0], Parameters->KernelShape[1],      // kernel dimensions
                Parameters->StrideShape[0], Parameters->StrideShape[1],      // kernel stride dimensions
                Parameters->DilationShape[0], Parameters->DilationShape[1],  // kernel dilation
                Parameters->Padding[0],                                      // image padding
                Parameters->GroupCount,                                      // filter groups
                Filter, Bias,
                reinterpret_cast<const std::byte*>(Parameters->PackedFilter),
                Parameters->PackedFilterGroupStride,
                Input, Output, WorkingBuffer, Parameters->ChannelsLast, ThreadPool);

    const bool grouped_channels_last = Parameters->ChannelsLast && Parameters->GroupCount > 1;
    const size_t activation_rows = grouped_channels_last ? Parameters->OutputSize : Parameters->FilterCount;
    const size_t activation_cols =
        grouped_channels_last ? Parameters->GroupCount * Parameters->FilterCount : Parameters->OutputSize;
    MlasActivation(Parameters->Activation, Output, nullptr, activation_rows, activation_cols, activation_cols);
    return true;
}
