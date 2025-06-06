// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include <cassert>
#include <map>
#include <iostream>
#include <algorithm>
#include "mlasi_kleidiai.h"

#include "kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.h"

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

static void CheckCapabilitiesSme(const MLAS_CONV_PARAMETERS* Parameters) {

    //functional checks - logically can the conv be performed
    if ((Parameters->Dimensions != 2) ||
        (Parameters->BatchCount != 1) ||
        (Parameters->Beta != 0.f) ||
        (Parameters->Padding[0] != Parameters->Padding[1]) ||
        (Parameters->Padding[0] != Parameters->Padding[2]) ||
        (Parameters->Padding[0] != Parameters->Padding[3]) ||
        (ComputeConvOutSize(Parameters->InputShape[0],
                            ComputeKernelSize(Parameters->DilationShape[0],Parameters->KernelShape[0]),
                            Parameters->Padding[0], Parameters->StrideShape[0]) *
         ComputeConvOutSize(Parameters->InputShape[1],
                            ComputeKernelSize(Parameters->DilationShape[1],Parameters->KernelShape[1]),
                            Parameters->Padding[1], Parameters->StrideShape[1]) == 0)) {

        throw ARMKleidiAI::NotSupported();
    }

    //optimization checks - is the implementation optimal for the conv request

    const auto n_step = kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa();
    const auto m_step = kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa();

    auto M = ComputeConvOutSize(Parameters->InputShape[0], ComputeKernelSize(Parameters->DilationShape[0],
                                Parameters->KernelShape[0]), Parameters->Padding[0], Parameters->StrideShape[0]) *
             ComputeConvOutSize(Parameters->InputShape[1], ComputeKernelSize(Parameters->DilationShape[1],
                                Parameters->KernelShape[1]), Parameters->Padding[1], Parameters->StrideShape[1]);
    auto N = Parameters->FilterCount;
    auto K = Parameters->InputChannels * Parameters->KernelShape[0] * Parameters->KernelShape[1];

    //Can use these variables to add other conditions as required
    MLAS_UNREFERENCED_PARAMETER(M);
    MLAS_UNREFERENCED_PARAMETER(K);
    MLAS_UNREFERENCED_PARAMETER(m_step);
    MLAS_UNREFERENCED_PARAMETER(n_step);

    if (N == 1 || Parameters->KernelShape[0] < 3 || Parameters->KernelShape[1] < 3) {
        throw ARMKleidiAI::NotSupported();
    }
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

    auto out_ = std::make_unique_for_overwrite<float[]>(shape_in[0] * shape_in[1] * shape_in[2] * shape_in[3]);
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
                                           const float* __restrict__ in,
                                           const size_t dilationh=1,
                                           const size_t dilationw=1,
                                           const bool zero_fill=false,
                                           MLAS_THREADPOOL* ThreadPool=nullptr) {

    const auto d_h = ComputeKernelSize(dilationh, h);
    const auto d_w = ComputeKernelSize(dilationw, w);

    auto t = std::make_unique_for_overwrite<float[]>(n*d_h*d_w*c);
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
                    float* __restrict__ out3 = out2;
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

static void MultiThreadedLHSPackSme(MLAS_THREADPOOL* ThreadPool, const size_t ci, const size_t m, const size_t kh,
                                    const size_t kw, const void * const* lhs_ptrs, std::byte* lhs_data,
                                    const float* in_data,
                                    const float* pad_ptr) {

    auto m_step = kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa();

    // Minimize the kernel call count for the number of available threads
    auto RequiredTiles = MlasDivRoundup(m, m_step);
    auto MaxTiles = std::min(static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)), RequiredTiles);
    m_step *= MlasDivRoundup(RequiredTiles, MaxTiles);
    RequiredTiles = MlasDivRoundup(m, m_step);

    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(RequiredTiles), [&](ptrdiff_t tid) {

        auto m_idx = static_cast<size_t>(tid) * m_step;
        auto offset = kai_get_lhs_packed_offset_lhs_imatmul_pack_x32p2vlx1_x32p_sme(m_idx,kh*kw,ci);

        kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme(
            m < (m_idx + m_step) ? m - m_idx : m_step, kh * kw, ci,
            lhs_ptrs + m_idx * kh * kw,
            reinterpret_cast<size_t>(in_data),
            reinterpret_cast<const void*>(pad_ptr),
            lhs_data + offset
        );
    });
}

using CacheKey = float const * const;
//Generate a unique key for rhs weight packing caching
static CacheKey GenerateRhsKey(const size_t co, const size_t ci,
                               const size_t kh, const size_t kw,
                               const size_t dilationh, const size_t dilationw,
                               const float* weights) {

    //Weights, co, ci, kh, kw are constants. Dilation factors could vary
    //Encode the dilation factors into key
    auto key = &weights[(dilationh*2) + dilationw];
    if (key != std::clamp(key, &weights[0], &weights[(co * ci * kh * kw) - 1])) {
        throw std::invalid_argument("Invalid Rhs key");
    }
    return key;
}

//Generate a unique key for lhs indirection pointer array caching
static CacheKey GenerateLhsKey(const size_t ci,
                               const size_t ih, const size_t iw,
                               const size_t padding, const size_t sh, const size_t sw,
                               const size_t kh, const size_t kw,
                               const size_t dilationh, const size_t dilationw,
                               const float* data) {
    //Data, ci, ih, iw are constants.
    //Encode the others into key
    const auto d_kh = ComputeKernelSize(dilationh, kh);
    const auto d_kw = ComputeKernelSize(dilationw, kw);

    auto ih_out_size = ComputeConvOutSize(ih, d_kh, padding, sh);
    auto iw_out_size = ComputeConvOutSize(iw, d_kw, padding, sw);

    auto key = &data[ih_out_size * 2 + iw_out_size];
    if (key != std::clamp(key, &data[0], &data[(ci * ih * iw) - 1])) {
        throw std::invalid_argument("Invalid Lhs key");
    }
    return key;
}

static std::shared_ptr<std::byte[]> RhsPackWeightsBiasSme(CacheKey key,
                                                          const size_t co, const size_t ci,
                                                          const size_t kh, const size_t kw,
                                                          const size_t dilationh, const size_t dilationw,
                                                          const float* weights, const float* bias,
                                                          MLAS_THREADPOOL* ThreadPool)
{
    //cache of prepacked kai rhs weights and biases
    static std::map<CacheKey, std::shared_ptr<std::byte[]>> rhs_cache;

    //sanity check key
    if (rhs_cache.count(key) > 1) {
        throw std::invalid_argument("Invalid conv rhs key");
    }

    if (auto found = rhs_cache.find(key); found != rhs_cache.end()) {
        return found->second;
    } else {
        // prepare mlas filter weights for kai rhs packing
        // dilated nhwc format
        auto nhwc = NChwToNhwc(co, ci, kh, kw, weights, dilationh, dilationw, true, ThreadPool);

        //dilation, axis swap (n x k -> k x n) where n == co, k == d_kh x d_kw x ci
        const auto d_kh = ComputeKernelSize(dilationh,kh);
        const auto d_kw = ComputeKernelSize(dilationw,kw);

        //t_weights[d_kh][d_kw][ci][co] = nhwc[co][d_kh][d_kw][ci]
        auto t_weights = Transpose4D({co,d_kh,d_kw,ci},&nhwc[0],{1,2,3,0});

        const auto packed_size = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(co,d_kh*d_kw,ci);
        auto packed = std::make_shared_for_overwrite<std::byte[]>(packed_size);

        kai_run_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
            co, d_kh*d_kw, ci, co * sizeof(float), &t_weights[0], bias, packed.get()
        );

        rhs_cache[key] = packed;

        return packed;
    }
}

static std::shared_ptr<const void*[]> LhsPtrFill(const size_t ci, const size_t ih, const size_t iw,
                                                 const size_t kh, const size_t kw, size_t sh, size_t sw,
                                                 const size_t padding,
                                                 const float* pad_ptr) {
    size_t check_filled{0};

    const auto m = ComputeConvOutSize(ih, kh, padding, sh) * ComputeConvOutSize(iw, kw, padding, sw);

    const auto m_step = kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa();
    const auto lhs_ptrs_k = kh * kw;
    const auto lhs_ptrs_m = m_step * MlasDivRoundup(m, m_step);
    auto lhs_ptrs = std::make_shared_for_overwrite<const void*[]>(lhs_ptrs_k * lhs_ptrs_m);

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
    auto lhs_ptrs_ = lhs_ptrs.get();
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

static std::unique_ptr<std::byte[]> LhsPackImageDataSme(CacheKey key,
                                                        const size_t ci, const size_t ih, const size_t iw,
                                                        const size_t kh, const size_t kw, const size_t sh,
                                                        const size_t sw, const size_t padding, const float* in,
                                                        MLAS_THREADPOOL* ThreadPool)
{
    static const std::array<float, 256> pad_ptr{0.f};
    assert(pad_ptr.size() > ci);

    //create lhs in format required for imatmul
    const auto m = ComputeConvOutSize(ih, kh, padding, sh) * ComputeConvOutSize(iw, kw, padding, sw);

    const auto lhs_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme(m,kh*kw,ci);
    auto lhs = std::make_unique_for_overwrite<std::byte[]>(lhs_size);

    auto nhwc = NChwToNhwc(1, ci, ih, iw, in, 1, 1, false, ThreadPool);

    //cache of computed lhs ptr offsets
    static std::map<CacheKey, std::shared_ptr<const void*[]>> lhs_ptrs_cache;

    // sanity check key
    if (lhs_ptrs_cache.count(key) > 1) {
        throw std::invalid_argument("Invalid conv lhs_ptrs key");
    }

    std::shared_ptr<const void*[]> lhs_ptrs;
    if (auto found = lhs_ptrs_cache.find(key); found != lhs_ptrs_cache.end()) {
        lhs_ptrs = found->second;
    } else {
        lhs_ptrs = LhsPtrFill(ci, ih, iw, kh, kw, sh, sw, padding, &pad_ptr[0]);
        lhs_ptrs_cache[key] = lhs_ptrs;
    }

    MultiThreadedLHSPackSme(ThreadPool, ci, m, kh, kw, &lhs_ptrs[0], &lhs[0], &nhwc[0], &pad_ptr[0]);

    return lhs;
}

static void ConvolveSme(const size_t co, //channels out
                        const size_t ci, //channels in
                        const size_t ih, //image height
                        const size_t iw, //image width
                        const size_t kh, //kernel height
                        const size_t kw, //kernel width
                        const size_t sh, //kernel stride height
                        const size_t sw, //kernel stride width
                        const size_t dilationh, //kernel dilation stride
                        const size_t dilationw, //kernel dilation stride
                        const size_t padding,   //padding size
                        const size_t groups,       //number of filter groups
                        const float* weights,      //kernel weights [co,ci,ih,iw]
                        const float* bias,         //kernel biases
                        const float* in,           //in image data
                        float* out,                //out image data
                        float* tmp_mlas_aligned,   //intermediate buffer if we need to perform a transpose
                        MLAS_THREADPOOL* ThreadPool) {

    //RhsPackWeightsBiasSme() - to perform dilation increases kernel size and masks unused weights
    //compute corrected dimensions of dilated kernel
    const auto d_kh = ComputeKernelSize(dilationh, kh);
    const auto d_kw = ComputeKernelSize(dilationw, kw);

    //run igemm based convolution
    const auto m = ComputeConvOutSize(ih, d_kh, padding, sh) *
                   ComputeConvOutSize(iw, d_kw, padding, sw);

    auto n_step = kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa();
    auto m_step = kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa();

    //tile iteration dimensions
    std::array<size_t,3> dim;
    dim[0] = 1;                          // B
    dim[1] = MlasDivRoundup(m, m_step);  // M
    dim[2] = MlasDivRoundup(co, n_step); // N

    //Minimize the kernel call count for the number of available threads
    auto RequiredTiles = std::min(static_cast<size_t>(MlasGetMaximumThreadCount(ThreadPool)), dim[0]*dim[1]*dim[2]);

    //scale required tiles over available tile processors
    dim[1] = MlasDivRoundup(RequiredTiles * dim[1], dim[1] * dim[2]);
    dim[2] = MlasDivRoundup(RequiredTiles * dim[2], dim[1] * dim[2]);

    //compute new step sizes
    m_step *= MlasDivRoundup(MlasDivRoundup(m, dim[1]), m_step);
    n_step *= MlasDivRoundup(MlasDivRoundup(co, dim[2]), n_step);

    //update tile iterations
    dim[1] = MlasDivRoundup(m, m_step);
    dim[2] = MlasDivRoundup(co, n_step);

    for (size_t g = 0; g < groups; ++g) {

        auto result{out};
        //do we require a post matmul transpose ?
        //output is m x n or image_data x co or hw x co
        //MLAS require it as n x m (or co x hw), transpose required
        if (co > 1) {
            //intermediate buffer required, pre-transpose
            //Note: because we are calling MlasTranspose() need to ensure we use a MLAS aligned buffer
            result = tmp_mlas_aligned;
        }

        auto rhs_key = GenerateRhsKey(co, ci, kh, kw, dilationh, dilationw, weights);
        auto lhs_key = GenerateLhsKey(ci, ih, iw, padding, sh, sw, kh, kw, dilationh, dilationw, in);

        auto lhs = LhsPackImageDataSme(lhs_key, ci, ih, iw, d_kh, d_kw, sh, sw, padding, in, ThreadPool);
        auto rhs = RhsPackWeightsBiasSme(rhs_key, co, ci, kh, kw, dilationh, dilationw, weights, bias, ThreadPool);

        MlasTrySimpleParallel(ThreadPool,
            static_cast<ptrdiff_t>(dim[0]*dim[1]*dim[2]),
            [&](ptrdiff_t tid)
        {
            //compute B,M,N index from iteration index
            //ptrdiff_t BIdx = tid / (dim[1] * dim[2]);
            ptrdiff_t MIdx = (tid % (dim[1] * dim[2])) / dim[2];
            ptrdiff_t NIdx = (tid % (dim[1] * dim[2])) % dim[2];

            // Get rhs tile, B
            const size_t rhs_packed_offset =
                kai_get_rhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa(NIdx*n_step,
                                                                                                   d_kh*d_kw,ci);

            auto BTile = reinterpret_cast<const void*>(
                reinterpret_cast<const std::byte*>(rhs.get()) + rhs_packed_offset
            );

            // Get lhs tile, A
            const size_t lhs_packed_offset =
                kai_get_lhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa(MIdx*m_step,
                                                                                                   d_kh*d_kw,ci);

            auto ATile = reinterpret_cast<const float*>(
                reinterpret_cast<const std::byte*>(lhs.get()) + lhs_packed_offset
            );

            auto TileSizeM = (MIdx + 1) * m_step > m ? (m - MIdx * m_step) : m_step;
            auto TileSizeN = (NIdx + 1) * n_step > co ? (co - NIdx * n_step) : n_step;

            // Get result tile, C
            auto CTile = &reinterpret_cast<std::byte*>(result)[
                MIdx * m_step * co * sizeof(float) +
                NIdx * n_step * sizeof(float)];

            kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa(
                TileSizeM, TileSizeN, d_kh*d_kw, ci, ATile, BTile, CTile, co * sizeof(float),
                -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
            );
        });

        if (result == tmp_mlas_aligned) {
            //Note: this could be absorbed into post conv activation
            MlasTranspose(tmp_mlas_aligned, out, m, co, ThreadPool);
        }

        in += ci * ih * iw;
        out += m * co;
        weights += co * ci * kh * kw;
        bias += co;
    }
}

void MLASCALL
ARMKleidiAI::MlasConvPrepare(MLAS_CONV_PARAMETERS* Parameters,
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
                float Beta,
                MLAS_THREADPOOL* ThreadPool)
{
    //Check dimensions before accessing
    if (Dimensions < 2) {
        throw ARMKleidiAI::NotSupported();
    }

    Parameters->Activation = Activation;
    Parameters->Dimensions = Dimensions;
    Parameters->BatchCount = BatchCount;
    Parameters->GroupCount = GroupCount;
    Parameters->InputChannels = InputChannels;
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

    CheckCapabilitiesSme(Parameters);

    //Allocate an aligned buffer for MlasTranspose()
    *WorkingBufferSize = ComputeMlasWorkingBufferSize(Parameters->FilterCount,
                                                      Parameters->InputShape[0], Parameters->InputShape[1],
                                                      Parameters->KernelShape[0], Parameters->KernelShape[1],
                                                      Parameters->DilationShape[0], Parameters->DilationShape[1],
                                                      Parameters->StrideShape[0], Parameters->StrideShape[1],
                                                      Parameters->Padding[0]);
}

void
MLASCALL
ARMKleidiAI::MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
{
    CheckCapabilitiesSme(Parameters);

    ConvolveSme(Parameters->FilterCount, Parameters->InputChannels,          // channel out, in
                Parameters->InputShape[0], Parameters->InputShape[1],        // image dimensions
                Parameters->KernelShape[0], Parameters->KernelShape[1],      // kernel dimensions
                Parameters->StrideShape[0], Parameters->StrideShape[1],      // kernel stride dimensions
                Parameters->DilationShape[0], Parameters->DilationShape[1],  // kernel dilation
                Parameters->Padding[0],                                      // image padding
                Parameters->GroupCount,                                      // filter groups
                Filter, Bias, Input, Output, WorkingBuffer, ThreadPool);

    MlasActivation(Parameters->Activation, Output, nullptr, Parameters->FilterCount, Parameters->OutputSize,
                   Parameters->OutputSize);
}
