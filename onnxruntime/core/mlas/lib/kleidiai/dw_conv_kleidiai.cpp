//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "mlasi_kleidiai.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kai_ukernel_interface.h"

#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"

const KaiF32DepthwiseConvKernel& dwconv = GetKleidiAIDepthwiseConvUKernel();

namespace ArmKleidiAI {
namespace {

struct DwconvTlsBuffers {
    std::vector<float> feature_map_nhwc;
    std::vector<float> weights_hwcn;
    std::vector<std::byte> weights_packed;
    std::vector<float> nhwc_out;
    std::vector<float> bias_fallback;
};

thread_local DwconvTlsBuffers g_dwconv_tls;

constexpr size_t kDwconvColsPerTile = 4;

static void ConvertNchwToNhwc(const float* src,
                              float* dst,
                              size_t batches,
                              size_t channels,
                              size_t height,
                              size_t width) {
    const size_t src_stride_n = channels * height * width;
    const size_t src_stride_c = height * width;
    const size_t src_stride_h = width;
    const size_t dst_stride_n = height * width * channels;
    const size_t dst_stride_h = width * channels;
    const size_t dst_stride_w = channels;

    for (size_t n = 0; n < batches; ++n) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                for (size_t c = 0; c < channels; ++c) {
                    const size_t src_index = n * src_stride_n + c * src_stride_c + h * src_stride_h + w;
                    const size_t dst_index = n * dst_stride_n + h * dst_stride_h + w * dst_stride_w + c;
                    dst[dst_index] = src[src_index];
                }
            }
        }
    }
}

static void ConvertNhwcToNchw(const float* src,
                              float* dst,
                              size_t batches,
                              size_t channels,
                              size_t height,
                              size_t width) {
    const size_t dst_stride_n = channels * height * width;
    const size_t dst_stride_c = height * width;
    const size_t dst_stride_h = width;
    const size_t src_stride_n = height * width * channels;
    const size_t src_stride_h = width * channels;
    const size_t src_stride_w = channels;

    for (size_t n = 0; n < batches; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    const size_t src_index = n * src_stride_n + h * src_stride_h + w * src_stride_w + c;
                    const size_t dst_index = n * dst_stride_n + c * dst_stride_c + h * dst_stride_h + w;
                    dst[dst_index] = src[src_index];
                }
            }
        }
    }
}

static void ConvertDepthwiseWeightsToHwcn(const float* src,
                                          float* dst,
                                          size_t channels,
                                          size_t filter_height,
                                          size_t filter_width) {
    const size_t kernel_size = filter_height * filter_width;
    for (size_t c = 0; c < channels; ++c) {
        const float* channel_weights = src + c * kernel_size;
        for (size_t kh = 0; kh < filter_height; ++kh) {
            for (size_t kw = 0; kw < filter_width; ++kw) {
                const size_t dst_index = (kh * filter_width + kw) * channels + c;
                dst[dst_index] = channel_weights[kh * filter_width + kw];
            }
        }
    }
}

static bool TryComputeDepthwiseOutputShape(size_t in_height,
                                           size_t in_width,
                                           size_t filter_height,
                                           size_t filter_width,
                                           size_t pad_top,
                                           size_t pad_left,
                                           size_t pad_bottom,
                                           size_t pad_right,
                                           size_t& out_height,
                                           size_t& out_width) {
    const size_t padded_height = in_height + pad_top + pad_bottom;
    const size_t padded_width = in_width + pad_left + pad_right;

    if (padded_height < filter_height || padded_width < filter_width) {
        return false;
    }

    out_height = padded_height + 1 - filter_height;
    out_width = padded_width + 1 - filter_width;
    return true;
}

}  // namespace

bool
MLASCALL
DepthwiseConvKleidiAISupported(const MLAS_CONV_PARAMETERS* Parameters) {
    if (Parameters == nullptr) {
        return false;
    }

    if (!UseSME2) {
        return false;
    }

    if (Parameters->BackendKernelSelectorConfig && !Parameters->BackendKernelSelectorConfig->use_kleidiai) {
        return false;
    }

    if (Parameters->Dimensions != 2) {
        return false;
    }

    // The current direct kernel path processes a single batch at a time.
    if (Parameters->BatchCount != 1) {
        return false;
    }

    if (Parameters->Beta != 0.0f) {
        return false;
    }

    // Depthwise convolution with multiplier 1: one input channel and one filter per group.
    if (Parameters->InputChannels != 1 || Parameters->FilterCount != 1) {
        return false;
    }

    // Kernel specialization is 3x3 with unit stride and dilation.
    if (Parameters->KernelShape[0] != 3 || Parameters->KernelShape[1] != 3) {
        return false;
    }

    if (Parameters->StrideShape[0] != 1 || Parameters->StrideShape[1] != 1) {
        return false;
    }

    if (Parameters->DilationShape[0] != 1 || Parameters->DilationShape[1] != 1) {
        return false;
    }

    const bool zero_padding = Parameters->Padding[0] == 0 && Parameters->Padding[1] == 0 &&
                              Parameters->Padding[2] == 0 && Parameters->Padding[3] == 0;
    const bool unit_padding = Parameters->Padding[0] == 1 && Parameters->Padding[1] == 1 &&
                              Parameters->Padding[2] == 1 && Parameters->Padding[3] == 1;
    if (!zero_padding && !unit_padding) {
        return false;
    }

    size_t out_height = 0;
    size_t out_width = 0;
    if (!TryComputeDepthwiseOutputShape(Parameters->InputShape[0],
                                        Parameters->InputShape[1],
                                        Parameters->KernelShape[0],
                                        Parameters->KernelShape[1],
                                        Parameters->Padding[0],
                                        Parameters->Padding[1],
                                        Parameters->Padding[2],
                                        Parameters->Padding[3],
                                        out_height,
                                        out_width)) {
        return false;
    }

    return out_height >= dwconv.ukernel.get_m_step() && out_width >= kDwconvColsPerTile;
}

bool
MLASCALL
DepthwiseConvKleidiAI(size_t batches,
                      size_t in_height,
                      size_t in_width,
                      size_t channels,
                      size_t filter_height,
                      size_t filter_width,
                      size_t pad_top,
                      size_t pad_left,
                      size_t pad_bottom,
                      size_t pad_right,
                      bool channels_last,
                      const float* feature_map,
                      const float* weights,
                      const float* bias,
                      float* out,
                      float clamp_min,
                      float clamp_max) {
    if (!UseSME2 || feature_map == nullptr || weights == nullptr || out == nullptr) {
        return false;
    }

    if (batches != 1 || channels == 0) {
        return false;
    }

    size_t out_height = 0;
    size_t out_width = 0;
    if (!TryComputeDepthwiseOutputShape(in_height,
                                        in_width,
                                        filter_height,
                                        filter_width,
                                        pad_top,
                                        pad_left,
                                        pad_bottom,
                                        pad_right,
                                        out_height,
                                        out_width)) {
        return false;
    }

    const size_t rows_handled = dwconv.ukernel.get_m_step();
    if (out_height < rows_handled || out_width < kDwconvColsPerTile) {
        return false;
    }

    auto& tls = g_dwconv_tls;

    const float* feature_map_nhwc = feature_map;
    if (!channels_last) {
        const size_t input_size = batches * in_height * in_width * channels;
        tls.feature_map_nhwc.resize(input_size);
        ConvertNchwToNhwc(feature_map, tls.feature_map_nhwc.data(), batches, channels, in_height, in_width);
        feature_map_nhwc = tls.feature_map_nhwc.data();
    }

    const size_t weights_size = filter_height * filter_width * channels;
    tls.weights_hwcn.resize(weights_size);
    ConvertDepthwiseWeightsToHwcn(weights, tls.weights_hwcn.data(), channels, filter_height, filter_width);

    const float* bias_data = bias;
    if (bias_data == nullptr) {
        tls.bias_fallback.assign(channels, 0.0f);
        bias_data = tls.bias_fallback.data();
    }

    const size_t packed_size_bytes =
        kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme(filter_height, filter_width, channels);
    tls.weights_packed.resize(packed_size_bytes);
    KLEIDIAI_KERNEL_LOG("kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme"
                        << " filter_height=" << filter_height << " filter_width=" << filter_width
                        << " channels=" << channels);
    kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(filter_height,
                                                   filter_width,
                                                   filter_height,
                                                   filter_width,
                                                   channels,
                                                   tls.weights_hwcn.data(),
                                                   bias_data,
                                                   tls.weights_packed.data());

    float* nhwc_out = out;
    if (!channels_last) {
        const size_t output_size = batches * out_height * out_width * channels;
        tls.nhwc_out.assign(output_size, 0.0f);
        nhwc_out = tls.nhwc_out.data();
    }

    const size_t in_row_stride_elements = in_width * channels;
    const size_t out_row_stride_elements = out_width * channels;
    for (size_t out_row = 0; out_row < out_height; out_row += rows_handled) {
        const ptrdiff_t start_in_row = static_cast<ptrdiff_t>(out_row) - static_cast<ptrdiff_t>(pad_top);
        const size_t kernel_pad_top = start_in_row < 0 ? static_cast<size_t>(-start_in_row) : 0;
        const size_t in_row = start_in_row < 0 ? 0 : static_cast<size_t>(start_in_row);

        const size_t rows_to_process = std::min(rows_handled, out_height - out_row);
        size_t valid_input_rows = 0;
        if (in_row < in_height) {
            const size_t max_rows_available = in_height - in_row;
            const size_t needed_rows = filter_height + rows_to_process - 1;
            valid_input_rows = std::min(max_rows_available, needed_rows);
        }

        const float* inptr = feature_map_nhwc + in_row * in_row_stride_elements;
        float* outptr = nhwc_out + out_row * out_row_stride_elements;

        KLEIDIAI_KERNEL_LOG(dwconv.name
                            << " valid_input_rows=" << valid_input_rows
                            << " valid_dst_rows=" << rows_to_process
                            << " pad_left=" << pad_left << " pad_top=" << kernel_pad_top);
        dwconv.ukernel.run_dwconv(inptr,
                                  tls.weights_packed.data(),
                                  outptr,
                                  in_row_stride_elements * sizeof(float),
                                  channels * sizeof(float),
                                  out_row_stride_elements * sizeof(float),
                                  channels * sizeof(float),
                                  valid_input_rows,
                                  rows_to_process,
                                  pad_left,
                                  kernel_pad_top,
                                  0.0f,
                                  clamp_min,
                                  clamp_max);
    }

    if (!channels_last) {
        ConvertNhwcToNchw(nhwc_out, out, batches, channels, out_height, out_width);
    }

    return true;
}

}  // namespace ArmKleidiAI
