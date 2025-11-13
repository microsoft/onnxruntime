//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include "mlasi_kleidiai.h"
#include <float.h>

#include <cstddef>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"

using VEC_TYPE = std::vector<float>;

struct Padding2D {
    size_t left = 0;
    size_t right = 0;
    size_t bottom = 0;
    size_t top = 0;
};

struct Shape {
    size_t n = 1;
    size_t h = 1;
    size_t w = 1;
    size_t c = 1;

    [[nodiscard]] auto size() const -> size_t {
        return n * h * w * c;
    }

    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << " [ " << shape.n << " , " << shape.h << " ," << shape.w << " , " << shape.c << " ] ";
        return os;
    }

    constexpr const std::size_t& operator[](std::size_t idx) const {
        switch (idx) {
            case 0:
                return n;
            case 1:
                return h;
            case 2:
                return w;
            case 3:
                return c;
            default:
                throw std::out_of_range("Shape-index out of range (0-3)");
        }
    }
};

void print_raw(const Shape& shape, const char* name, const VEC_TYPE& src) {
    std::cout << "\n\n" << name << " = [";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) std::cout << " , ";
        std::cout << std::setprecision(1) << std::fixed << (float)src[i];
    }
    std::cout << "]\n";
}

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
                    const size_t src_index =
                        n * src_stride_n + c * src_stride_c + h * src_stride_h + w;
                    const size_t dst_index =
                        n * dst_stride_n + h * dst_stride_h + w * dst_stride_w + c;
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
                    const size_t src_index =
                        n * src_stride_n + h * src_stride_h + w * src_stride_w + c;
                    const size_t dst_index =
                        n * dst_stride_n + c * dst_stride_c + h * dst_stride_h + w;
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
                const size_t dst_index = ((kh * filter_width) + kw) * channels + c;
                dst[dst_index] = channel_weights[kh * filter_width + kw];
            }
        }
    }
}

/// Depthwise Convolution - Expects NHWC dataformat. Padding value is 0.
///
/// @tparam T Data type.
///
/// @param[in] batches   Batch dimension of feature map.
/// @param[in] in_height height of feature map.
/// @param[in] in_width  width of feature map.
/// @param[in] channels  Number of channels in feature map.
/// @param[in] filter_height Height dimension in filter.
/// @param[in] filter_width  Width of convolution filter.
/// @param[in] feature_map Ptr to start of feature map.
/// @param[in] weights Ptr to start of weights buffer/tensor.
/// @param[in] bias Ptr to start of bias buffer.
/// @param[in] clamp_min float value to clamp output to (lower bound).
/// @param[in] clamp_max float value to clamp output to (upper bound).
///
/// @return The result data buffer.
void DepthwiseReference(const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, void* out, float clamp_min, float clamp_max){
            // Calculate output dims (Padding = Valid).
    const Padding2D padding = Padding2D{}; //Using default padding value
    const size_t out_height = (in_height + padding.top + padding.bottom + 1 - filter_height);
    const size_t out_width = in_width + padding.left + padding.right + 1 - filter_width;
    const size_t out_size = out_height * out_width * batches * channels;

    // We accumulate in FP32 and clamp and cast to return type later.
    std::vector<float> acc(out_size, 0.0f);

    for (size_t b = 0; b < batches; ++b) {
        for (size_t out_h = 0; out_h < out_height; ++out_h) {
            for (size_t out_w = 0; out_w < out_width; ++out_w) {
                const size_t out_base = ((b * out_height + out_h) * out_width + out_w) * channels;

                // Apply filter to feature map.
                for (size_t ic = 0; ic < channels; ++ic) {
                    float sum = 0.0f;

                    for (size_t kernel_h = 0; kernel_h < filter_height; ++kernel_h) {
                        // Determine if input height bounds. If not, then this is padding.
                        const int in_y = static_cast<int>(out_h + kernel_h) - static_cast<int>(padding.top);
                        if (in_y < 0 || in_height <= static_cast<size_t>(in_y)) continue;

                        for (size_t kernel_w = 0; kernel_w < filter_width; ++kernel_w) {
                            // Determine if in input width bounds, if not this is padding.
                            const int in_x = static_cast<int>(out_w + kernel_w) - static_cast<int>(padding.left);
                            if (in_x < 0 || in_width <= static_cast<size_t>(in_x)) continue;

                            auto in_idx = ((b * in_height + in_y) * in_width + in_x) * channels + ic;
                            auto weights_idx = ((kernel_h * filter_width) + kernel_w) * channels + ic;

                            auto wei_value = reinterpret_cast<const float*>(weights)[weights_idx];
                            auto in_value = reinterpret_cast<const float*>(feature_map)[in_idx];

                            // Perform actual accumulation and store in output vector
                            sum += in_value * wei_value;
                        }
                    }

                    auto out_idx = out_base + ic;
                    float bias_value = reinterpret_cast<const float*>(bias)[ic];
                    sum = sum + bias_value;
                    sum = std::clamp(sum, clamp_min, clamp_max);
                    reinterpret_cast<float*>(out)[out_idx] = sum;
                }
            }
        }
    }
}

bool DepthwiseConvKleidiAISupported(const MLAS_CONV_PARAMETERS* Parameters) {
    if (Parameters == nullptr) {
        return false;
    }

    if (Parameters->Dimensions != 2) {
        return false;
    }

    // Currently only support single batch depthwise to match kernel layout expectations.
    if (Parameters->BatchCount != 1) {
        return false;
    }

    if (Parameters->Beta != 0.0f) {
        return false;
    }

    // Depthwise conv with multiplier 1 => one input channel per group and one filter per group.
    if (Parameters->InputChannels != 1 || Parameters->FilterCount != 1) {
        return false;
    }

    // Kernel specialization is for 3x3 with unit stride and dilation, and no padding.
    if (Parameters->KernelShape[0] != 3 || Parameters->KernelShape[1] != 3) {
        return false;
    }

    if (Parameters->StrideShape[0] != 1 || Parameters->StrideShape[1] != 1) {
        return false;
    }

    if (Parameters->DilationShape[0] != 1 || Parameters->DilationShape[1] != 1) {
        return false;
    }

    if (Parameters->Padding[0] != 0 || Parameters->Padding[1] != 0 ||
        Parameters->Padding[2] != 0 || Parameters->Padding[3] != 0) {
        return false;
    }

    return true;
}

bool DepthwiseConvKleidiAI(const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const float* feature_map, const float* weights,
    const float* bias, float* out, float clamp_min, float clamp_max){
    // -------------------------------------------------
    // 1. Constants
    // -------------------------------------------------
    const Padding2D padding = Padding2D{}; //Using default padding values
    const int padded_in_height = static_cast<int>(in_height + padding.top + padding.bottom);
    const int padded_in_width  = static_cast<int>(in_width + padding.left + padding.right);

    const int out_h = padded_in_height + 1 - static_cast<int>(filter_height);
    const int out_w = padded_in_width  + 1 - static_cast<int>(filter_width);

    if (feature_map == nullptr || weights == nullptr || out == nullptr) {
        return false;
    }

    if (channels == 0 || batches == 0) {
        return false;
    }

    const size_t input_size = batches * in_height * in_width * channels;
    std::vector<float> feature_map_nhwc(input_size);
    ConvertNchwToNhwc(feature_map, feature_map_nhwc.data(), batches, channels, in_height, in_width);

    const size_t weights_size = filter_height * filter_width * channels;
    std::vector<float> weights_hwcn(weights_size);
    ConvertDepthwiseWeightsToHwcn(weights, weights_hwcn.data(), channels, filter_height, filter_width);

    if (out_h <= 4 || out_w <= 4) {
        return false;
    }

    if (batches != 1) {
        return false;
    }

    Shape out_shape{batches, static_cast<size_t>(out_h), static_cast<size_t>(out_w), channels};

    // -------------------------------------------------
    // 2. Calculate Reference Depthwise Values.
    // -------------------------------------------------
    //DepthwiseReference(
    //    batches, in_height, in_width, channels, filter_height, filter_width, feature_map,
    //    weights, bias, out, clamp_min, clamp_max);

    // -------------------------------------------------
    // 3. Pack weights for use in SME Kernel
    // -------------------------------------------------
    // const size_t vec_length = kai_get_sme_vector_length_u32();
    // Bias is optional in MLAS so fall back to zeros when not supplied.
    std::vector<float> bias_fallback;
    const float* bias_data = bias;
    if (bias_data == nullptr) {
        bias_fallback.assign(channels, 0.0f);
        bias_data = bias_fallback.data();
    }

    const size_t packed_size_bytes =
        kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme(filter_height, filter_width, channels);

    // Run packing kernel.
    std::vector<uint8_t> weights_packed(packed_size_bytes);
    kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(
        filter_height, filter_width, filter_height, filter_width, channels,
        static_cast<const void*>(weights_hwcn.data()), static_cast<const void*>(bias_data), weights_packed.data());

    // -------------------------------------------------
    // 4. Kernel takes in 6 rows of input and generates
    //    rows of output across all channels at a time.
    // -------------------------------------------------
    std::vector<float> nhwc_out_buffer(out_shape.size(), 0.0f);
    constexpr size_t rows_handled = 4;  // no of rows kernel handles each time.
    for (size_t out_row = 0; out_row < out_shape.h; out_row += rows_handled) {
        // Variables below used to calculate start of input pointer.
        const ptrdiff_t start_in_row = static_cast<ptrdiff_t>(out_row) - static_cast<ptrdiff_t>(padding.top);
        const unsigned int pad_top = start_in_row < 0 ? static_cast<unsigned int>(-start_in_row) : 0U;
        const size_t in_row = start_in_row < 0 ? 0 : static_cast<size_t>(start_in_row);

        // Calculate row strides for pointer.
        const size_t in_row_stride_elements = (in_width * channels);
        const size_t out_row_stride_elements = (out_shape.w * out_shape.c);

        // Number of input rows that can be read, number of output rows to calculate.
        const size_t rows_to_process = std::min(rows_handled, out_shape.h - out_row);
        size_t valid_input_rows = 0;
        if (in_row < in_height) {
            const size_t max_rows_available = in_height - in_row;
            const size_t needed_rows = filter_height + rows_to_process - 1;
            valid_input_rows = std::min(max_rows_available, needed_rows);
        }

        // Increment output/input pointers according to tile being calculated.
        const auto inptr = feature_map_nhwc.data() + (in_row * in_row_stride_elements);
        auto outptr = nhwc_out_buffer.data() + (out_row * out_row_stride_elements);

        // NOTE: Kernel expects strides to be passed as bytes.
        // f32_f32_f32p1vl -> f32 output, f32 LHS, packed F32 rhs as 1VL blocks.
        // 3x3_s : 3x3 filter with stride 1
        // 4xc : 4 output channels across the plane(c) is produced.

        kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
            static_cast<const void*>(inptr),
            static_cast<const void*>(weights_packed.data()),
            static_cast<void*>(outptr),
            in_row_stride_elements * sizeof(float),
            channels * sizeof(float),
            out_row_stride_elements * sizeof(float),
            out_shape.c * sizeof(float),
            valid_input_rows,
            rows_to_process,
            padding.left,
            pad_top,
            0.0f,
            clamp_min,
            clamp_max);
    }

    // Convert kernel NHWC output back to MLAS expected NCHW layout.
    ConvertNhwcToNchw(nhwc_out_buffer.data(), out, out_shape.n, out_shape.c, out_shape.h, out_shape.w);
    return true;
}
