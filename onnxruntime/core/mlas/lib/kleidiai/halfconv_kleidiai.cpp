//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <limits>
#include <vector>

#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.h"
#include "kai_ukernel_interface.h"
#include "mlasi_kleidiai.h"

namespace
{

// Cache-oriented heuristics for the chunked IMATMUL path. These bound the
// packed LHS working set so we only split when the single-pass pack/compute
// footprint is large enough that chunking can improve locality.
constexpr size_t AutomaticMaximumLhsChunkBytes = 2 * 1024 * 1024;
constexpr size_t MinimumLhsPackedBytesForAutomaticChunking = 8 * 1024 * 1024;
constexpr size_t MinimumEffectiveKForAutomaticChunking = 1024;
// Small filter counts typically do not amortize the extra chunking overhead.
constexpr size_t MaximumFilterCountForAutomaticChunking = 32;

struct KaiHalfConvTlsBuffers {
    std::vector<std::byte> packed_lhs;

    void ReleaseLargeBuffers() {
        ArmKleidiAI::MlasShrinkKleidiAIScratchIfTooLarge(packed_lhs);
    }
};

struct ScopedKaiHalfConvTlsCleanup {
    KaiHalfConvTlsBuffers& buffers;

    ~ScopedKaiHalfConvTlsCleanup() {
        buffers.ReleaseLargeBuffers();
    }
};

thread_local KaiHalfConvTlsBuffers g_kai_half_conv_tls;

template <typename T>
bool TryResizeVector(std::vector<T>& buffer, size_t size) {
    if (size > buffer.max_size()) {
        return false;
    }
    buffer.resize(size);
    return true;
}

bool
TryComputeKernelSize(
    size_t dilation,
    size_t kernel,
    size_t& dilated_kernel
)
{
    if (dilation == 0 || kernel == 0) {
        return false;
    }

    size_t scaled_kernel = 0;
    if (MlasMultiplyOverflowsSizeT(dilation, kernel, &scaled_kernel) || scaled_kernel < (dilation - 1)) {
        return false;
    }

    dilated_kernel = scaled_kernel - (dilation - 1);
    return true;
}

bool
TryComputeConvOutSize(
    size_t input,
    size_t kernel,
    size_t padding,
    size_t stride,
    size_t& output
)
{
    output = 0;
    if (stride == 0) {
        return false;
    }

    size_t total_padding = 0;
    size_t padded_input = 0;
    if (MlasMultiplyOverflowsSizeT(padding, 2, &total_padding) ||
        MlasAddOverflowsSizeT(input, total_padding, &padded_input)) {
        return false;
    }

    if (padded_input < kernel) {
        return true;
    }

    output = ((padded_input - kernel) / stride) + 1;
    return true;
}

bool
TryComputeOutputSize(
    size_t input_height,
    size_t input_width,
    size_t kernel_height,
    size_t kernel_width,
    size_t padding_height,
    size_t padding_width,
    size_t stride_height,
    size_t stride_width,
    size_t& output_height,
    size_t& output_width,
    size_t& output_size
)
{
    if (!TryComputeConvOutSize(input_height, kernel_height, padding_height, stride_height, output_height) ||
        !TryComputeConvOutSize(input_width, kernel_width, padding_width, stride_width, output_width) ||
        MlasMultiplyOverflowsSizeT(output_height, output_width, &output_size)) {
        return false;
    }

    return true;
}

bool
TryComputeOutputSize(
    size_t input_height,
    size_t input_width,
    size_t kernel_height,
    size_t kernel_width,
    size_t padding_height,
    size_t padding_width,
    size_t stride_height,
    size_t stride_width,
    size_t& output_size
)
{
    size_t output_height = 0;
    size_t output_width = 0;
    return TryComputeOutputSize(
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        padding_height,
        padding_width,
        stride_height,
        stride_width,
        output_height,
        output_width,
        output_size);
}

size_t
SelectMaximumLhsChunkBytes(
    size_t full_lhs_size,
    size_t filter_count,
    size_t effective_k
)
{
    // The bounded-LHS path is most useful when LHS packing is cache-hostile and
    // there are few output channels, so full-LHS reuse across N tiles is limited.
    if (full_lhs_size >= MinimumLhsPackedBytesForAutomaticChunking &&
        filter_count <= MaximumFilterCountForAutomaticChunking &&
        effective_k >= MinimumEffectiveKForAutomaticChunking) {
        return AutomaticMaximumLhsChunkBytes;
    }

    return 0;
}

bool
IsPaddingSymmetric2D(const MLAS_CONV_PARAMETERS* parameters)
{
    return parameters->Padding[0] == parameters->Padding[1] &&
           parameters->Padding[0] == parameters->Padding[2] &&
           parameters->Padding[0] == parameters->Padding[3];
}

bool
CheckCapabilitiesSme(const MLAS_CONV_PARAMETERS* parameters)
{
    if (parameters == nullptr) {
        return false;
    }

    if (parameters->BackendKernelSelectorConfig != nullptr &&
        !parameters->BackendKernelSelectorConfig->use_kleidiai) {
        KLEIDIAI_DEBUG_LOG("User explicitly disabled KleidiAI, returning false from MlasHalfConv.");
        return false;
    }

    if ((parameters->Dimensions != 2) ||
        (parameters->BatchCount != 1) ||
        (parameters->GroupCount != 1) ||
        (parameters->Beta != 0.0f) ||
        !(parameters->Activation == nullptr ||
          parameters->Activation->ActivationKind == MlasIdentityActivation) ||
        !IsPaddingSymmetric2D(parameters)) {
        KLEIDIAI_DEBUG_LOG("MlasHalfConv capability check failed: unsupported configuration.");
        return false;
    }

    size_t d_kh = 0;
    size_t d_kw = 0;
    size_t output_size = 0;
    if (!TryComputeKernelSize(parameters->DilationShape[0], parameters->KernelShape[0], d_kh) ||
        !TryComputeKernelSize(parameters->DilationShape[1], parameters->KernelShape[1], d_kw) ||
        !TryComputeOutputSize(
            parameters->InputShape[0],
            parameters->InputShape[1],
            d_kh,
            d_kw,
            parameters->Padding[0],
            parameters->Padding[1],
            parameters->StrideShape[0],
            parameters->StrideShape[1],
            output_size)) {
        return false;
    }

    if (output_size == 0 ||
        output_size != parameters->OutputSize ||
        parameters->InputChannels == 0 ||
        parameters->FilterCount == 0 ||
        parameters->FilterCount == 1 ||
        parameters->KernelShape[0] < 3 ||
        parameters->KernelShape[1] < 3) {
        KLEIDIAI_DEBUG_LOG("MlasHalfConv capability check failed: shape/heuristic gating.");
        return false;
    }

    return true;
}

bool
GetPackedFilterSize(
    size_t filter_count,
    size_t input_channels,
    const int64_t* kernel_shape,
    const int64_t* dilation_shape,
    size_t* packed_size
)
{
    if (packed_size == nullptr ||
        kernel_shape == nullptr ||
        dilation_shape == nullptr ||
        filter_count <= 1 ||
        input_channels == 0 ||
        kernel_shape[0] < 3 ||
        kernel_shape[1] < 3 ||
        dilation_shape[0] <= 0 ||
        dilation_shape[1] <= 0) {
        return false;
    }

    size_t d_kh = 0;
    size_t d_kw = 0;
    size_t k_chunk_count = 0;
    if (!TryComputeKernelSize(static_cast<size_t>(dilation_shape[0]), static_cast<size_t>(kernel_shape[0]), d_kh) ||
        !TryComputeKernelSize(static_cast<size_t>(dilation_shape[1]), static_cast<size_t>(kernel_shape[1]), d_kw) ||
        MlasMultiplyOverflowsSizeT(d_kh, d_kw, &k_chunk_count)) {
        return false;
    }

    *packed_size = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
        filter_count, k_chunk_count, input_channels
    );
    return *packed_size != 0;
}

bool
NchwToNhwc(
    const MLAS_FP16* input,
    size_t channels,
    size_t height,
    size_t width,
    std::vector<MLAS_FP16>& output
)
{
    size_t element_count = 0;
    if (MlasMultiplyOverflowsSizeT(channels, height, &element_count) ||
        MlasMultiplyOverflowsSizeT(element_count, width, &element_count)) {
        return false;
    }
    output.resize(element_count);

    if (input == nullptr) {
        return false;
    }

    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                output[(h * width + w) * channels + c] = input[(c * height + h) * width + w];
            }
        }
    }

    return true;
}

bool
FillIndirectionTable(
    const MLAS_FP16* input_nhwc,
    const MLAS_FP16* pad,
    size_t input_channels,
    size_t input_height,
    size_t input_width,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_height,
    size_t stride_width,
    size_t padding,
    size_t output_size,
    std::vector<const void*>& indirection
)
{
    const auto& imatmul = GetKleidiAIF16IMatmulUKernel();
    const size_t m_step = imatmul.ukernel.get_m_step();
    size_t lhs_ptrs_k = 0;
    if (MlasMultiplyOverflowsSizeT(kernel_height, kernel_width, &lhs_ptrs_k)) {
        return false;
    }

    size_t lhs_ptrs_m = 0;
    if (MlasMultiplyOverflowsSizeT(m_step, MlasDivRoundup(output_size, m_step), &lhs_ptrs_m)) {
        return false;
    }

    size_t table_size = 0;
    if (MlasMultiplyOverflowsSizeT(lhs_ptrs_k, lhs_ptrs_m, &table_size)) {
        return false;
    }
    indirection.resize(table_size);

    std::fill(indirection.begin(), indirection.end(), pad);

    auto ptr_offset = [lhs_ptrs_k, m_step](size_t k, size_t m) {
        return ((m / m_step) * lhs_ptrs_k * m_step) + (k * m_step) + (m % m_step);
    };

    auto pixel_ptr = [=](size_t h, size_t w) -> const void* {
        if (h < padding || w < padding) {
            return pad;
        }

        h -= padding;
        w -= padding;
        if (h >= input_height || w >= input_width) {
            return pad;
        }

        return input_nhwc + (h * input_width + w) * input_channels;
    };

    size_t output_height = 0;
    size_t output_width = 0;
    size_t computed_output_size = 0;
    if (!TryComputeOutputSize(
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            padding,
            padding,
            stride_height,
            stride_width,
            output_height,
            output_width,
            computed_output_size) ||
        computed_output_size != output_size) {
        return false;
    }

    size_t m = 0;
    for (size_t oh = 0; oh < output_height; ++oh) {
        for (size_t ow = 0; ow < output_width; ++ow, ++m) {
            size_t k = 0;
            const size_t input_base_h = oh * stride_height;
            const size_t input_base_w = ow * stride_width;
            for (size_t kh = 0; kh < kernel_height; ++kh) {
                for (size_t kw = 0; kw < kernel_width; ++kw, ++k) {
                    indirection[ptr_offset(k, m)] = pixel_ptr(input_base_h + kh, input_base_w + kw);
                }
            }
        }
    }

    return m == output_size;
}

bool
PrepareLhsInput(
    const MLAS_CONV_PARAMETERS* parameters,
    const MLAS_FP16* input,
    size_t output_size,
    std::vector<MLAS_FP16>& input_nhwc,
    std::vector<MLAS_FP16>& pad,
    std::vector<const void*>& indirection
)
{
    size_t d_kh = 0;
    size_t d_kw = 0;
    if (!TryComputeKernelSize(parameters->DilationShape[0], parameters->KernelShape[0], d_kh) ||
        !TryComputeKernelSize(parameters->DilationShape[1], parameters->KernelShape[1], d_kw)) {
        return false;
    }

    const size_t input_channels = parameters->InputChannels;

    const MLAS_FP16* input_nhwc_data = input;
    if (!parameters->InputOutputChannelsLast) {
        if (!NchwToNhwc(
                input,
                input_channels,
                parameters->InputShape[0],
                parameters->InputShape[1],
                input_nhwc
            )) {
            return false;
        }
        input_nhwc_data = input_nhwc.data();
    }

    pad.resize(input_channels);
    std::fill(pad.begin(), pad.end(), MLAS_FP16::FromBits(0));

    return FillIndirectionTable(
            input_nhwc_data,
            pad.data(),
            input_channels,
            parameters->InputShape[0],
            parameters->InputShape[1],
            d_kh,
            d_kw,
            parameters->StrideShape[0],
            parameters->StrideShape[1],
            parameters->Padding[0],
            output_size,
            indirection
        );
}

bool
PackFilter(
    size_t filter_count,
    size_t input_channels,
    const int64_t* kernel_shape,
    const int64_t* dilation_shape,
    const MLAS_FP16* filter,
    const MLAS_FP16* bias,
    void* packed_filter
)
{
    if (filter == nullptr || packed_filter == nullptr ||
        kernel_shape == nullptr || dilation_shape == nullptr ||
        filter_count <= 1 || input_channels == 0 ||
        kernel_shape[0] < 3 || kernel_shape[1] < 3 ||
        dilation_shape[0] <= 0 || dilation_shape[1] <= 0) {
        return false;
    }

    const size_t kernel_height = static_cast<size_t>(kernel_shape[0]);
    const size_t kernel_width = static_cast<size_t>(kernel_shape[1]);
    const size_t dilation_height = static_cast<size_t>(dilation_shape[0]);
    const size_t dilation_width = static_cast<size_t>(dilation_shape[1]);
    size_t d_kh = 0;
    size_t d_kw = 0;
    size_t k_chunk_count = 0;
    if (!TryComputeKernelSize(dilation_height, kernel_height, d_kh) ||
        !TryComputeKernelSize(dilation_width, kernel_width, d_kw) ||
        MlasMultiplyOverflowsSizeT(d_kh, d_kw, &k_chunk_count)) {
        return false;
    }

    size_t reordered_size = 0;
    if (MlasMultiplyOverflowsSizeT(k_chunk_count, input_channels, &reordered_size) ||
        MlasMultiplyOverflowsSizeT(reordered_size, filter_count, &reordered_size)) {
        return false;
    }

    std::vector<MLAS_FP16> reordered_filter;
    reordered_filter.resize(reordered_size);
    std::fill(reordered_filter.begin(), reordered_filter.end(), MLAS_FP16::FromBits(0));

    for (size_t oc = 0; oc < filter_count; ++oc) {
        for (size_t ic = 0; ic < input_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_height; ++kh) {
                for (size_t kw = 0; kw < kernel_width; ++kw) {
                    const size_t src = ((oc * input_channels + ic) * kernel_height + kh) * kernel_width + kw;
                    const size_t dk = ((kh * dilation_height) * d_kw + (kw * dilation_width)) * input_channels + ic;
                    reordered_filter[dk * filter_count + oc] = filter[src];
                }
            }
        }
    }

    std::vector<MLAS_FP16> zero_bias;
    const MLAS_FP16* bias_data = bias;
    if (bias_data == nullptr) {
        zero_bias.resize(filter_count);
        std::fill(zero_bias.begin(), zero_bias.end(), MLAS_FP16::FromBits(0));
        bias_data = zero_bias.data();
    }

    KLEIDIAI_KERNEL_LOG("kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme" << " N=" << filter_count << " k_chunk_count=" << k_chunk_count << " k_chunk_length=" << input_channels << " rhs_stride_row=" << (filter_count * sizeof(MLAS_FP16)));
    kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
        filter_count,
        k_chunk_count,
        input_channels,
        filter_count * sizeof(MLAS_FP16),
        reordered_filter.data(),
        bias_data,
        packed_filter
    );

    return true;
}

bool
ConvolveSme(
    const MLAS_CONV_PARAMETERS* parameters,
    const MLAS_FP16* input,
    const MLAS_FP16* filter,
    bool filter_and_bias_are_packed,
    const MLAS_FP16* bias,
    MLAS_FP16* working_buffer,
    MLAS_FP16* output,
    MLAS_THREADPOOL* thread_pool
)
{
    if (input == nullptr || filter == nullptr || output == nullptr ||
        (!parameters->InputOutputChannelsLast && working_buffer == nullptr)) {
        return false;
    }

    size_t d_kh = 0;
    size_t d_kw = 0;
    size_t output_size = 0;
    if (!TryComputeKernelSize(parameters->DilationShape[0], parameters->KernelShape[0], d_kh) ||
        !TryComputeKernelSize(parameters->DilationShape[1], parameters->KernelShape[1], d_kw) ||
        !TryComputeOutputSize(
            parameters->InputShape[0],
            parameters->InputShape[1],
            d_kh,
            d_kw,
            parameters->Padding[0],
            parameters->Padding[1],
            parameters->StrideShape[0],
            parameters->StrideShape[1],
            output_size)) {
        return false;
    }

    std::vector<MLAS_FP16> input_nhwc;
    std::vector<MLAS_FP16> pad;
    std::vector<const void*> indirection;
    if (!PrepareLhsInput(parameters, input, output_size, input_nhwc, pad, indirection)) {
        return false;
    }

    std::vector<std::byte> packed_filter_buffer;
    const std::byte* packed_filter = reinterpret_cast<const std::byte*>(filter);
    if (!filter_and_bias_are_packed) {
        const std::array<int64_t, 2> kernel_shape{
            static_cast<int64_t>(parameters->KernelShape[0]),
            static_cast<int64_t>(parameters->KernelShape[1])
        };
        const std::array<int64_t, 2> dilation_shape{
            static_cast<int64_t>(parameters->DilationShape[0]),
            static_cast<int64_t>(parameters->DilationShape[1])
        };
        size_t packed_filter_size = 0;
        if (!GetPackedFilterSize(
                parameters->FilterCount,
                parameters->InputChannels,
                kernel_shape.data(),
                dilation_shape.data(),
                &packed_filter_size
            )) {
            return false;
        }
        packed_filter_buffer.resize(packed_filter_size);
        if (!PackFilter(
                parameters->FilterCount,
                parameters->InputChannels,
                kernel_shape.data(),
                dilation_shape.data(),
                filter,
                bias,
                packed_filter_buffer.data()
            )) {
            return false;
        }
        packed_filter = packed_filter_buffer.data();
    }

    const auto& imatmul = GetKleidiAIF16IMatmulUKernel();
    const size_t base_n_step = imatmul.ukernel.get_n_step();
    const size_t base_m_step = imatmul.ukernel.get_m_step();
    size_t n_step = base_n_step;
    size_t m_step = base_m_step;
    const size_t filter_count = parameters->FilterCount;
    const size_t input_channels = parameters->InputChannels;

    std::array<size_t, 2> dim{
        MlasDivRoundup(output_size, m_step),
        MlasDivRoundup(filter_count, n_step)
    };

    size_t tile_count = 0;
    if (MlasMultiplyOverflowsSizeT(dim[0], dim[1], &tile_count)) {
        return false;
    }

    const size_t required_tiles = std::min(
        static_cast<size_t>(MlasGetMaximumThreadCount(thread_pool)),
        tile_count
    );

    if (required_tiles == 0) {
        return false;
    }

    const size_t original_dim0 = dim[0];
    const size_t original_dim1 = dim[1];
    size_t scaled_dim0 = 0;
    size_t scaled_dim1 = 0;
    if (MlasMultiplyOverflowsSizeT(required_tiles, original_dim0, &scaled_dim0) ||
        MlasMultiplyOverflowsSizeT(required_tiles, original_dim1, &scaled_dim1)) {
        return false;
    }

    dim[0] = MlasDivRoundup(scaled_dim0, tile_count);
    dim[1] = MlasDivRoundup(scaled_dim1, tile_count);

    size_t new_m_step = 0;
    size_t new_n_step = 0;
    if (MlasMultiplyOverflowsSizeT(m_step, MlasDivRoundup(MlasDivRoundup(output_size, dim[0]), m_step), &new_m_step) ||
        MlasMultiplyOverflowsSizeT(n_step, MlasDivRoundup(MlasDivRoundup(filter_count, dim[1]), n_step), &new_n_step)) {
        return false;
    }
    m_step = new_m_step;
    n_step = new_n_step;

    dim[0] = MlasDivRoundup(output_size, m_step);
    dim[1] = MlasDivRoundup(filter_count, n_step);
    size_t finalized_tile_count = 0;
    if (MlasMultiplyOverflowsSizeT(dim[0], dim[1], &finalized_tile_count)) {
        return false;
    }

    const float clamp_min = -std::numeric_limits<float>::infinity();
    const float clamp_max = std::numeric_limits<float>::infinity();
    size_t dst_stride = 0;
    if (MlasMultiplyOverflowsSizeT(filter_count, sizeof(MLAS_FP16), &dst_stride)) {
        return false;
    }

    MLAS_FP16* destination = parameters->InputOutputChannelsLast ? output : working_buffer;

    size_t kernel_chunk_count = 0;
    size_t effective_k = 0;
    if (MlasMultiplyOverflowsSizeT(d_kh, d_kw, &kernel_chunk_count) ||
        MlasMultiplyOverflowsSizeT(kernel_chunk_count, input_channels, &effective_k)) {
        return false;
    }

    const size_t full_lhs_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
        output_size, kernel_chunk_count, input_channels
    );
    if (full_lhs_size == 0) {
        return false;
    }

    const size_t maximum_lhs_chunk_bytes = SelectMaximumLhsChunkBytes(
        full_lhs_size,
        filter_count,
        effective_k
    );

    if (maximum_lhs_chunk_bytes == 0 || full_lhs_size <= maximum_lhs_chunk_bytes) {
        std::vector<std::byte> packed_lhs;
        packed_lhs.resize(full_lhs_size);

        KLEIDIAI_KERNEL_LOG("kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme"
                            << " M=" << output_size
                            << " k_chunk_count=" << kernel_chunk_count
                            << " k_chunk_length=" << input_channels);
        kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
            output_size,
            kernel_chunk_count,
            input_channels,
            indirection.data(),
            0,
            pad.data(),
            packed_lhs.data()
        );

        std::atomic<bool> ok{true};
        MlasTrySimpleParallel(thread_pool, static_cast<ptrdiff_t>(finalized_tile_count), [&](ptrdiff_t tid) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }

            const size_t m_idx = (static_cast<size_t>(tid) / dim[1]) * m_step;
            const size_t n_idx = (static_cast<size_t>(tid) % dim[1]) * n_step;
            const size_t tile_m = std::min(m_step, output_size - m_idx);
            const size_t tile_n = std::min(n_step, filter_count - n_idx);

            const std::byte* lhs_tile =
                packed_lhs.data() + imatmul.ukernel.get_lhs_packed_offset(m_idx, kernel_chunk_count, input_channels);
            const std::byte* rhs_tile =
                packed_filter + imatmul.ukernel.get_rhs_packed_offset(n_idx, kernel_chunk_count, input_channels);

            size_t dst_elements = 0;
            size_t dst_bytes = 0;
            if (MlasMultiplyOverflowsSizeT(m_idx, filter_count, &dst_elements) ||
                MlasAddOverflowsSizeT(dst_elements, n_idx, &dst_elements) ||
                MlasMultiplyOverflowsSizeT(dst_elements, sizeof(MLAS_FP16), &dst_bytes)) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }

            std::byte* dst_tile = reinterpret_cast<std::byte*>(destination) + dst_bytes;

            KLEIDIAI_KERNEL_LOG(imatmul.name << " M=" << tile_m
                                             << " N=" << tile_n
                                             << " k_chunk_count=" << kernel_chunk_count
                                             << " k_chunk_length=" << input_channels);
            imatmul.ukernel.run_imatmul(
                tile_m,
                tile_n,
                kernel_chunk_count,
                input_channels,
                lhs_tile,
                rhs_tile,
                dst_tile,
                dst_stride,
                clamp_min,
                clamp_max
            );
        });

        if (!ok.load(std::memory_order_relaxed)) {
            return false;
        }
    } else {
        const size_t bytes_per_m_step = kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
            base_m_step, kernel_chunk_count, input_channels
        );
        if (bytes_per_m_step == 0) {
            return false;
        }

        const size_t max_m_steps_per_chunk = std::min(
            MlasDivRoundup(output_size, base_m_step),
            std::max<size_t>(1, maximum_lhs_chunk_bytes / bytes_per_m_step)
        );
        size_t lhs_chunk_m = 0;
        if (MlasMultiplyOverflowsSizeT(base_m_step, max_m_steps_per_chunk, &lhs_chunk_m)) {
            return false;
        }

        const size_t lhs_chunk_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
            lhs_chunk_m, kernel_chunk_count, input_channels
        );
        if (lhs_chunk_size == 0 || lhs_chunk_size > std::vector<std::byte>().max_size()) {
            return false;
        }

        const size_t m_chunk_count = MlasDivRoundup(output_size, lhs_chunk_m);
        std::atomic<bool> ok{true};
        MlasTrySimpleParallel(thread_pool, static_cast<ptrdiff_t>(m_chunk_count), [&](ptrdiff_t tid) {
            if (!ok.load(std::memory_order_relaxed)) {
                return;
            }

            const size_t global_m_idx = static_cast<size_t>(tid) * lhs_chunk_m;
            const size_t chunk_m = std::min(lhs_chunk_m, output_size - global_m_idx);
            size_t indirection_offset = 0;
            size_t indirection_tiles = 0;
            if (MlasMultiplyOverflowsSizeT(global_m_idx / base_m_step, kernel_chunk_count, &indirection_tiles) ||
                MlasMultiplyOverflowsSizeT(indirection_tiles, base_m_step, &indirection_offset)) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }

            auto& packed_lhs = g_kai_half_conv_tls.packed_lhs;
            ScopedKaiHalfConvTlsCleanup cleanup{g_kai_half_conv_tls};
            if (!TryResizeVector(packed_lhs, lhs_chunk_size)) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }

            KLEIDIAI_KERNEL_LOG("kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme"
                                << " M=" << chunk_m
                                << " k_chunk_count=" << kernel_chunk_count
                                << " k_chunk_length=" << input_channels);
            kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme(
                chunk_m,
                kernel_chunk_count,
                input_channels,
                indirection.data() + indirection_offset,
                0,
                pad.data(),
                packed_lhs.data()
            );

            for (size_t n_tile_idx = 0; n_tile_idx < dim[1]; ++n_tile_idx) {
                const size_t n_idx = n_tile_idx * n_step;
                const size_t tile_n = std::min(n_step, filter_count - n_idx);
                const std::byte* rhs_tile =
                    packed_filter + imatmul.ukernel.get_rhs_packed_offset(n_idx, kernel_chunk_count, input_channels);

                size_t dst_elements = 0;
                size_t dst_bytes = 0;
                if (MlasMultiplyOverflowsSizeT(global_m_idx, filter_count, &dst_elements) ||
                    MlasAddOverflowsSizeT(dst_elements, n_idx, &dst_elements) ||
                    MlasMultiplyOverflowsSizeT(dst_elements, sizeof(MLAS_FP16), &dst_bytes)) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                std::byte* dst_tile = reinterpret_cast<std::byte*>(destination) + dst_bytes;

                KLEIDIAI_KERNEL_LOG(imatmul.name << " M=" << chunk_m
                                                 << " N=" << tile_n
                                                 << " k_chunk_count=" << kernel_chunk_count
                                                 << " k_chunk_length=" << input_channels);
                imatmul.ukernel.run_imatmul(
                    chunk_m,
                    tile_n,
                    kernel_chunk_count,
                    input_channels,
                    packed_lhs.data(),
                    rhs_tile,
                    dst_tile,
                    dst_stride,
                    clamp_min,
                    clamp_max
                );
            }
        });

        if (!ok.load(std::memory_order_relaxed)) {
            return false;
        }
    }

    if (parameters->InputOutputChannelsLast) {
        return true;
    }

    MlasTranspose(working_buffer, output, output_size, filter_count, thread_pool);
    return true;
}

}  // namespace

bool
    MLASCALL
    ArmKleidiAI::MlasHalfConvPrepare(
        MLAS_CONV_PARAMETERS* Parameters,
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
        bool InputOutputChannelsLast,
        MLAS_THREADPOOL* ThreadPool,
        const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    if (Parameters == nullptr ||
        InputShape == nullptr ||
        KernelShape == nullptr ||
        DilationShape == nullptr ||
        Padding == nullptr ||
        StrideShape == nullptr ||
        OutputShape == nullptr ||
        WorkingBufferSize == nullptr ||
        Dimensions != 2) {
        return false;
    }

    if (BackendKernelSelectorConfig != nullptr &&
        !BackendKernelSelectorConfig->use_kleidiai) {
        KLEIDIAI_DEBUG_LOG("User explicitly disabled KleidiAI, returning false from MlasHalfConvPrepare.");
        return false;
    }

    Parameters->BackendKernelSelectorConfig = BackendKernelSelectorConfig;
    Parameters->Activation = Activation;
    Parameters->Dimensions = Dimensions;
    Parameters->BatchCount = BatchCount;
    Parameters->GroupCount = GroupCount;
    Parameters->InputChannels = InputChannels;
    Parameters->FilterCount = FilterCount;
    Parameters->Beta = Beta;
    Parameters->ChannelsLast = InputOutputChannelsLast;
    Parameters->InputOutputChannelsLast = InputOutputChannelsLast;

    size_t input_size = 1;
    size_t output_size = 1;
    size_t k = InputChannels;
    size_t dilated_kernel_size = InputChannels;
    for (size_t dim = 0; dim < Dimensions; ++dim) {
        if (InputShape[dim] <= 0 ||
            OutputShape[dim] <= 0 ||
            KernelShape[dim] <= 0 ||
            DilationShape[dim] <= 0 ||
            StrideShape[dim] <= 0 ||
            Padding[dim] < 0 ||
            Padding[dim + Dimensions] < 0) {
            return false;
        }

        Parameters->InputShape[dim] = static_cast<size_t>(InputShape[dim]);
        Parameters->OutputShape[dim] = static_cast<size_t>(OutputShape[dim]);
        Parameters->KernelShape[dim] = static_cast<size_t>(KernelShape[dim]);
        Parameters->DilationShape[dim] = static_cast<size_t>(DilationShape[dim]);
        Parameters->Padding[dim] = static_cast<size_t>(Padding[dim]);
        Parameters->Padding[dim + Dimensions] = static_cast<size_t>(Padding[dim + Dimensions]);
        Parameters->StrideShape[dim] = static_cast<size_t>(StrideShape[dim]);

        size_t dilated_kernel_dim = 0;
        if (!TryComputeKernelSize(Parameters->DilationShape[dim], Parameters->KernelShape[dim], dilated_kernel_dim)) {
            return false;
        }

        if (MlasMultiplyOverflowsSizeT(input_size, Parameters->InputShape[dim], &input_size) ||
            MlasMultiplyOverflowsSizeT(output_size, Parameters->OutputShape[dim], &output_size) ||
            MlasMultiplyOverflowsSizeT(k, Parameters->KernelShape[dim], &k) ||
            MlasMultiplyOverflowsSizeT(dilated_kernel_size, dilated_kernel_dim, &dilated_kernel_size)) {
            return false;
        }
    }

    Parameters->InputSize = input_size;
    Parameters->OutputSize = output_size;
    Parameters->K = k;
    Parameters->ThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (!CheckCapabilitiesSme(Parameters)) {
        return false;
    }

    size_t working_elements = 0;
    if (MlasMultiplyOverflowsSizeT(Parameters->OutputSize, Parameters->FilterCount, &working_elements)) {
        return false;
    }
    if (Parameters->InputOutputChannelsLast) {
        *WorkingBufferSize = 0;
        return true;
    }

    if (MlasMultiplyOverflowsSizeT(working_elements, sizeof(MLAS_FP16), WorkingBufferSize)) {
        return false;
    }
    return true;
}

bool
    MLASCALL
    ArmKleidiAI::MlasHalfConv(
        const MLAS_CONV_PARAMETERS* Parameters,
        const MLAS_FP16* Input,
        const MLAS_FP16* Filter,
        bool FilterAndBiasArePacked,
        const MLAS_FP16* Bias,
        MLAS_FP16* WorkingBuffer,
        MLAS_FP16* Output,
        MLAS_THREADPOOL* ThreadPool
    )
{
    if (!CheckCapabilitiesSme(Parameters)) {
        return false;
    }

    return ConvolveSme(Parameters, Input, Filter, FilterAndBiasArePacked, Bias, WorkingBuffer, Output, ThreadPool);
}

size_t
    MLASCALL
    ArmKleidiAI::MlasHalfConvPackWeightsAndBiasSize(
        size_t FilterCount,
        size_t InputChannels,
        const int64_t* KernelShape,
        const int64_t* DilationShape
    )
{
    size_t packed_size = 0;
    if (!GetPackedFilterSize(FilterCount, InputChannels, KernelShape, DilationShape, &packed_size)) {
        return 0;
    }
    return packed_size;
}

bool
    MLASCALL
    ArmKleidiAI::MlasHalfConvPackWeightsAndBias(
        size_t FilterCount,
        size_t InputChannels,
        const int64_t* KernelShape,
        const int64_t* DilationShape,
        const MLAS_FP16* Filter,
        const MLAS_FP16* Bias,
        void* PackedWeightsAndBias,
        MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    return PackFilter(
        FilterCount,
        InputChannels,
        KernelShape,
        DilationShape,
        Filter,
        Bias,
        PackedWeightsAndBias
    );
}
