#include "mlasi.h"

// filter 3x3, dilations are all 1. pad is 0 or 1, input_image_width > 2
template <bool IsAccumulate>
static
void
MlasConvSingleChannelAccumulate_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
{
    const float w00 = Filter[0];
    const float w01 = Filter[1];
    const float w02 = Filter[2];
    const float w10 = Filter[3];
    const float w11 = Filter[4];
    const float w12 = Filter[5];
    const float w20 = Filter[6];
    const float w21 = Filter[7];
    const float w22 = Filter[8];

    const size_t pad_top = Parameters->Padding[0];
    const size_t pad_left = Parameters->Padding[1];
    const size_t pad_right = Parameters->Padding[3];
    const size_t H = Parameters->InputShape[0];
    const size_t W = Parameters->InputShape[1];
    const size_t stride_h = Parameters->StrideShape[0];
    const size_t stride_w = Parameters->StrideShape[1];

    const float* row0 = (pad_top > 0) ? Zeros : (Input - pad_left);
    const float* row1 = (Input + (1 - pad_top) * W) - pad_left;
    const float* row2 = (H + pad_top <= 2) ? Zeros : (row1 + W);

    for (size_t h = 0, out_row = Parameters->OutputShape[0]; out_row > 0; --out_row) {
        auto out_col = Parameters->OutputShape[1];

        if (pad_left == 1) {
            float dotsum = w01 * row0[1] + w02 * row0[2] +
                           w11 * row1[1] + w12 * row1[2] +
                           w21 * row2[1] + w22 * row2[2];
            if (IsAccumulate) {
                *Output += dotsum;
            } else {
                *Output = dotsum;
            }
            Output++;
            out_col--;
            row0 += stride_w;
            row1 += stride_w;
            row2 += stride_w;
        }

        for (; out_col > pad_right; out_col--) {
            float dotsum =
                w00 * row0[0] + w01 * row0[1] + w02 * row0[2] +
                w10 * row1[0] + w11 * row1[1] + w12 * row1[2] +
                w20 * row2[0] + w21 * row2[1] + w22 * row2[2];
            if (IsAccumulate) {
                *Output += dotsum;
            } else {
                *Output = dotsum;
            }
            Output++;
            row0 += stride_w;
            row1 += stride_w;
            row2 += stride_w;
        }

        if (out_col == 1) { // pad_right == 1
            float dotsum =
                w00 * row0[0] + w01 * row0[1] +
                w10 * row1[0] + w11 * row1[1] +
                w20 * row2[0] + w21 * row2[1];
            if (IsAccumulate) {
                *Output += dotsum;
            } else {
                *Output = dotsum;
            }
            Output++;
        }

        h += stride_h;
        row0 = (Input + (h - pad_top) * W) - pad_left;
        row1 = row0 + W;
        row2 = (h + 2 >= H + pad_top) ? Zeros : (row1 + W);
    }
}


// filter 3x3, dilations are all 1. pad is 0 or 1, input_image_width > 2
void
MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
{
    MlasConvSingleChannelAccumulate_CHW<false>(Parameters, Input, Filter, Output, Zeros);
}


// filter 3x3, dilations are all 1. pad is 0 or 1
void
MlasConvDirectFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
{
    const auto kernel_size = Parameters->KernelShape[0] * Parameters->KernelShape[1];
    const auto filter_size = kernel_size * Parameters->InputChannels;

    auto input_channel = Parameters->InputChannels;
    if (input_channel > 0) {
        const auto* filter = Filter;
        auto* output = Output;
        for (auto out_channel = Parameters->FilterCount; out_channel > 0; out_channel--) {
            MlasConvSingleChannelAccumulate_CHW<false>(Parameters, Input, filter, output, Zeros);
            filter += filter_size;
            output += Parameters->OutputSize;
        }
        Input += Parameters->InputSize;
        Filter += kernel_size;
        input_channel--;
    }

    for (; input_channel > 0; input_channel--) {
        const auto* filter = Filter;
        auto* output = Output;
        for (auto out_channel = Parameters->FilterCount; out_channel > 0; out_channel--) {
            MlasConvSingleChannelAccumulate_CHW<true>(Parameters, Input, filter, output, Zeros);
            filter += filter_size;
            output += Parameters->OutputSize;
        }
        Input += Parameters->InputSize;
        Filter += kernel_size;    
    }
}
