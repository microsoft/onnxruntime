// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

#if defined(MLAS_TARGET_ARM64) && defined(MLAS_USE_ARM_NEON_NCHWC)

#include <algorithm>

#include "../../../core/mlas/lib/mlasi.h"
#include "../../../core/mlas/lib/sconv_nchwc_kernel_neon.h"

class MlasNchwcConvKernelTest : public MlasTestBase {
 private:
  static constexpr size_t BlockSize = 16;

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferFilter;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;

  template <typename FillFunc>
  static void FillBuffer(float* buffer, size_t count, FillFunc&& fill_func) {
    for (size_t i = 0; i < count; ++i) {
      buffer[i] = fill_func(i);
    }
  }

  static void ReferenceKernel(const float* Input,
                              const float* Filter,
                              float* Output,
                              size_t StrideWidth,
                              size_t DilationWidth,
                              size_t KernelHeight,
                              size_t KernelWidth,
                              size_t InputWidth,
                              size_t DilatedInputWidth,
                              size_t OutputCount,
                              const float* Bias,
                              unsigned KernelFlags) {
    const size_t stride_width_elements = StrideWidth / sizeof(float);
    const size_t dilation_width_elements = DilationWidth / sizeof(float);
    const size_t dilated_input_width_elements = DilatedInputWidth / sizeof(float);

    for (size_t output_idx = 0; output_idx < OutputCount; ++output_idx) {
      float accumulator[BlockSize]{};

      for (size_t kh = 0; kh < KernelHeight; ++kh) {
        for (size_t kw = 0; kw < KernelWidth; ++kw) {
          const float* input_base = Input + output_idx * stride_width_elements +
                                    kh * dilated_input_width_elements +
                                    kw * dilation_width_elements;
          const size_t kernel_base_pos = kh * (KernelWidth * BlockSize * BlockSize) +
                                         kw * (BlockSize * BlockSize);

          for (size_t input_lane = 0; input_lane < BlockSize; ++input_lane) {
            const float input_value = input_base[input_lane];
            const float* filter_row = Filter + kernel_base_pos + input_lane * BlockSize;

            for (size_t output_lane = 0; output_lane < BlockSize; ++output_lane) {
              accumulator[output_lane] += input_value * filter_row[output_lane];
            }
          }
        }
      }

      float* output_block = Output + output_idx * BlockSize;

      for (size_t output_lane = 0; output_lane < BlockSize; ++output_lane) {
        float value = accumulator[output_lane];

        if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0) {
          value += output_block[output_lane];
        }
        if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0) {
          value += Bias[output_lane];
        }
        if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0) {
          value = std::max(value, 0.0f);
        }

        output_block[output_lane] = value;
      }
    }

    MLAS_UNREFERENCED_PARAMETER(InputWidth);
  }

  void TestKernel(size_t OutputCount,
                  size_t KernelHeight,
                  size_t KernelWidth,
                  unsigned KernelFlags) {
    const size_t InputWidth = OutputCount + KernelWidth - 1;
    const size_t InputElements = KernelHeight * InputWidth * BlockSize;
    const size_t FilterElements = KernelHeight * KernelWidth * BlockSize * BlockSize;
    const size_t OutputElements = OutputCount * BlockSize;
    const size_t BiasElements = BlockSize;

    float* Input = BufferInput.GetFilledBuffer(InputElements, [](float* start, size_t count) {
      FillBuffer(start, count, [](size_t i) {
        return float((int(i * 7 % 23) - 11)) / 8.0f;
      });
    });
    float* Filter = BufferFilter.GetFilledBuffer(FilterElements, [](float* start, size_t count) {
      FillBuffer(start, count, [](size_t i) {
        return float((int(i * 5 % 29) - 14)) / 9.0f;
      });
    });
    float* Bias = BufferBias.GetFilledBuffer(BiasElements, [](float* start, size_t count) {
      FillBuffer(start, count, [](size_t i) {
        return float((int(i % 9) - 4)) / 8.0f;
      });
    });
    float* Output = BufferOutput.GetFilledBuffer(OutputElements, [](float* start, size_t count) {
      FillBuffer(start, count, [](size_t i) {
        return float((int(i * 3 % 17) - 8)) / 7.0f;
      });
    });
    float* OutputReference = BufferOutputReference.GetFilledBuffer(OutputElements, [](float* start, size_t count) {
      FillBuffer(start, count, [](size_t i) {
        return float((int(i * 3 % 17) - 8)) / 7.0f;
      });
    });

    const size_t StrideWidthBytes = BlockSize * sizeof(float);
    const size_t DilationWidthBytes = BlockSize * sizeof(float);
    const size_t InputWidthBytes = BlockSize * InputWidth * sizeof(float);
    const size_t DilatedInputWidthBytes = InputWidthBytes;
    const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;
    const size_t FilterStrideBytes = FilterElements * sizeof(float);
    const size_t OutputStrideBytes = OutputCount * BlockSize * sizeof(float);

    ReferenceKernel(Input,
                    Filter,
                    OutputReference,
                    StrideWidthBytes,
                    DilationWidthBytes,
                    KernelHeight,
                    KernelWidth,
                    InputWidthBytes,
                    DilatedInputWidthBytes,
                    OutputCount,
                    Bias,
                    KernelFlags);

    MlasConvNchwcFloatKernelNeon(Input,
                                 Filter,
                                 Output,
                                 StrideWidthBytes,
                                 DilationWidthBytes,
                                 1,
                                 InputStrideBytes,
                                 FilterStrideBytes,
                                 OutputStrideBytes,
                                 KernelHeight,
                                 KernelWidth,
                                 Input,
                                 InputWidthBytes,
                                 DilatedInputWidthBytes,
                                 0,
                                 OutputCount,
                                 0,
                                 Bias,
                                 KernelFlags);

    ASSERT_EQ(memcmp(Output, OutputReference, OutputElements * sizeof(float)), 0)
        << "OutputCount=" << OutputCount
        << "/KH=" << KernelHeight
        << "/KW=" << KernelWidth
        << "/Flags=" << KernelFlags;
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Conv2dNchwcKernel");
    return suite_name.c_str();
  }

  void ExecuteShort() override {
    if (MlasNchwcGetBlockSize() != BlockSize) {
      return;
    }

    TestKernel(1, 1, 1, 0);
    TestKernel(1, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION);
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT);
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT |
                           MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION |
                           MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION);
    TestKernel(3, 1, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT |
                           MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION |
                           MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION);
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return (MlasNchwcGetBlockSize() > 1 && is_short_execute)
             ? MlasDirectShortExecuteTests<MlasNchwcConvKernelTest>::RegisterShortExecute()
             : 0;
});

#endif
