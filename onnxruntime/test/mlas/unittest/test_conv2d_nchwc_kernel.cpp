// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

#if defined(MLAS_TARGET_ARM64) && defined(MLAS_USE_ARM_NEON_NCHWC)

#include <algorithm>
#include <cstdio>

#include "../../../core/mlas/lib/mlasi.h"
#include "../../../core/mlas/lib/sconv_nchwc_kernel_neon.h"

class MlasNchwcConvKernelTest : public MlasTestBase {
 private:
  static constexpr size_t BlockSize = 16;

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferFilter;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputCpp;
  MatrixGuardBuffer<float> BufferOutputAsm;
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
                              const float* InputBase,
                              size_t InputWidth,
                              size_t DilatedInputWidth,
                              size_t FilterCount,
                              size_t OutputStride,
                              size_t OutputCountLeftPad,
                              size_t OutputCount,
                              size_t OutputCountRightPad,
                              const float* Bias,
                              unsigned KernelFlags) {
    const size_t stride_width_elements = StrideWidth / sizeof(float);
    const size_t dilation_width_elements = DilationWidth / sizeof(float);
    const size_t input_width_elements = InputWidth / sizeof(float);
    const size_t dilated_input_width_elements = DilatedInputWidth / sizeof(float);
    const size_t output_stride_elements = OutputStride / sizeof(float);
    const size_t filter_stride_elements = KernelHeight * KernelWidth * BlockSize * BlockSize;
    const size_t total_output_count = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t filter_set = 0; filter_set < FilterCount; ++filter_set) {
      const float* filter_block = Filter + filter_set * filter_stride_elements;
      float* output_block_base = Output + filter_set * output_stride_elements;
      const float* bias_block = Bias + filter_set * BlockSize;

      for (size_t output_idx = 0; output_idx < total_output_count; ++output_idx) {
        float accumulator[BlockSize]{};

        for (size_t kh = 0; kh < KernelHeight; ++kh) {
          for (size_t kw = 0; kw < KernelWidth; ++kw) {
            const float* input_base = Input + output_idx * stride_width_elements +
                                      kh * dilated_input_width_elements +
                                      kw * dilation_width_elements;
            const size_t kernel_base_pos = kh * (KernelWidth * BlockSize * BlockSize) +
                                           kw * (BlockSize * BlockSize);
            const float* input_row_start = InputBase + kh * dilated_input_width_elements;
            const float* input_row_end = input_row_start + input_width_elements;

            for (size_t input_lane = 0; input_lane < BlockSize; ++input_lane) {
              const float* input_element = input_base + input_lane;
              const float input_value = (input_element >= input_row_start && input_element < input_row_end)
                                            ? *input_element
                                            : 0.0f;
              const float* filter_row = filter_block + kernel_base_pos + input_lane * BlockSize;

              for (size_t output_lane = 0; output_lane < BlockSize; ++output_lane) {
                accumulator[output_lane] += input_value * filter_row[output_lane];
              }
            }
          }
        }

        float* output_block = output_block_base + output_idx * BlockSize;

        for (size_t output_lane = 0; output_lane < BlockSize; ++output_lane) {
          float value = accumulator[output_lane];

          if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0) {
            value += output_block[output_lane];
          }
          if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0) {
            value += bias_block[output_lane];
          }
          if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0) {
            value = std::max(value, 0.0f);
          }

          output_block[output_lane] = value;
        }
      }
    }
  }

  void AssertClose(const float* actual,
                   const float* expected,
                   size_t count,
                   const char* actual_label,
                   const char* expected_label,
                   size_t FilterCount,
                   size_t OutputCountLeftPad,
                   size_t OutputCount,
                   size_t OutputCountRightPad,
                   size_t KernelHeight,
                   size_t KernelWidth,
                   unsigned KernelFlags) {
    for (size_t i = 0; i < count; ++i) {
      ASSERT_TRUE(CloseEnough(actual[i], expected[i]))
          << actual_label << " vs " << expected_label
          << " @" << i
          << " got=" << actual[i]
          << " expected=" << expected[i]
          << " FilterCount=" << FilterCount
          << " LeftPad=" << OutputCountLeftPad
          << " OutputCount=" << OutputCount
          << " RightPad=" << OutputCountRightPad
          << "/KH=" << KernelHeight
          << "/KW=" << KernelWidth
          << "/Flags=" << KernelFlags;
    }
  }

  void TestKernel(size_t OutputCount,
                  size_t KernelHeight,
                  size_t KernelWidth,
                  unsigned KernelFlags,
                  size_t FilterCount = 1,
                  size_t OutputCountLeftPad = 0,
                  size_t OutputCountRightPad = 0) {
    std::fprintf(stderr,
                 "Start case FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                 FilterCount,
                 OutputCountLeftPad,
                 OutputCount,
                 OutputCountRightPad,
                 KernelHeight,
                 KernelWidth,
                 KernelFlags);
    std::fflush(stderr);

    const size_t InputWidth = OutputCount + KernelWidth - 1;
    const size_t TotalInputWidth = OutputCountLeftPad + InputWidth + OutputCountRightPad;
    const size_t InputElements = KernelHeight * TotalInputWidth * BlockSize;
    const size_t FilterElementsPerBlock = KernelHeight * KernelWidth * BlockSize * BlockSize;
    const size_t FilterElements = FilterCount * FilterElementsPerBlock;
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;
    const size_t OutputStrideElements = TotalOutputCount * BlockSize;
    const size_t OutputElements = FilterCount * OutputStrideElements;
    const size_t BiasElements = FilterCount * BlockSize;

    float* InputStorage = BufferInput.GetFilledBuffer(InputElements, [](float* start, size_t count) {
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
    float* OutputCpp = BufferOutputCpp.GetFilledBuffer(OutputElements, [](float* start, size_t count) {
      FillBuffer(start, count, [](size_t i) {
        return float((int(i * 3 % 17) - 8)) / 7.0f;
      });
    });
    float* OutputAsm = BufferOutputAsm.GetFilledBuffer(OutputElements, [](float* start, size_t count) {
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
    const size_t StrideWidthElements = StrideWidthBytes / sizeof(float);
    const size_t InputWidthBytes = BlockSize * InputWidth * sizeof(float);
    const size_t DilatedInputWidthBytes = BlockSize * TotalInputWidth * sizeof(float);
    const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;
    const size_t FilterStrideBytes = FilterElementsPerBlock * sizeof(float);
    const size_t OutputStrideBytes = OutputStrideElements * sizeof(float);
    const float* InputBase = InputStorage + OutputCountLeftPad * StrideWidthElements;
    const float* Input = InputBase - OutputCountLeftPad * StrideWidthElements;

    ReferenceKernel(Input,
                    Filter,
                    OutputReference,
                    StrideWidthBytes,
                    DilationWidthBytes,
                    KernelHeight,
                    KernelWidth,
                    InputBase,
                    InputWidthBytes,
                    DilatedInputWidthBytes,
                    FilterCount,
                    OutputStrideBytes,
                    OutputCountLeftPad,
                    OutputCount,
                    OutputCountRightPad,
                    Bias,
                    KernelFlags);

    std::fprintf(stderr,
                 "Completed reference FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                 FilterCount,
                 OutputCountLeftPad,
                 OutputCount,
                 OutputCountRightPad,
                 KernelHeight,
                 KernelWidth,
                 KernelFlags);
    std::fflush(stderr);

    MlasConvNchwcFloatKernelNeonCpp(Input,
                                    Filter,
                                    OutputCpp,
                                    StrideWidthBytes,
                                    DilationWidthBytes,
                                    FilterCount,
                                    InputStrideBytes,
                                    FilterStrideBytes,
                                    OutputStrideBytes,
                                    KernelHeight,
                                    KernelWidth,
                                    InputBase,
                                    InputWidthBytes,
                                    DilatedInputWidthBytes,
                                    OutputCountLeftPad,
                                    OutputCount,
                                    OutputCountRightPad,
                                    Bias,
                                    KernelFlags);

    std::fprintf(stderr,
                 "Completed cpp FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                 FilterCount,
                 OutputCountLeftPad,
                 OutputCount,
                 OutputCountRightPad,
                 KernelHeight,
                 KernelWidth,
                 KernelFlags);
    std::fflush(stderr);

    MlasConvNchwcFloatKernelNeon(Input,
                                 Filter,
                                 Output,
                                 StrideWidthBytes,
                                 DilationWidthBytes,
                                 FilterCount,
                                 InputStrideBytes,
                                 FilterStrideBytes,
                                 OutputStrideBytes,
                                 KernelHeight,
                                 KernelWidth,
                                 InputBase,
                                 InputWidthBytes,
                                 DilatedInputWidthBytes,
                                 OutputCountLeftPad,
                                 OutputCount,
                                 OutputCountRightPad,
                                 Bias,
                                 KernelFlags);

    std::fprintf(stderr,
                 "Completed wrapper FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                 FilterCount,
                 OutputCountLeftPad,
                 OutputCount,
                 OutputCountRightPad,
                 KernelHeight,
                 KernelWidth,
                 KernelFlags);
    std::fflush(stderr);

#if !defined(_WIN32)
    if (OutputCountLeftPad == 0 && OutputCountRightPad == 0) {
      std::fprintf(stderr,
                   "Calling asm FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                   FilterCount,
                   OutputCountLeftPad,
                   OutputCount,
                   OutputCountRightPad,
                   KernelHeight,
                   KernelWidth,
                   KernelFlags);
      std::fflush(stderr);

      MlasConvNchwcFloatKernelNeonAsm(Input,
                                      Filter,
                                      OutputAsm,
                                      StrideWidthBytes,
                                      DilationWidthBytes,
                                      FilterCount,
                                      InputStrideBytes,
                                      FilterStrideBytes,
                                      OutputStrideBytes,
                                      KernelHeight,
                                      KernelWidth,
                                      InputBase,
                                      InputWidthBytes,
                                      DilatedInputWidthBytes,
                                      OutputCountLeftPad,
                                      OutputCount,
                                      OutputCountRightPad,
                                      Bias,
                                      KernelFlags);

      std::fprintf(stderr,
                   "Completed asm FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                   FilterCount,
                   OutputCountLeftPad,
                   OutputCount,
                   OutputCountRightPad,
                   KernelHeight,
                   KernelWidth,
                   KernelFlags);
      std::fflush(stderr);
    } else {
      std::fprintf(stderr,
                   "Skipping direct asm FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                   FilterCount,
                   OutputCountLeftPad,
                   OutputCount,
                   OutputCountRightPad,
                   KernelHeight,
                   KernelWidth,
                   KernelFlags);
      std::fflush(stderr);
    }
#endif

    AssertClose(OutputCpp, OutputReference, OutputElements, "cpp", "reference", FilterCount, OutputCountLeftPad, OutputCount, OutputCountRightPad, KernelHeight, KernelWidth, KernelFlags);
    std::fprintf(stderr,
                 "Completed cpp-vs-reference FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                 FilterCount,
                 OutputCountLeftPad,
                 OutputCount,
                 OutputCountRightPad,
                 KernelHeight,
                 KernelWidth,
                 KernelFlags);
    std::fflush(stderr);
    AssertClose(Output, OutputCpp, OutputElements, "wrapper", "cpp", FilterCount, OutputCountLeftPad, OutputCount, OutputCountRightPad, KernelHeight, KernelWidth, KernelFlags);
    std::fprintf(stderr,
                 "Completed wrapper-vs-cpp FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                 FilterCount,
                 OutputCountLeftPad,
                 OutputCount,
                 OutputCountRightPad,
                 KernelHeight,
                 KernelWidth,
                 KernelFlags);
    std::fflush(stderr);
#if !defined(_WIN32)
    if (OutputCountLeftPad == 0 && OutputCountRightPad == 0) {
      AssertClose(OutputAsm, OutputCpp, OutputElements, "asm", "cpp", FilterCount, OutputCountLeftPad, OutputCount, OutputCountRightPad, KernelHeight, KernelWidth, KernelFlags);
      std::fprintf(stderr,
                   "Completed asm-vs-cpp FilterCount=%zu/LeftPad=%zu/OutputCount=%zu/RightPad=%zu/KH=%zu/KW=%zu/Flags=%u\n",
                   FilterCount,
                   OutputCountLeftPad,
                   OutputCount,
                   OutputCountRightPad,
                   KernelHeight,
                   KernelWidth,
                   KernelFlags);
      std::fflush(stderr);
    }
  #endif
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

    // TestKernel(OutputCount, KernelHeight, KernelWidth, KernelFlags, FilterCount)

    // Single-output microkernel coverage.
    TestKernel(1, 1, 1, 0);
    TestKernel(1, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION);

    // Two-output fast path with and without bias.
    TestKernel(2, 1, 1, 0);
    TestKernel(2, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION);

    // Single-output multi-row and multi-column coverage.
    TestKernel(1, 3, 3, 0);
    TestKernel(1, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION);

    // Two-output fast path on a larger spatial kernel.
    TestKernel(2, 3, 3, 0);
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION);

    // Two-output postprocess coverage: accumulate only and accumulate+bias+ReLU.
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT);
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT |
                           MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION |
                           MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION);

    // Three outputs exercise the two-output fast path followed by the one-output tail.
    TestKernel(3, 3, 3, 0);
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION);

    // Three outputs on a 1x3 kernel with full postprocess coverage.
    TestKernel(3, 1, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT |
                           MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION |
                           MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION);

    // FC2 single-output coverage.
    TestKernel(1, 1, 1, 0, 2);
    TestKernel(1, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 2);
    TestKernel(1, 3, 3, 0, 2);
    TestKernel(1, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 2);

    // FC2 two-output fast path.
    TestKernel(2, 1, 1, 0, 2);
    TestKernel(2, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 2);
    TestKernel(2, 3, 3, 0, 2);
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT |
                 MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION |
                 MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION, 2);

    // FC2 tail coverage: two-output fast path followed by one-output tail.
    TestKernel(3, 3, 3, 0, 2);
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 2);

    // FC3 single-output coverage.
    TestKernel(1, 1, 1, 0, 3);
    TestKernel(1, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 3);
    TestKernel(1, 3, 3, 0, 3);
    TestKernel(1, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 3);

    // FC3 multi-output and postprocess coverage.
    TestKernel(2, 1, 1, 0, 3);
    TestKernel(2, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 3);
    TestKernel(2, 3, 3, 0, 3);
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT |
                 MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION |
                 MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION,
           3);

    // FC3 tail coverage.
    TestKernel(3, 3, 3, 0, 3);
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 3);

    // FC4 single-output coverage.
    TestKernel(1, 1, 1, 0, 4);
    TestKernel(1, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 4);
    TestKernel(1, 3, 3, 0, 4);
    TestKernel(1, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 4);

    // FC4 multi-output and postprocess coverage.
    TestKernel(2, 1, 1, 0, 4);
    TestKernel(2, 1, 1, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 4);
    TestKernel(2, 3, 3, 0, 4);
    TestKernel(2, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT |
                 MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION |
                 MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION,
           4);

    // FC4 tail coverage.
    TestKernel(3, 3, 3, 0, 4);
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 4);

    // Padded wrapper coverage: C++ edges with asm interior.
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 1, 1, 1);
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT | MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION | MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION,
               2,
               1,
               1);

    // Asymmetric padded wrapper coverage.
    TestKernel(4, 3, 3, 0, 1, 1, 0);
    TestKernel(4, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 1, 0, 1);
    TestKernel(4, 1, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT | MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION,
               2,
               1,
               0);
    TestKernel(4, 1, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT | MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION | MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION,
               2,
               0,
               1);

    // Wider padded rows make the interior asm span non-trivial.
    TestKernel(8, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 1, 1, 1);
    TestKernel(8, 3, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT | MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION | MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION,
               2,
               1,
               1);

    // FC3 padded coverage uses C++ edges with asm on the interior span.
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 3, 1, 1);
    TestKernel(4, 1, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT | MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION | MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION,
               3,
               1,
               0);

    // FC4 padded coverage uses C++ edges with asm on the interior span.
    TestKernel(3, 3, 3, MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION, 4, 1, 1);
    TestKernel(4, 1, 3, MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT | MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION | MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION,
               4,
               1,
               0);
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return (MlasNchwcGetBlockSize() > 1 && is_short_execute)
             ? MlasDirectShortExecuteTests<MlasNchwcConvKernelTest>::RegisterShortExecute()
             : 0;
});

#endif
