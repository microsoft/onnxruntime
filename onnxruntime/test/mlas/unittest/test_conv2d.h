// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>

#include "test_util.h"

#if defined(MLAS_TARGET_AMD64)
#include "core/mlas/lib/mlasi.h"
#endif

template <bool Threaded>
class MlasConv2DTest : public MlasTestBase {
 protected:
  void MlasConv2DWithOptions(size_t BatchCount,
                             size_t GroupCount,
                             size_t InputChannels,
                             size_t InputHeight,
                             size_t InputWidth,
                             size_t FilterCount,
                             size_t KernelHeight,
                             size_t KernelWidth,
                             size_t PaddingLeftHeight,
                             size_t PaddingLeftWidth,
                             size_t PaddingRightHeight,
                             size_t PaddingRightWidth,
                             size_t DilationHeight,
                             size_t DilationWidth,
                             size_t StrideHeight,
                             size_t StrideWidth,
                             size_t OutputHeight,
                             size_t OutputWidth,
                             const float* Input,
                             const float* Filter,
                             const float* Bias,
                             const MLAS_ACTIVATION& Activation,
                             float Beta,
                             const float* InitialOutput,
                             float* Output) {
    int64_t InputShape[] = {int64_t(InputHeight), int64_t(InputWidth)};
    int64_t KernelShape[] = {int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t DilationShape[] = {int64_t(DilationHeight), int64_t(DilationWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};
    int64_t StrideShape[] = {int64_t(StrideHeight), int64_t(StrideWidth)};
    int64_t OutputShape[] = {int64_t(OutputHeight), int64_t(OutputWidth)};

    const size_t OutputElements = BatchCount * GroupCount * FilterCount * OutputHeight * OutputWidth;
    if (InitialOutput != nullptr) {
      std::memcpy(Output, InitialOutput, OutputElements * sizeof(float));
    } else {
      std::fill_n(Output, OutputElements, 0.0f);
    }

    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize;

    MlasConvPrepare(&Parameters,
                    2,
                    BatchCount,
                    GroupCount,
                    InputChannels,
                    InputShape,
                    KernelShape,
                    DilationShape,
                    Padding,
                    StrideShape,
                    OutputShape,
                    FilterCount,
                    &Activation,
                    &WorkingBufferSize,
                    Beta,
                    threadpool_);

    MlasConv(&Parameters,
             Input,
             Filter,
             Bias,
             BufferWorking.GetBuffer(WorkingBufferSize),
             Output,
             threadpool_);
  }

  virtual void MlasConv2D(size_t BatchCount,
                          size_t GroupCount,
                          size_t InputChannels,
                          size_t InputHeight,
                          size_t InputWidth,
                          size_t FilterCount,
                          size_t KernelHeight,
                          size_t KernelWidth,
                          size_t PaddingLeftHeight,
                          size_t PaddingLeftWidth,
                          size_t PaddingRightHeight,
                          size_t PaddingRightWidth,
                          size_t DilationHeight,
                          size_t DilationWidth,
                          size_t StrideHeight,
                          size_t StrideWidth,
                          size_t OutputHeight,
                          size_t OutputWidth,
                          const float* Input,
                          const float* Filter,
                          const float* Bias,
                          float* Output) {
    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    MlasConv2DWithOptions(BatchCount,
                          GroupCount,
                          InputChannels,
                          InputHeight,
                          InputWidth,
                          FilterCount,
                          KernelHeight,
                          KernelWidth,
                          PaddingLeftHeight,
                          PaddingLeftWidth,
                          PaddingRightHeight,
                          PaddingRightWidth,
                          DilationHeight,
                          DilationWidth,
                          StrideHeight,
                          StrideWidth,
                          OutputHeight,
                          OutputWidth,
                          Input,
                          Filter,
                          Bias,
                          Activation,
                          0.0f,
                          nullptr,
                          Output);
  }

  static float ApplyReferenceActivation(float value, const MLAS_ACTIVATION& Activation) {
    switch (Activation.ActivationKind) {
      case MlasIdentityActivation:
        return value;
      case MlasReluActivation:
        return std::max(value, 0.0f);
      default:
        ADD_FAILURE() << "Unsupported activation kind in Conv2D test reference path: "
                      << static_cast<int>(Activation.ActivationKind);
        return value;
    }
  }

  void ReferenceConv2DWithOptions(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      const MLAS_ACTIVATION& Activation,
      float Beta,
      const float* InitialOutput,
      float* Output) {
    size_t InputSize = InputHeight * InputWidth;
    size_t OutputSize = OutputHeight * OutputWidth;
    size_t KernelSize = KernelHeight * KernelWidth;
    size_t OutputElements = BatchCount * GroupCount * FilterCount * OutputSize;

    size_t K = InputChannels * KernelSize;
    size_t Im2ColElements = OutputSize * K;

    if (InitialOutput != nullptr) {
      std::memcpy(Output, InitialOutput, OutputElements * sizeof(float));
    } else {
      std::fill_n(Output, OutputElements, 0.0f);
    }

    for (size_t b = 0; b < BatchCount; b++) {
      const float* filter = Filter;
      const float* bias = Bias;

      for (size_t g = 0; g < GroupCount; g++) {
        //
        // Transform the image using IM2COL and invoke the GEMM.
        //

        float* Im2Col = BufferIm2Col.GetBuffer(Im2ColElements);
        float* Im2ColOut = Im2Col;

        for (size_t c = 0; c < InputChannels; c++) {
          for (size_t ky = 0; ky < KernelHeight; ky++) {
            for (size_t kx = 0; kx < KernelWidth; kx++) {
              for (size_t oh = 0; oh < OutputHeight; oh++) {
                size_t ih = oh * StrideHeight + ky * DilationHeight - PaddingLeftHeight;

                for (size_t ow = 0; ow < OutputWidth; ow++) {
                  size_t iw = ow * StrideWidth + kx * DilationWidth - PaddingLeftWidth;

                  *Im2ColOut++ = (ih < InputHeight && iw < InputWidth) ? Input[ih * InputWidth + iw] : 0;
                }
              }
            }
          }

          Input += InputSize;
        }

        float* output_group = Output;

        MlasGemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f,
                 filter, K, Im2Col, OutputSize, Beta, output_group, OutputSize, threadpool_, nullptr);

        for (size_t f = 0; f < FilterCount; f++) {
          float biasValue = *bias++;
          float* output_row = output_group + f * OutputSize;

          for (size_t o = 0; o < OutputSize; o++) {
            output_row[o] = ApplyReferenceActivation(output_row[o] + biasValue, Activation);
          }
        }

        filter += FilterCount * InputChannels * KernelSize;
        Output += FilterCount * OutputSize;
      }
    }
  }

  void ReferenceConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output) {
    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    ReferenceConv2DWithOptions(BatchCount,
                               GroupCount,
                               InputChannels,
                               InputHeight,
                               InputWidth,
                               FilterCount,
                               KernelHeight,
                               KernelWidth,
                               PaddingLeftHeight,
                               PaddingLeftWidth,
                               DilationHeight,
                               DilationWidth,
                               StrideHeight,
                               StrideWidth,
                               OutputHeight,
                               OutputWidth,
                               Input,
                               Filter,
                               Bias,
                               Activation,
                               0.0f,
                               nullptr,
                               Output);
  }

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferFilter;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MatrixGuardBuffer<float> BufferInitialOutput;
  MatrixGuardBuffer<float> BufferWorking;
  MatrixGuardBuffer<float> BufferIm2Col;

  MLAS_THREADPOOL* threadpool_;

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name(Threaded ? "Conv2d_Threaded" : "Conv2d_SingleThread");
    return suite_name.c_str();
  }

  MlasConv2DTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

#if defined(MLAS_TARGET_AMD64)
  void TestMobileClipAvx512DispatchSelection(size_t GroupCount,
                                             size_t InputHeight,
                                             size_t InputWidth) {
    if (GetMlasPlatform().ConvNchwFloatKernel != MlasConvNchwFloatKernelAvx512F) {
      return;
    }

    constexpr size_t BatchCount = 1;
    constexpr size_t InputChannels = 1;
    constexpr size_t FilterCount = 2;
    constexpr size_t KernelHeight = 7;
    constexpr size_t KernelWidth = 7;
    constexpr size_t PaddingLeftHeight = 3;
    constexpr size_t PaddingLeftWidth = 3;
    constexpr size_t PaddingRightHeight = 3;
    constexpr size_t PaddingRightWidth = 3;
    constexpr size_t DilationHeight = 1;
    constexpr size_t DilationWidth = 1;
    constexpr size_t StrideHeight = 2;
    constexpr size_t StrideWidth = 2;

    const int64_t OutputHeight64 =
        ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
         (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) /
            int64_t(StrideHeight) +
        1;
    const int64_t OutputWidth64 =
        ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
         (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) /
            int64_t(StrideWidth) +
        1;

    ASSERT_GT(OutputHeight64, 0);
    ASSERT_GT(OutputWidth64, 0);

    int64_t InputShape[] = {int64_t(InputHeight), int64_t(InputWidth)};
    int64_t KernelShape[] = {int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t DilationShape[] = {int64_t(DilationHeight), int64_t(DilationWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};
    int64_t StrideShape[] = {int64_t(StrideHeight), int64_t(StrideWidth)};
    int64_t OutputShape[] = {OutputHeight64, OutputWidth64};

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize = 0;

    MlasConvPrepare(&Parameters,
                    2,
                    BatchCount,
                    GroupCount,
                    InputChannels,
                    InputShape,
                    KernelShape,
                    DilationShape,
                    Padding,
                    StrideShape,
                    OutputShape,
                    FilterCount,
                    &Activation,
                    &WorkingBufferSize,
                    0.0f,
                    threadpool_);

    ASSERT_EQ(Parameters.Algorithm, MlasConvAlgorithmDepthwiseMultiplierGreaterThan1)
        << "Expected AVX512 MobileClip dispatch for G" << GroupCount
        << "/H" << InputHeight
        << "/W" << InputWidth;
  }
#endif

  void TestMobileClipBetaActivationRegression(size_t GroupCount,
                                              size_t InputHeight,
                                              size_t InputWidth) {
    constexpr size_t BatchCount = 1;
    constexpr size_t InputChannels = 1;
    constexpr size_t FilterCount = 2;
    constexpr size_t KernelHeight = 7;
    constexpr size_t KernelWidth = 7;
    constexpr size_t PaddingLeftHeight = 3;
    constexpr size_t PaddingLeftWidth = 3;
    constexpr size_t PaddingRightHeight = 3;
    constexpr size_t PaddingRightWidth = 3;
    constexpr size_t DilationHeight = 1;
    constexpr size_t DilationWidth = 1;
    constexpr size_t StrideHeight = 2;
    constexpr size_t StrideWidth = 2;
    constexpr float Beta = 1.0f;

    const int64_t OutputHeight64 =
        ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
         (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) /
            int64_t(StrideHeight) +
        1;
    const int64_t OutputWidth64 =
        ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
         (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) /
            int64_t(StrideWidth) +
        1;

    ASSERT_GT(OutputHeight64, 0);
    ASSERT_GT(OutputWidth64, 0);

    const size_t OutputHeight = static_cast<size_t>(OutputHeight64);
    const size_t OutputWidth = static_cast<size_t>(OutputWidth64);
    const size_t InputSize = InputHeight * InputWidth;
    const size_t KernelSize = KernelHeight * KernelWidth;
    const size_t OutputSize = OutputHeight * OutputWidth;

    const size_t InputElements = BatchCount * GroupCount * InputChannels * InputSize;
    const size_t FilterElements = GroupCount * FilterCount * InputChannels * KernelSize;
    const size_t BiasElements = GroupCount * FilterCount;
    const size_t OutputElements = BatchCount * GroupCount * FilterCount * OutputSize;

    const float* Input = BufferInput.GetBuffer(InputElements);
    const float* Filter = BufferFilter.GetBuffer(FilterElements);
    const float* Bias = BufferBias.GetBuffer(BiasElements);
    const float* InitialOutput = BufferInitialOutput.GetBuffer(OutputElements);
    float* Output = BufferOutput.GetBuffer(OutputElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputElements);

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasReluActivation;

    MlasConv2DWithOptions(BatchCount,
                          GroupCount,
                          InputChannels,
                          InputHeight,
                          InputWidth,
                          FilterCount,
                          KernelHeight,
                          KernelWidth,
                          PaddingLeftHeight,
                          PaddingLeftWidth,
                          PaddingRightHeight,
                          PaddingRightWidth,
                          DilationHeight,
                          DilationWidth,
                          StrideHeight,
                          StrideWidth,
                          OutputHeight,
                          OutputWidth,
                          Input,
                          Filter,
                          Bias,
                          Activation,
                          Beta,
                          InitialOutput,
                          Output);

    ReferenceConv2DWithOptions(BatchCount,
                               GroupCount,
                               InputChannels,
                               InputHeight,
                               InputWidth,
                               FilterCount,
                               KernelHeight,
                               KernelWidth,
                               PaddingLeftHeight,
                               PaddingLeftWidth,
                               DilationHeight,
                               DilationWidth,
                               StrideHeight,
                               StrideWidth,
                               OutputHeight,
                               OutputWidth,
                               Input,
                               Filter,
                               Bias,
                               Activation,
                               Beta,
                               InitialOutput,
                               OutputReference);

    for (size_t i = 0; i < OutputElements; ++i) {
      ASSERT_TRUE(CloseEnough(Output[i], OutputReference[i]))
          << "Mismatch at output index " << i
          << " for G" << GroupCount
          << "/H" << InputHeight
          << "/W" << InputWidth
          << ": actual=" << Output[i]
          << ", expected=" << OutputReference[i];
    }
  }

  void Test(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth) {
    int64_t OutputHeight64 =
        ((int64_t(InputHeight) + int64_t(PaddingLeftHeight) + int64_t(PaddingRightHeight)) -
         (int64_t(DilationHeight) * (int64_t(KernelHeight) - 1) + 1)) /
            int64_t(StrideHeight) +
        1;
    int64_t OutputWidth64 =
        ((int64_t(InputWidth) + int64_t(PaddingLeftWidth) + int64_t(PaddingRightWidth)) -
         (int64_t(DilationWidth) * (int64_t(KernelWidth) - 1) + 1)) /
            int64_t(StrideWidth) +
        1;

    if (OutputHeight64 <= 0 || OutputWidth64 <= 0) {
      return;
    }

    size_t OutputHeight = size_t(OutputHeight64);
    size_t OutputWidth = size_t(OutputWidth64);

    size_t InputSize = InputHeight * InputWidth;
    size_t KernelSize = KernelHeight * KernelWidth;
    size_t OutputSize = OutputHeight * OutputWidth;

    size_t InputElements = BatchCount * GroupCount * InputChannels * InputSize;
    size_t FilterElements = GroupCount * FilterCount * InputChannels * KernelSize;
    size_t BiasElements = GroupCount * FilterCount;
    size_t OutputElements = BatchCount * GroupCount * FilterCount * OutputSize;

    const float* Input = BufferInput.GetBuffer(InputElements);
    const float* Filter = BufferFilter.GetBuffer(FilterElements);
    const float* Bias = BufferBias.GetBuffer(BiasElements);
    float* Output = BufferOutput.GetBuffer(OutputElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputElements);

    MlasConv2D(BatchCount,
               GroupCount,
               InputChannels,
               InputHeight, InputWidth,
               FilterCount,
               KernelHeight, KernelWidth,
               PaddingLeftHeight, PaddingLeftWidth,
               PaddingRightHeight, PaddingRightWidth,
               DilationHeight, DilationWidth,
               StrideHeight, StrideWidth,
               OutputHeight, OutputWidth,
               Input,
               Filter,
               Bias,
               Output);

    ReferenceConv2D(BatchCount,
                    GroupCount,
                    InputChannels,
                    InputHeight, InputWidth,
                    FilterCount,
                    KernelHeight, KernelWidth,
                    PaddingLeftHeight, PaddingLeftWidth,
                    DilationHeight, DilationWidth,
                    StrideHeight, StrideWidth,
                    OutputHeight, OutputWidth,
                    Input,
                    Filter,
                    Bias,
                    OutputReference);

    ASSERT_EQ(memcmp(Output, OutputReference, OutputElements * sizeof(float)), 0)
        << "B" << BatchCount << "/"
        << "G" << GroupCount << "/"
        << "Cpg" << InputChannels << "/"
        << "Fpg" << FilterCount << "/"
        << "H" << InputHeight << "/"
        << "W" << InputWidth << "/"
        << "KH" << KernelHeight << "/"
        << "KW" << KernelWidth << "/"
        << "Pad" << PaddingLeftHeight << "," << PaddingLeftWidth << "," << PaddingRightHeight << "," << PaddingRightWidth << "/"
        << "Dilation" << DilationHeight << "," << DilationWidth << "/"
        << "Stride" << StrideHeight << "," << StrideWidth;
  }

  void ExecuteLong(void) override {
    static const unsigned cs[] = {32, 14, 1};
    static const unsigned is[] = {53, 11, 5, 1};

    for (unsigned i = 1; i <= 32; i++) {
      Test(4, 18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
      Test(4, 18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
      Test(4, 18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    for (unsigned b = 1; b < 64; b++) {
      Test(b, 1, 64, 11, 11, 128, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    for (unsigned gc = 0; gc < _countof(cs); gc++) {
      for (unsigned ih = 0; ih < _countof(is); ih++) {
        for (unsigned iw = 0; iw < _countof(is); iw++) {
          fprintf(stderr, "Handling depthwise %ux%ux%u\n", cs[gc], is[ih], is[iw]);
          for (unsigned p0 = 0; p0 < 2; p0++) {
            for (unsigned p1 = 0; p1 < 2; p1++) {
              for (unsigned p2 = 0; p2 < 2; p2++) {
                for (unsigned p3 = 0; p3 < 2; p3++) {
                  for (unsigned dh = 1; dh <= 2; dh++) {
                    for (unsigned dw = 1; dw <= 2; dw++) {
                      for (unsigned sh = 1; sh <= 2; sh++) {
                        for (unsigned sw = 1; sw <= 2; sw++) {
                          Test(1, cs[gc], 1, is[ih], is[iw], 1, 3, 3, p0, p1, p2, p3, dh, dw, sh, sw);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    for (unsigned ic = 0; ic < _countof(cs); ic++) {
      for (unsigned ih = 0; ih < _countof(is); ih++) {
        for (unsigned iw = 0; iw < _countof(is); iw++) {
          fprintf(stderr, "Handling %ux%ux%u\n", cs[ic], is[ih], is[iw]);
          for (unsigned fc = 0; fc < _countof(cs); fc++) {
            for (unsigned kh = 1; kh <= 5; kh++) {
              if (kh == 4) continue;
              for (unsigned kw = 1; kw <= 5; kw++) {
                if (kw == 4) continue;
                for (unsigned p0 = 0; p0 < 2; p0++) {
                  for (unsigned p1 = 0; p1 < 2; p1++) {
                    for (unsigned p2 = 0; p2 < 2; p2++) {
                      for (unsigned p3 = 0; p3 < 2; p3++) {
                        for (unsigned dh = 1; dh <= 2; dh++) {
                          for (unsigned dw = 1; dw <= 2; dw++) {
                            for (unsigned sh = 1; sh <= 2; sh++) {
                              for (unsigned sw = 1; sw <= 2; sw++) {
                                Test(1, 1, cs[ic], is[ih], is[iw], cs[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    //
    // Regression test: exercise a KleidiAI Conv2D path when KleidiAI is enabled.
    // See https://github.com/microsoft/onnxruntime/issues/26669.
    //
    // The KleidiAI implementation uses an internal per-thread padding buffer for out-of-bounds pixels
    // when constructing the LHS indirection table. Historically, if the buffer was too small for a later
    // convolution (larger CI), resizing could invalidate cached indirection pointers and lead to
    // non-deterministic corruption.
    //
    // This sequence forces pad-buffer growth by running a smaller-CI convolution followed by a larger-CI
    // convolution (with padding to ensure pad pointers are used), then runs the smaller-CI convolution again.
    // Repeat a few times to increase the likelihood of triggering a reallocation and verify the path.
    //
    for (int i = 0; i < 4; ++i) {
      Test(1, 1, 64, 11, 11, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);   // smaller CI
      Test(1, 1, 320, 11, 11, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);  // larger CI forces pad buffer growth
      Test(1, 1, 64, 11, 11, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);   // sanity: back to smaller CI after growth
    }
  }

  void ExecuteShort(void) override {
#if defined(MLAS_TARGET_AMD64)
    TestMobileClipAvx512DispatchSelection(64, 64, 64);
    TestMobileClipAvx512DispatchSelection(128, 32, 32);
    TestMobileClipAvx512DispatchSelection(256, 16, 16);
#endif
    TestMobileClipBetaActivationRegression(64, 64, 64);
    TestMobileClipBetaActivationRegression(128, 32, 32);
    TestMobileClipBetaActivationRegression(256, 16, 16);
  }
};
