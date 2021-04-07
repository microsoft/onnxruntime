// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_util.h"

template <MLAS_POOLING_KIND PoolingKind, bool Threaded>
class MlasPool2DTest : public MlasTestBase {
 public:
  MlasPool2DTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  static const char* GetTestSuiteName() {
    static std::string suite_name =
        std::string(PoolingKind == MlasMaximumPooling
                        ? "Pool2dMax"
                        : (PoolingKind == MlasAveragePoolingExcludePad ? "Pool2dAverageExcludePad" : "Pool2dAverageIncludePad")) +
        (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

  void Test(size_t BatchCount,
            size_t InputChannels,
            size_t InputHeight,
            size_t InputWidth,
            size_t KernelHeight,
            size_t KernelWidth,
            size_t PaddingLeftHeight,
            size_t PaddingLeftWidth,
            size_t PaddingRightHeight,
            size_t PaddingRightWidth,
            size_t StrideHeight,
            size_t StrideWidth) {
    const size_t DilationHeight = 1;
    const size_t DilationWidth = 1;

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

    int64_t InputShape[] = {int64_t(BatchCount), int64_t(InputChannels), int64_t(InputHeight), int64_t(InputWidth)};
    int64_t KernelShape[] = {int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};
    int64_t StrideShape[] = {int64_t(StrideHeight), int64_t(StrideWidth)};
    int64_t OutputShape[] = {int64_t(BatchCount), int64_t(InputChannels), OutputHeight64, OutputWidth64};

    size_t InputBufferElements = size_t(InputShape[0] * InputShape[1] * InputShape[2] * InputShape[3]);
    size_t OutputBufferElements = size_t(OutputShape[0] * OutputShape[1] * OutputShape[2] * OutputShape[3]);

    const float* Input = BufferInput.GetBuffer(InputBufferElements);
    float* Output = BufferOutput.GetBuffer(OutputBufferElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

    MlasPool2D(InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output);
    if (PoolingKind == MlasMaximumPooling) {
      ReferenceMaximumPool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference);
    } else if (PoolingKind == MlasAveragePoolingExcludePad) {
      ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, false);
    } else if (PoolingKind == MlasAveragePoolingIncludePad) {
      ReferenceAveragePool2D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, true);
    }

    ASSERT_EQ(memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)), 0)
        << "PoolingKind:" << int(PoolingKind) << " "
        << "input(" << InputChannels << "," << InputHeight << ", " << InputWidth << "), "
        << "Kernel(" << KernelHeight << "," << KernelWidth << ")";
  }

 protected:
  virtual void MlasPool2D(const int64_t* InputShape,
                          const int64_t* KernelShape,
                          const int64_t* Padding,
                          const int64_t* StrideShape,
                          const int64_t* OutputShape,
                          const float* Input,
                          float* Output) {
    MlasPool(PoolingKind, 2, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, threadpool_);
  }

  void ReferenceMaximumPool2D(const int64_t* InputShape,
                              const int64_t* KernelShape,
                              const int64_t* Padding,
                              const int64_t* StrideShape,
                              const float* Input,
                              float* Output) {
    int64_t ChannelCount = InputShape[0] * InputShape[1];

    int64_t InputHeight = InputShape[2];
    int64_t InputWidth = InputShape[3];

    int64_t KernelHeight = KernelShape[0];
    int64_t KernelWidth = KernelShape[1];

    int64_t PaddingLeftY = Padding[0];
    int64_t PaddingLeftX = Padding[1];
    int64_t PaddingRightY = Padding[2];
    int64_t PaddingRightX = Padding[3];

    int64_t StrideHeight = StrideShape[0];
    int64_t StrideWidth = StrideShape[1];

    int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
    int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

    for (int64_t c = 0; c < ChannelCount; c++) {
      for (int64_t ph = 0; ph < OutputHeight; ph++) {
        int64_t ihStart = ph * StrideHeight - PaddingLeftY;
        int64_t ihEnd = ihStart + KernelHeight;

        ihStart = (std::max)(ihStart, int64_t(0));
        ihEnd = (std::min)(ihEnd, InputHeight);

        for (int64_t pw = 0; pw < OutputWidth; pw++) {
          int64_t iwStart = pw * StrideWidth - PaddingLeftX;
          int64_t iwEnd = iwStart + KernelWidth;

          iwStart = (std::max)(iwStart, int64_t(0));
          iwEnd = (std::min)(iwEnd, InputWidth);

          float m = std::numeric_limits<float>::lowest();

          for (int64_t ih = ihStart; ih < ihEnd; ih++) {
            for (int64_t iw = iwStart; iw < iwEnd; iw++) {
              m = (std::max)(m, Input[ih * InputWidth + iw]);
            }
          }

          Output[ph * OutputWidth + pw] = m;
        }
      }

      Input += InputHeight * InputWidth;
      Output += OutputHeight * OutputWidth;
    }
  }

  void ReferenceAveragePool2D(const int64_t* InputShape,
                              const int64_t* KernelShape,
                              const int64_t* Padding,
                              const int64_t* StrideShape,
                              const float* Input,
                              float* Output,
                              bool CountIncludePad) {
    int64_t ChannelCount = InputShape[0] * InputShape[1];

    int64_t InputHeight = InputShape[2];
    int64_t InputWidth = InputShape[3];

    int64_t KernelHeight = KernelShape[0];
    int64_t KernelWidth = KernelShape[1];

    int64_t PaddingLeftY = Padding[0];
    int64_t PaddingLeftX = Padding[1];
    int64_t PaddingRightY = Padding[2];
    int64_t PaddingRightX = Padding[3];

    int64_t StrideHeight = StrideShape[0];
    int64_t StrideWidth = StrideShape[1];

    int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
    int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

    for (int64_t c = 0; c < ChannelCount; c++) {
      for (int64_t ph = 0; ph < OutputHeight; ph++) {
        int64_t ihStart = ph * StrideHeight - PaddingLeftY;
        int64_t ihEnd = ihStart + KernelHeight;

        ihStart = (std::max)(ihStart, int64_t(0));
        ihEnd = (std::min)(ihEnd, InputHeight);

        for (int64_t pw = 0; pw < OutputWidth; pw++) {
          int64_t iwStart = pw * StrideWidth - PaddingLeftX;
          int64_t iwEnd = iwStart + KernelWidth;

          iwStart = (std::max)(iwStart, int64_t(0));
          iwEnd = (std::min)(iwEnd, InputWidth);

          float m = 0.0f;

          for (int64_t ih = ihStart; ih < ihEnd; ih++) {
            for (int64_t iw = iwStart; iw < iwEnd; iw++) {
              m += Input[ih * InputWidth + iw];
            }
          }

          if (CountIncludePad) {
            m /= (KernelHeight * KernelWidth);
          } else {
            m /= (ihEnd - ihStart) * (iwEnd - iwStart);
          }

          Output[ph * OutputWidth + pw] = m;
        }
      }

      Input += InputHeight * InputWidth;
      Output += OutputHeight * OutputWidth;
    }
  }

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MLAS_THREADPOOL* threadpool_;

 public:
  void ExecuteLong(void) override {
    static const unsigned is[] = {53, 17, 11, 5, 4, 3, 2, 1};

    for (unsigned i = 1; i < 2058; i++) {
      Test(1, 1, 4, i, 2, 4, 0, 2, 0, 1, 1, 1);
    }

    for (unsigned ih = 0; ih < _countof(is); ih++) {
      for (unsigned iw = 0; iw < _countof(is); iw++) {
        fprintf(stderr, "Handling %ux%u\n", is[ih], is[iw]);
        Test(1, 1, is[ih], is[iw], is[ih], is[iw], 0, 0, 0, 0, 1, 1);
        Test(1, 1, is[ih], is[iw], is[ih], 1, 0, 0, 0, 0, 1, 1);
        Test(1, 1, is[ih], is[iw], 1, is[iw], 0, 0, 0, 0, 1, 1);
        for (unsigned kh = 1; kh <= 5; kh++) {
          if (kh > is[ih]) break;
          for (unsigned kw = 1; kw <= 5; kw++) {
            if (kw > is[iw]) break;
            for (unsigned sh = 1; sh <= 3; sh++) {
              for (unsigned sw = 1; sw <= 3; sw++) {
                for (unsigned p0 = 0; p0 < kh; p0++) {
                  for (unsigned p1 = 0; p1 < kw; p1++) {
                    for (unsigned p2 = 0; p2 < kh; p2++) {
                      for (unsigned p3 = 0; p3 < kw; p3++) {
                        Test(5, 3, is[ih], is[iw], kh, kw, p0, p1, p2, p3, sh, sw);
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
};
