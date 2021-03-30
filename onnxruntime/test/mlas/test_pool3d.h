// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_util.h"

template <MLAS_POOLING_KIND PoolingKind, bool Threaded>
class MlasPool3DTest : public MlasTestBase {
 public:
  MlasPool3DTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  static const char* GetTestSuiteName() {
    static std::string suite_name =
        std::string(PoolingKind == MlasMaximumPooling
                        ? "Pool3dMax"
                        : (PoolingKind == MlasAveragePoolingExcludePad
                               ? "Pool3dAverageExcludePad"
                               : "Pool3dAverageIncludePad")) +
        (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

  void Test(size_t BatchCount,
            size_t InputChannels,
            size_t InputDepth,
            size_t InputHeight,
            size_t InputWidth,
            size_t KernelDepth,
            size_t KernelHeight,
            size_t KernelWidth,
            size_t PaddingLeftDepth,
            size_t PaddingLeftHeight,
            size_t PaddingLeftWidth,
            size_t PaddingRightDepth,
            size_t PaddingRightHeight,
            size_t PaddingRightWidth,
            size_t StrideDepth,
            size_t StrideHeight,
            size_t StrideWidth) {
    const size_t DilationDepth = 1;
    const size_t DilationHeight = 1;
    const size_t DilationWidth = 1;

    int64_t OutputDepth64 =
        ((int64_t(InputDepth) + int64_t(PaddingLeftDepth) + int64_t(PaddingRightDepth)) -
         (int64_t(DilationDepth) * (int64_t(KernelDepth) - 1) + 1)) /
            int64_t(StrideDepth) +
        1;
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

    if (OutputDepth64 <= 0 || OutputHeight64 <= 0 || OutputWidth64 <= 0) {
      return;
    }

    int64_t InputShape[] = {int64_t(BatchCount), int64_t(InputChannels), int64_t(InputDepth), int64_t(InputHeight), int64_t(InputWidth)};
    int64_t KernelShape[] = {int64_t(KernelDepth), int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftDepth), int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightDepth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};
    int64_t StrideShape[] = {int64_t(StrideDepth), int64_t(StrideHeight), int64_t(StrideWidth)};
    int64_t OutputShape[] = {int64_t(BatchCount), int64_t(InputChannels), OutputDepth64, OutputHeight64, OutputWidth64};

    OutputShape[2] = (InputShape[2] + Padding[0] + Padding[3] - KernelShape[0]) / StrideShape[0] + 1;
    OutputShape[3] = (InputShape[3] + Padding[1] + Padding[4] - KernelShape[1]) / StrideShape[1] + 1;
    OutputShape[4] = (InputShape[4] + Padding[2] + Padding[5] - KernelShape[2]) / StrideShape[2] + 1;

    size_t InputBufferElements = size_t(InputShape[0] * InputShape[1] * InputShape[2] * InputShape[3] * InputShape[4]);
    size_t OutputBufferElements = size_t(OutputShape[0] * OutputShape[1] * OutputShape[2] * OutputShape[3] * OutputShape[4]);

    const float* Input = BufferInput.GetBuffer(InputBufferElements);
    float* Output = BufferOutput.GetBuffer(OutputBufferElements);
    float* OutputReference = BufferOutputReference.GetBuffer(OutputBufferElements);

    MlasPool(PoolingKind, 3, InputShape, KernelShape, Padding, StrideShape, OutputShape, Input, Output, threadpool_);
    if (PoolingKind == MlasMaximumPooling) {
      ReferenceMaximumPool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference);
    } else if (PoolingKind == MlasAveragePoolingExcludePad) {
      ReferenceAveragePool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, false);
    } else if (PoolingKind == MlasAveragePoolingIncludePad) {
      ReferenceAveragePool3D(InputShape, KernelShape, Padding, StrideShape, Input, OutputReference, true);
    }

    ASSERT_EQ(memcmp(Output, OutputReference, OutputBufferElements * sizeof(float)), 0)
        << "PoolingKind:" << int(PoolingKind) << " "
        << "input(" << InputChannels << "," << InputDepth << "," << InputHeight << ", " << InputWidth << "), "
        << "Kernel(" << KernelDepth << "," << KernelHeight << "," << KernelWidth << ")";
  }

 protected:
  void ReferenceMaximumPool3D(const int64_t* InputShape,
                              const int64_t* KernelShape,
                              const int64_t* Padding,
                              const int64_t* StrideShape,
                              const float* Input,
                              float* Output) {
    int64_t ChannelCount = InputShape[0] * InputShape[1];

    int64_t InputDepth = InputShape[2];
    int64_t InputHeight = InputShape[3];
    int64_t InputWidth = InputShape[4];

    int64_t KernelDepth = KernelShape[0];
    int64_t KernelHeight = KernelShape[1];
    int64_t KernelWidth = KernelShape[2];

    int64_t PaddingLeftZ = Padding[0];
    int64_t PaddingLeftY = Padding[1];
    int64_t PaddingLeftX = Padding[2];
    int64_t PaddingRightZ = Padding[3];
    int64_t PaddingRightY = Padding[4];
    int64_t PaddingRightX = Padding[5];

    int64_t StrideDepth = StrideShape[0];
    int64_t StrideHeight = StrideShape[1];
    int64_t StrideWidth = StrideShape[2];

    int64_t OutputDepth = (InputDepth + PaddingLeftZ + PaddingRightZ - KernelDepth) / StrideDepth + 1;
    int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
    int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

    for (int64_t c = 0; c < ChannelCount; c++) {
      for (int64_t pd = 0; pd < OutputDepth; pd++) {
        int64_t idStart = pd * StrideDepth - PaddingLeftZ;
        int64_t idEnd = idStart + KernelDepth;

        idStart = (std::max)(idStart, int64_t(0));
        idEnd = (std::min)(idEnd, InputDepth);

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

            for (int64_t id = idStart; id < idEnd; id++) {
              for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                  m = (std::max)(m, Input[id * InputHeight * InputWidth + ih * InputWidth + iw]);
                }
              }
            }

            Output[pd * OutputHeight * OutputWidth + ph * OutputWidth + pw] = m;
          }
        }
      }

      Input += InputDepth * InputHeight * InputWidth;
      Output += OutputDepth * OutputHeight * OutputWidth;
    }
  }

  void ReferenceAveragePool3D(const int64_t* InputShape,
                              const int64_t* KernelShape,
                              const int64_t* Padding,
                              const int64_t* StrideShape,
                              const float* Input,
                              float* Output,
                              bool CountIncludePad) {
    int64_t ChannelCount = InputShape[0] * InputShape[1];

    int64_t InputDepth = InputShape[2];
    int64_t InputHeight = InputShape[3];
    int64_t InputWidth = InputShape[4];

    int64_t KernelDepth = KernelShape[0];
    int64_t KernelHeight = KernelShape[1];
    int64_t KernelWidth = KernelShape[2];

    int64_t PaddingLeftZ = Padding[0];
    int64_t PaddingLeftY = Padding[1];
    int64_t PaddingLeftX = Padding[2];
    int64_t PaddingRightZ = Padding[3];
    int64_t PaddingRightY = Padding[4];
    int64_t PaddingRightX = Padding[5];

    int64_t StrideDepth = StrideShape[0];
    int64_t StrideHeight = StrideShape[1];
    int64_t StrideWidth = StrideShape[2];

    int64_t OutputDepth = (InputDepth + PaddingLeftZ + PaddingRightZ - KernelDepth) / StrideDepth + 1;
    int64_t OutputHeight = (InputHeight + PaddingLeftY + PaddingRightY - KernelHeight) / StrideHeight + 1;
    int64_t OutputWidth = (InputWidth + PaddingLeftX + PaddingRightX - KernelWidth) / StrideWidth + 1;

    for (int64_t c = 0; c < ChannelCount; c++) {
      for (int64_t pd = 0; pd < OutputDepth; pd++) {
        int64_t idStart = pd * StrideDepth - PaddingLeftZ;
        int64_t idEnd = idStart + KernelDepth;

        idStart = (std::max)(idStart, int64_t(0));
        idEnd = (std::min)(idEnd, InputDepth);

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

            for (int64_t id = idStart; id < idEnd; id++) {
              for (int64_t ih = ihStart; ih < ihEnd; ih++) {
                for (int64_t iw = iwStart; iw < iwEnd; iw++) {
                  m += Input[id * InputHeight * InputWidth + ih * InputWidth + iw];
                }
              }
            }

            if (CountIncludePad) {
              m /= (KernelDepth * KernelHeight * KernelWidth);
            } else {
              m /= (idEnd - idStart) * (ihEnd - ihStart) * (iwEnd - iwStart);
            }

            Output[pd * OutputHeight * OutputWidth + ph * OutputWidth + pw] = m;
          }
        }
      }

      Input += InputDepth * InputHeight * InputWidth;
      Output += OutputDepth * OutputHeight * OutputWidth;
    }
  }

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MLAS_THREADPOOL* threadpool_;

 public:
  // void
  // ExecuteShort(
  //     void
  //     ) override
  // {
  //     for (unsigned i = 1; i < 64; i <<= 1) {
  //         Test(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1);
  //         Test(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2, 2);
  //         Test(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1);
  //         Test(1, 16, i, i, i, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  //         Test(1, 16, i, i, i, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1);
  //         Test(1, 16, i, i, i, 1, i, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1);
  //         Test(1, 16, i, i, i, 1, 1, i, 0, 0, 0, 0, 0, 0, 1, 1, 1);
  //     }
  // }

  void ExecuteLong(void) override {
    static const unsigned is[] = {11, 5, 4, 3, 2, 1};

    for (unsigned id = 0; id < _countof(is); id++) {
      for (unsigned ih = 0; ih < _countof(is); ih++) {
        for (unsigned iw = 0; iw < _countof(is); iw++) {
          fprintf(stderr, "Handling %ux%ux%u\n", is[id], is[ih], is[iw]);
          Test(1, 1, is[id], is[ih], is[iw], is[id], is[ih], is[iw], 0, 0, 0, 0, 0, 0, 1, 1, 1);
          for (unsigned kd = 1; kd <= 4; kd++) {
            if (kd > is[id]) break;
            for (unsigned kh = 1; kh <= 4; kh++) {
              if (kh > is[ih]) break;
              for (unsigned kw = 1; kw <= 4; kw++) {
                if (kw > is[iw]) break;
                for (unsigned sd = 1; sd <= 3; sd++) {
                  for (unsigned sh = 1; sh <= 3; sh++) {
                    for (unsigned sw = 1; sw <= 3; sw++) {
                      for (unsigned p0 = 0; p0 < kd; p0++) {
                        for (unsigned p1 = 0; p1 < kh; p1++) {
                          for (unsigned p2 = 0; p2 < kw; p2++) {
                            for (unsigned p3 = 0; p3 < kd; p3++) {
                              for (unsigned p4 = 0; p4 < kh; p4++) {
                                for (unsigned p5 = 0; p5 < kw; p5++) {
                                  Test(1, 1, is[id], is[ih], is[iw], kd, kh, kw, p0, p1, p2, p3, p4, p5, sd, sh, sw);
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
    }
  }
};
