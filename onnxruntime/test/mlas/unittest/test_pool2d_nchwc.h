// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_pool2d.h"

template <MLAS_POOLING_KIND PoolingKind, bool Threaded>
class MlasNchwcPool2DTest : public MlasPool2DTest<PoolingKind, Threaded> {
 public:
  static const char* GetTestSuiteName() {
    static std::string suite_name =
        std::string(PoolingKind == MlasMaximumPooling
                        ? "Pool2dNchwcMax"
                        : (PoolingKind == MlasAveragePoolingExcludePad ? "Pool2dNchwcAverageExcludePad" : "Pool2dNchwcAverageIncludePad")) +
        (Threaded ? "_Threaded" : "_SingleThread");
    return suite_name.c_str();
  }

 protected:
  void MlasPool2D(
      const int64_t* InputShape,
      const int64_t* KernelShape,
      const int64_t* Padding,
      const int64_t* StrideShape,
      const int64_t* OutputShape,
      const float* Input,
      float* Output) override {
    size_t NchwcChannels = (size_t(InputShape[1]) + BlockSize - 1) & ~(BlockSize - 1);

    int64_t NchwcInputShape[] = {InputShape[0], int64_t(NchwcChannels), InputShape[2], InputShape[3]};
    size_t NchwcInputElements = size_t(NchwcInputShape[0]) * size_t(NchwcInputShape[1]) * size_t(NchwcInputShape[2]) * size_t(NchwcInputShape[3]);
    float* NchwcInput = BufferNchwcInput.GetBuffer(NchwcInputElements);

    int64_t NchwcOutputShape[] = {OutputShape[0], int64_t(NchwcChannels), OutputShape[2], OutputShape[3]};
    size_t NchwcOutputElements = size_t(NchwcOutputShape[0]) * size_t(NchwcOutputShape[1]) * size_t(NchwcOutputShape[2]) * size_t(NchwcOutputShape[3]);
    float* NchwcOutput = BufferNchwcOutput.GetBuffer(NchwcOutputElements);

    MlasReorderInput(InputShape, Input, NchwcInput);

    MlasNchwcPool(PoolingKind,
                  NchwcInputShape,
                  KernelShape,
                  nullptr,
                  Padding,
                  StrideShape,
                  NchwcOutputShape,
                  NchwcInput,
                  NchwcOutput,
                  nullptr);

    MlasReorderOutputNchw(OutputShape, NchwcOutput, Output);
  }

  MatrixGuardBuffer<float> BufferNchwcInput;
  MatrixGuardBuffer<float> BufferNchwcOutput;

  const size_t BlockSize = MlasNchwcGetBlockSize();

 public:
  void ExecuteLong(void) override {
    static const unsigned is[] = {53, 11, 1};

    for (unsigned ih = 0; ih < _countof(is); ih++) {
      for (unsigned iw = 0; iw < _countof(is); iw++) {
        fprintf(stderr, "Handling %ux%u\n", is[ih], is[iw]);
        MlasPool2DTest<PoolingKind, Threaded>::Test(1, 12, is[ih], is[iw], is[ih], is[iw], 0, 0, 0, 0, 1, 1);
        MlasPool2DTest<PoolingKind, Threaded>::Test(1, 32, is[ih], is[iw], is[ih], 1, 0, 0, 0, 0, 1, 1);
        MlasPool2DTest<PoolingKind, Threaded>::Test(1, 68, is[ih], is[iw], 1, is[iw], 0, 0, 0, 0, 1, 1);
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
                        MlasPool2DTest<PoolingKind, Threaded>::Test(1, 32, is[ih], is[iw], kh, kw, p0, p1, p2, p3, sh, sw);
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
