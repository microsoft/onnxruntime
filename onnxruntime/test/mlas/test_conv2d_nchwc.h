// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_conv2d.h"

template <bool Threaded>
class MlasNchwcConv2DTest : public MlasConv2DTest<Threaded> {
 protected:
  void MlasConv2D(
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
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output) override {
    int64_t InputShape[] = {int64_t(BatchCount), int64_t(GroupCount * InputChannels), int64_t(InputHeight), int64_t(InputWidth)};
    int64_t FilterShape[] = {int64_t(GroupCount * FilterCount), int64_t(InputChannels), int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t OutputShape[] = {int64_t(BatchCount), int64_t(GroupCount * FilterCount), int64_t(OutputHeight), int64_t(OutputWidth)};

    int64_t KernelShape[] = {int64_t(KernelHeight), int64_t(KernelWidth)};
    int64_t DilationShape[] = {int64_t(DilationHeight), int64_t(DilationWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};
    int64_t StrideShape[] = {int64_t(StrideHeight), int64_t(StrideWidth)};

    //
    // Select the type of convolution that will be performed.
    //

    bool DoReorderInput;
    bool ReorderFilterOIHWBo;

    if (GroupCount > 1 && InputChannels == 1 && FilterCount == 1) {
      // Depthwise convolution.
      DoReorderInput = true;
      ReorderFilterOIHWBo = true;
    } else if (InputChannels >= BlockSize) {
      // NCHWc or pointwise convolution;
      DoReorderInput = true;
      ReorderFilterOIHWBo = false;
    } else {
      // NCHW convolution.
      DoReorderInput = false;
      ReorderFilterOIHWBo = true;
    }

    size_t NchwcInputChannels = (GroupCount * InputChannels + BlockSize - 1) & ~(BlockSize - 1);
    size_t NchwcOutputChannels = (GroupCount * FilterCount + BlockSize - 1) & ~(BlockSize - 1);

    //
    // Reorder the filter buffer as needed.
    //

    float* ReorderedFilter;

    if (ReorderFilterOIHWBo) {
      size_t NchwcFilterElements = NchwcOutputChannels * InputChannels * KernelHeight * KernelWidth;
      ReorderedFilter = BufferNchwcFilter.GetBuffer(NchwcFilterElements);
      MlasReorderFilterOIHWBo(FilterShape, Filter, ReorderedFilter);
    } else {
      size_t NchwcFilterElements = NchwcOutputChannels * NchwcInputChannels * KernelHeight * KernelWidth;
      ReorderedFilter = BufferNchwcFilter.GetBuffer(NchwcFilterElements);
      MlasReorderFilterOIHWBiBo(FilterShape, Filter, ReorderedFilter);
    }

    //
    // Align the bias buffer to the filter count if needed.
    //

    if (Bias != nullptr && GroupCount * FilterCount < NchwcOutputChannels) {
      float* AlignedBias = BufferNchwcBias.GetBuffer(NchwcOutputChannels);

      size_t i;
      for (i = 0; i < GroupCount * FilterCount; i++) {
        AlignedBias[i] = Bias[i];
      }
      for (; i < NchwcOutputChannels; i++) {
        AlignedBias[i] = 0.0f;
      }

      Bias = AlignedBias;
    }

    //
    // Reorder the input buffer if needed.
    //

    if (DoReorderInput) {
      size_t NchwcInputElements = BatchCount * NchwcInputChannels * InputHeight * InputWidth;
      float* NchwcInput = BufferNchwcInput.GetBuffer(NchwcInputElements);
      MlasReorderInput(InputShape, Input, NchwcInput);
      Input = NchwcInput;
      InputShape[1] = NchwcInputChannels;
    }

    int64_t NchwcOutputShape[] = {int64_t(BatchCount), int64_t(NchwcOutputChannels), int64_t(OutputHeight), int64_t(OutputWidth)};

    size_t NchwcOutputElements = BatchCount * NchwcOutputChannels * OutputHeight * OutputWidth;
    float* NchwcOutput = BufferNchwcOutput.GetBuffer(NchwcOutputElements);

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    MlasNchwcConv(InputShape,
                  KernelShape,
                  DilationShape,
                  Padding,
                  StrideShape,
                  NchwcOutputShape,
                  GroupCount,
                  Input,
                  ReorderedFilter,
                  Bias,
                  NchwcOutput,
                  &Activation,
                  true,
                  MlasConv2DTest<Threaded>::threadpool_);

    //
    // Reorder the output buffer.
    //

    MlasReorderOutputNchw(OutputShape, NchwcOutput, Output);
  }

  const size_t BlockSize = MlasNchwcGetBlockSize();

  MatrixGuardBuffer<float> BufferNchwcInput;
  MatrixGuardBuffer<float> BufferNchwcFilter;
  MatrixGuardBuffer<float> BufferNchwcBias;
  MatrixGuardBuffer<float> BufferNchwcOutput;

 public:
  static const char* GetTestSuiteName(void) {
    static const std::string suite_name(Threaded? "Conv2dNchwc_Threaded" : "Conv2dNchwc_SingleThread");
    return suite_name.c_str();
  }

  MlasNchwcConv2DTest() : MlasConv2DTest<Threaded>() {}

  void ExecuteLong(void) override {
    // N.B. InputChannels must be a multiple of 4 if the count is greater
    // than the block size.
    static const unsigned cis[] = {32, 20, 5, 1};
    static const unsigned cos[] = {64, 15, 1};
    static const unsigned is[] = {27, 11, 5, 1};

    // Depthwise convolutions.
    for (unsigned i = 16; i < 256; i <<= 1) {
      MlasConv2DTest<Threaded>::Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
      MlasConv2DTest<Threaded>::Test(1, i, 1, 28, 28, 1, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
      MlasConv2DTest<Threaded>::Test(1, i, 1, 28, 28, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(1, i, 1, 28, 28, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(1, i, 1, 28, 28, 1, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(12, i, 1, 11, 11, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    // Test varying FilterCounts.
    for (unsigned i = 1; i < 128; i++) {
      MlasConv2DTest<Threaded>::Test(1, 1, 3, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(1, 1, 16, 34, 34, i, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(1, 1, 16, 34, 34, i, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    for (unsigned i = 1; i <= 32; i++) {
      MlasConv2DTest<Threaded>::Test(4, 18, 1, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(4, 18, 1, 32, 89, 48, i, 89, 1, 1, 1, 1, 1, 1, 1, 1);
      MlasConv2DTest<Threaded>::Test(4, 18, 2, 32, 89, 48, i, 89, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    for (unsigned b = 1; b < 64; b++) {
      MlasConv2DTest<Threaded>::Test(b, 1, 64, 11, 11, 128, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    }

    for (unsigned ic = 0; ic < _countof(cis); ic++) {
      for (unsigned ih = 0; ih < _countof(is); ih++) {
        for (unsigned iw = 0; iw < _countof(is); iw++) {
          fprintf(stderr, "Handling %ux%ux%u\n", cis[ic], is[ih], is[iw]);
          for (unsigned fc = 0; fc < _countof(cos); fc++) {
            for (unsigned kh = 1; kh <= 5; kh++) {
              if (kh == 4) continue;
              for (unsigned kw = 1; kw <= 5; kw++) {
                if (kw == 4) continue;
                for (unsigned p0 = 0; p0 <= 3; p0++) {
                  for (unsigned p1 = 0; p1 <= 3; p1++) {
                    for (unsigned p2 = 0; p2 <= 3; p2++) {
                      for (unsigned p3 = 0; p3 <= 3; p3++) {
                        for (unsigned dh = 1; dh <= 2; dh++) {
                          for (unsigned dw = 1; dw <= 2; dw++) {
                            for (unsigned sh = 1; sh <= 2; sh++) {
                              for (unsigned sw = 1; sw <= 2; sw++) {
                                MlasConv2DTest<Threaded>::Test(1, 1, cis[ic], is[ih], is[iw], cos[fc], kh, kw, p0, p1, p2, p3, dh, dw, sh, sw);
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
