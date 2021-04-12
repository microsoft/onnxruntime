
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_pool3d.h"

//
// Short Execute() test helper to register each test seperately by all parameters.
//
template <MLAS_POOLING_KIND PoolingKind, bool Threaded>
class Pooling3dShortExecuteTest : public MlasTestFixture<MlasPool3DTest<PoolingKind, Threaded>> {
 public:
  explicit Pooling3dShortExecuteTest(size_t BatchCount,
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
                                     size_t StrideWidth)
      : BatchCount_(BatchCount),
        InputChannels_(InputChannels),
        InputDepth_(InputDepth),
        InputHeight_(InputHeight),
        InputWidth_(InputWidth),
        KernelDepth_(KernelDepth),
        KernelHeight_(KernelHeight),
        KernelWidth_(KernelWidth),
        PaddingLeftDepth_(PaddingLeftDepth),
        PaddingLeftHeight_(PaddingLeftHeight),
        PaddingLeftWidth_(PaddingLeftWidth),
        PaddingRightDepth_(PaddingRightDepth),
        PaddingRightHeight_(PaddingRightHeight),
        PaddingRightWidth_(PaddingRightWidth),
        StrideDepth_(StrideDepth),
        StrideHeight_(StrideHeight),
        StrideWidth_(StrideWidth) {
  }

  void TestBody() override {
    MlasTestFixture<MlasPool3DTest<PoolingKind, Threaded>>::mlas_tester->Test(
        BatchCount_,
        InputChannels_,
        InputDepth_,
        InputHeight_,
        InputWidth_,
        KernelDepth_,
        KernelHeight_,
        KernelWidth_,
        PaddingLeftDepth_,
        PaddingLeftHeight_,
        PaddingLeftWidth_,
        PaddingRightDepth_,
        PaddingRightHeight_,
        PaddingRightWidth_,
        StrideDepth_,
        StrideHeight_,
        StrideWidth_);
  }

  static size_t RegisterSingleTest(size_t BatchCount,
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
    std::stringstream ss;
    ss << "B" << BatchCount << "/"
       << "C" << InputChannels << "/"
       << "Input_" << InputDepth << "x" << InputHeight << "x" << InputWidth << "/"
       << "Kernel" << KernelDepth << "x" << KernelHeight << "x" << KernelWidth << "/"
       << "Pad" << PaddingLeftDepth << "," << PaddingLeftHeight << "," << PaddingLeftWidth
       << "," << PaddingRightDepth << "," << PaddingRightHeight << "," << PaddingRightWidth << "/"
       << "Stride" << StrideDepth << "," << StrideHeight << "," << StrideWidth;
    auto test_name = ss.str();

    testing::RegisterTest(
        MlasPool3DTest<PoolingKind, Threaded>::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<MlasPool3DTest<PoolingKind, Threaded>>* {
          return new Pooling3dShortExecuteTest<PoolingKind, Threaded>(BatchCount,
                                                                      InputChannels,
                                                                      InputDepth,
                                                                      InputHeight,
                                                                      InputWidth,
                                                                      KernelDepth,
                                                                      KernelHeight,
                                                                      KernelWidth,
                                                                      PaddingLeftDepth,
                                                                      PaddingLeftHeight,
                                                                      PaddingLeftWidth,
                                                                      PaddingRightDepth,
                                                                      PaddingRightHeight,
                                                                      PaddingRightWidth,
                                                                      StrideDepth,
                                                                      StrideHeight,
                                                                      StrideWidth);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (unsigned i = 1; i < 64; i <<= 1) {
      test_registered += RegisterSingleTest(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2, 2);
      test_registered += RegisterSingleTest(1, 16, i, i, i, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, i, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, i, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, i, 1, i, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, i, 1, 1, i, 0, 0, 0, 0, 0, 0, 1, 1, 1);
    }

    return test_registered;
  }

 private:
  size_t BatchCount_;
  size_t InputChannels_;
  size_t InputDepth_;
  size_t InputHeight_;
  size_t InputWidth_;
  size_t KernelDepth_;
  size_t KernelHeight_;
  size_t KernelWidth_;
  size_t PaddingLeftDepth_;
  size_t PaddingLeftHeight_;
  size_t PaddingLeftWidth_;
  size_t PaddingRightDepth_;
  size_t PaddingRightHeight_;
  size_t PaddingRightWidth_;
  size_t StrideDepth_;
  size_t StrideHeight_;
  size_t StrideWidth_;
};
