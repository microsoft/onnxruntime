
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_pool2d.h"

//
// Short Execute() test helper to register each test seperately by all parameters.
//
template <typename Pool2DTester>
class Pooling2dShortExecuteTest : public MlasTestFixture<Pool2DTester> {
 public:
  explicit Pooling2dShortExecuteTest(size_t BatchCount,
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
                                     size_t StrideWidth)
      : BatchCount_(BatchCount),
        InputChannels_(InputChannels),
        InputHeight_(InputHeight),
        InputWidth_(InputWidth),
        KernelHeight_(KernelHeight),
        KernelWidth_(KernelWidth),
        PaddingLeftHeight_(PaddingLeftHeight),
        PaddingLeftWidth_(PaddingLeftWidth),
        PaddingRightHeight_(PaddingRightHeight),
        PaddingRightWidth_(PaddingRightWidth),
        StrideHeight_(StrideHeight),
        StrideWidth_(StrideWidth) {
  }

  void TestBody() override {
    MlasTestFixture<Pool2DTester>::mlas_tester->Test(
        BatchCount_,
        InputChannels_,
        InputHeight_,
        InputWidth_,
        KernelHeight_,
        KernelWidth_,
        PaddingLeftHeight_,
        PaddingLeftWidth_,
        PaddingRightHeight_,
        PaddingRightWidth_,
        StrideHeight_,
        StrideWidth_);
  }

  static size_t RegisterSingleTest(size_t BatchCount,
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
    std::stringstream ss;
    ss << "B" << BatchCount << "/"
       << "C" << InputChannels << "/"
       << "H" << InputHeight << "/"
       << "W" << InputWidth << "/"
       << "KH" << KernelHeight << "/"
       << "KW" << KernelWidth << "/"
       << "Pad" << PaddingLeftHeight << "," << PaddingLeftWidth << "," << PaddingRightHeight << "," << PaddingRightWidth << "/"
       << "Stride" << StrideHeight << "," << StrideWidth;
    auto test_name = ss.str();

    testing::RegisterTest(
        Pool2DTester::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<Pool2DTester>* {
          return new Pooling2dShortExecuteTest<Pool2DTester>(
              BatchCount,
              InputChannels,
              InputHeight,
              InputWidth,
              KernelHeight,
              KernelWidth,
              PaddingLeftHeight,
              PaddingLeftWidth,
              PaddingRightHeight,
              PaddingRightWidth,
              StrideHeight,
              StrideWidth);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;

    for (unsigned i = 1; i < 256; i <<= 1) {
      test_registered += RegisterSingleTest(1, 16, i, i, 3, 3, 0, 0, 0, 0, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, 3, 3, 0, 0, 0, 0, 2, 2);
      test_registered += RegisterSingleTest(1, 16, i, i, 3, 3, 0, 0, 0, 0, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, 3, 3, 1, 1, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, 1, 1, 0, 0, 0, 0, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, i, 1, 0, 0, 0, 0, 1, 1);
      test_registered += RegisterSingleTest(1, 16, i, i, 1, i, 0, 0, 0, 0, 1, 1);
    }

    return test_registered;
  }

 private:
  size_t BatchCount_;
  size_t InputChannels_;
  size_t InputHeight_;
  size_t InputWidth_;
  size_t KernelHeight_;
  size_t KernelWidth_;
  size_t PaddingLeftHeight_;
  size_t PaddingLeftWidth_;
  size_t PaddingRightHeight_;
  size_t PaddingRightWidth_;
  size_t StrideHeight_;
  size_t StrideWidth_;
};
