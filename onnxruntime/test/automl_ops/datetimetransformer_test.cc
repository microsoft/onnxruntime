// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/DateTimeFeaturizer.h"

namespace dft = Microsoft::Featurizer::DateTimeFeaturizer;

using SysClock = std::chrono::system_clock;

namespace onnxruntime {
namespace test {

TEST(DateTimeFeaturizer_DateTime, Past_1976_Nov_17__12_27_04) {

  const time_t date = 217081624;
  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSAutoMLDomain);

  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("X", {1}, {date});

  SysClock::time_point stp = SysClock::from_time_t(date);
  dft::TimePoint tp(stp);
  ASSERT_TRUE(tp.year == 1976);
  ASSERT_TRUE(tp.month == dft::TimePoint::NOVEMBER);
  ASSERT_TRUE(tp.day == 17);
  ASSERT_TRUE(tp.hour == 12);
  ASSERT_TRUE(tp.minute == 27);
  ASSERT_TRUE(tp.second == 4);
  ASSERT_TRUE(tp.dayOfWeek == dft::TimePoint::WEDNESDAY);
  ASSERT_TRUE(tp.dayOfYear == 321);
  ASSERT_TRUE(tp.quarterOfYear == 4);
  ASSERT_TRUE(tp.weekOfMonth == 2);

  // Expected output.
  test.AddOutput<dft::TimePoint>("Y", std::move(tp));
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(DateTimeFeaturizer_Transformer, Past_1976_Nov_17__12_27_05) {
  const time_t date = 32445842582;

  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSAutoMLDomain);
  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("X", {1}, {date});

  SysClock::time_point stp = SysClock::from_time_t(date);

  dft::Transformer dt;
  dft::TimePoint tp = dt.transform(stp);
  ASSERT_TRUE(tp.year == 2998);
  ASSERT_TRUE(tp.month == dft::TimePoint::MARCH);
  ASSERT_TRUE(tp.day == 2);
  ASSERT_TRUE(tp.hour == 14);
  ASSERT_TRUE(tp.minute == 3);
  ASSERT_TRUE(tp.second == 2);
  ASSERT_TRUE(tp.dayOfWeek == dft::TimePoint::FRIDAY);
  ASSERT_TRUE(tp.dayOfYear == 60);
  ASSERT_TRUE(tp.quarterOfYear == 1);
  ASSERT_TRUE(tp.weekOfMonth == 0);

  // Expected output.
  test.AddOutput<dft::TimePoint>("Y", std::move(tp));
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime
