// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/DateTimeFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace dft = Microsoft::Featurizer::Featurizers;

using SysClock = std::chrono::system_clock;

namespace onnxruntime {

namespace featurizers {

// Defined in date_time_transformer.cc
extern std::string GetDateTimeTransformerDataDir(void);

} // namespace featurizers

namespace test {

namespace {

std::vector<uint8_t> GetStream(std::string const &optionalCountryCode=std::string()) {
  dft::DateTimeTransformer dt(optionalCountryCode, onnxruntime::featurizers::GetDateTimeTransformerDataDir());
  Microsoft::Featurizer::Archive ar;
  dt.save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, DateTimeTransformer_past_1976_nov_17_12_27_04) {
  const time_t date = 217081624;
  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  auto stream = GetStream();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("Date", {1}, {date});

  SysClock::time_point stp = SysClock::from_time_t(date);
  dft::TimePoint tp(stp);

  ASSERT_EQ(tp.year, 1976);
  ASSERT_EQ(tp.month, dft::TimePoint::NOVEMBER);
  ASSERT_EQ(tp.day, 17);
  ASSERT_EQ(tp.hour, 12);
  ASSERT_EQ(tp.minute, 27);
  ASSERT_EQ(tp.second, 4);
  ASSERT_EQ(tp.amPm, 1);
  ASSERT_EQ(tp.hour12, 12);
  ASSERT_EQ(tp.dayOfWeek, dft::TimePoint::WEDNESDAY);
  ASSERT_EQ(tp.dayOfQuarter, 48);
  ASSERT_EQ(tp.dayOfYear, 321);
  ASSERT_EQ(tp.weekOfMonth, 2);
  ASSERT_EQ(tp.quarterOfYear, 4);
  ASSERT_EQ(tp.halfOfYear, 2);
  ASSERT_EQ(tp.weekIso, 47);
  ASSERT_EQ(tp.yearIso, 1976);
  ASSERT_EQ(tp.monthLabel, "November");
  ASSERT_EQ(tp.amPmLabel, "pm");
  ASSERT_EQ(tp.dayOfWeekLabel, "Wednesday");
  ASSERT_EQ(tp.holidayName, "");
  ASSERT_EQ(tp.isPaidTimeOff, 0);

  // Expected output.
  test.AddOutput<int32_t>("year", {1}, {tp.year});
  test.AddOutput<uint8_t>("month", {1}, {tp.month});
  test.AddOutput<uint8_t>("day", {1}, {tp.day});
  test.AddOutput<uint8_t>("hour", {1}, {tp.hour});
  test.AddOutput<uint8_t>("minute", {1}, {tp.minute});
  test.AddOutput<uint8_t>("second", {1}, {tp.second});
  test.AddOutput<uint8_t>("amPm", {1}, {tp.amPm});
  test.AddOutput<uint8_t>("hour12", {1}, {tp.hour12});
  test.AddOutput<uint8_t>("dayOfWeek", {1}, {tp.dayOfWeek});
  test.AddOutput<uint8_t>("dayOfQuarter", {1}, {tp.dayOfQuarter});
  test.AddOutput<uint16_t>("dayOfYear", {1}, {tp.dayOfYear});
  test.AddOutput<uint16_t>("weekOfMonth", {1}, {tp.weekOfMonth});
  test.AddOutput<uint8_t>("quarterOfYear", {1}, {tp.quarterOfYear});
  test.AddOutput<uint8_t>("halfOfYear", {1}, {tp.halfOfYear});
  test.AddOutput<uint8_t>("weekIso", {1}, {tp.weekIso});
  test.AddOutput<int32_t>("yearIso", {1}, {tp.yearIso});
  test.AddOutput<std::string>("monthLabel", {1}, {tp.monthLabel});
  test.AddOutput<std::string>("amPmLabel", {1}, {tp.amPmLabel});
  test.AddOutput<std::string>("dayOfWeekLabel", {1}, {tp.dayOfWeekLabel});
  test.AddOutput<std::string>("holidayName", {1}, {tp.holidayName});
  test.AddOutput<uint8_t>("isPaidTimeOff", {1}, {tp.isPaidTimeOff});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, DateTimeTransformer_past_1976_nov_17_12_27_05) {
  const time_t date = 217081625;
  const auto date_tp = SysClock::from_time_t(date);

  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  auto stream = GetStream();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("Date", {1}, {date});

  dft::DateTimeTransformer dt("", "");
  dft::TimePoint tp(dt.execute(date_tp));
  ASSERT_EQ(tp.year, 1976);
  ASSERT_EQ(tp.month, dft::TimePoint::NOVEMBER);
  ASSERT_EQ(tp.day, 17);
  ASSERT_EQ(tp.hour, 12);
  ASSERT_EQ(tp.minute, 27);
  ASSERT_EQ(tp.second, 5);
  ASSERT_EQ(tp.amPm, 1);
  ASSERT_EQ(tp.hour12, 12);
  ASSERT_EQ(tp.dayOfWeek, dft::TimePoint::WEDNESDAY);
  ASSERT_EQ(tp.dayOfQuarter, 48);
  ASSERT_EQ(tp.dayOfYear, 321);
  ASSERT_EQ(tp.weekOfMonth, 2);
  ASSERT_EQ(tp.quarterOfYear, 4);
  ASSERT_EQ(tp.halfOfYear, 2);
  ASSERT_EQ(tp.weekIso, 47);
  ASSERT_EQ(tp.yearIso, 1976);
  ASSERT_EQ(tp.monthLabel, "November");
  ASSERT_EQ(tp.amPmLabel, "pm");
  ASSERT_EQ(tp.dayOfWeekLabel, "Wednesday");
  ASSERT_EQ(tp.holidayName, "");
  ASSERT_EQ(tp.isPaidTimeOff, 0);

  // Expected output.
  test.AddOutput<int32_t>("year", {1}, {tp.year});
  test.AddOutput<uint8_t>("month", {1}, {tp.month});
  test.AddOutput<uint8_t>("day", {1}, {tp.day});
  test.AddOutput<uint8_t>("hour", {1}, {tp.hour});
  test.AddOutput<uint8_t>("minute", {1}, {tp.minute});
  test.AddOutput<uint8_t>("second", {1}, {tp.second});
  test.AddOutput<uint8_t>("amPm", {1}, {tp.amPm});
  test.AddOutput<uint8_t>("hour12", {1}, {tp.hour12});
  test.AddOutput<uint8_t>("dayOfWeek", {1}, {tp.dayOfWeek});
  test.AddOutput<uint8_t>("dayOfQuarter", {1}, {tp.dayOfQuarter});
  test.AddOutput<uint16_t>("dayOfYear", {1}, {tp.dayOfYear});
  test.AddOutput<uint16_t>("weekOfMonth", {1}, {tp.weekOfMonth});
  test.AddOutput<uint8_t>("quarterOfYear", {1}, {tp.quarterOfYear});
  test.AddOutput<uint8_t>("halfOfYear", {1}, {tp.halfOfYear});
  test.AddOutput<uint8_t>("weekIso", {1}, {tp.weekIso});
  test.AddOutput<int32_t>("yearIso", {1}, {tp.yearIso});
  test.AddOutput<std::string>("monthLabel", {1}, {tp.monthLabel});
  test.AddOutput<std::string>("amPmLabel", {1}, {tp.amPmLabel});
  test.AddOutput<std::string>("dayOfWeekLabel", {1}, {tp.dayOfWeekLabel});
  test.AddOutput<std::string>("holidayName", {1}, {tp.holidayName});
  test.AddOutput<uint8_t>("isPaidTimeOff", {1}, {tp.isPaidTimeOff});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, DateTimeTransformer_past_1976_nov_17__12_27_05_and_past_1976_nov_17_12_27_04) {
  const time_t date1 = 217081625;
  const auto date1_tp = SysClock::from_time_t(date1);
  const time_t date2 = 217081624;
  const auto date2_tp = SysClock::from_time_t(date2);

  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  auto stream = GetStream();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);


  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("Date", {2}, {date1, date2});

  dft::DateTimeTransformer dt("", "");
  dft::TimePoint tp1(dt.execute(date1_tp));
  dft::TimePoint tp2(dt.execute(date2_tp));

  // Date1
  ASSERT_EQ(tp1.year, 1976);
  ASSERT_EQ(tp1.month, dft::TimePoint::NOVEMBER);
  ASSERT_EQ(tp1.day, 17);
  ASSERT_EQ(tp1.hour, 12);
  ASSERT_EQ(tp1.minute, 27);
  ASSERT_EQ(tp1.second, 5);
  ASSERT_EQ(tp1.amPm, 1);
  ASSERT_EQ(tp1.hour12, 12);
  ASSERT_EQ(tp1.dayOfWeek, dft::TimePoint::WEDNESDAY);
  ASSERT_EQ(tp1.dayOfQuarter, 48);
  ASSERT_EQ(tp1.dayOfYear, 321);
  ASSERT_EQ(tp1.weekOfMonth, 2);
  ASSERT_EQ(tp1.quarterOfYear, 4);
  ASSERT_EQ(tp1.halfOfYear, 2);
  ASSERT_EQ(tp1.weekIso, 47);
  ASSERT_EQ(tp1.yearIso, 1976);
  ASSERT_EQ(tp1.monthLabel, "November");
  ASSERT_EQ(tp1.amPmLabel, "pm");
  ASSERT_EQ(tp1.dayOfWeekLabel, "Wednesday");
  ASSERT_EQ(tp1.holidayName, "");
  ASSERT_EQ(tp1.isPaidTimeOff, 0);

  // Date2
  ASSERT_EQ(tp2.year, 1976);
  ASSERT_EQ(tp2.month, dft::TimePoint::NOVEMBER);
  ASSERT_EQ(tp2.day, 17);
  ASSERT_EQ(tp2.hour, 12);
  ASSERT_EQ(tp2.minute, 27);
  ASSERT_EQ(tp2.second, 4);
  ASSERT_EQ(tp2.amPm, 1);
  ASSERT_EQ(tp2.hour12, 12);
  ASSERT_EQ(tp2.dayOfWeek, dft::TimePoint::WEDNESDAY);
  ASSERT_EQ(tp2.dayOfQuarter, 48);
  ASSERT_EQ(tp2.dayOfYear, 321);
  ASSERT_EQ(tp2.weekOfMonth, 2);
  ASSERT_EQ(tp2.quarterOfYear, 4);
  ASSERT_EQ(tp2.halfOfYear, 2);
  ASSERT_EQ(tp2.weekIso, 47);
  ASSERT_EQ(tp2.yearIso, 1976);
  ASSERT_EQ(tp2.monthLabel, "November");
  ASSERT_EQ(tp2.amPmLabel, "pm");
  ASSERT_EQ(tp2.dayOfWeekLabel, "Wednesday");
  ASSERT_EQ(tp2.holidayName, "");
  ASSERT_EQ(tp2.isPaidTimeOff, 0);

  // Expected output.
  test.AddOutput<int32_t>("year", {2}, {tp1.year, tp2.year});
  test.AddOutput<uint8_t>("month", {2}, {tp1.month, tp2.month});
  test.AddOutput<uint8_t>("day", {2}, {tp1.day, tp2.day});
  test.AddOutput<uint8_t>("hour", {2}, {tp1.hour, tp2.hour});
  test.AddOutput<uint8_t>("minute", {2}, {tp1.minute, tp2.minute});
  test.AddOutput<uint8_t>("second", {2}, {tp1.second, tp2.second});
  test.AddOutput<uint8_t>("amPm", {2}, {tp1.amPm, tp2.amPm});
  test.AddOutput<uint8_t>("hour12", {2}, {tp1.hour12, tp2.hour12});
  test.AddOutput<uint8_t>("dayOfWeek", {2}, {tp1.dayOfWeek, tp2.dayOfWeek});
  test.AddOutput<uint8_t>("dayOfQuarter", {2}, {tp1.dayOfQuarter, tp2.dayOfQuarter});
  test.AddOutput<uint16_t>("dayOfYear", {2}, {tp1.dayOfYear, tp2.dayOfYear});
  test.AddOutput<uint16_t>("weekOfMonth", {2}, {tp1.weekOfMonth, tp2.weekOfMonth});
  test.AddOutput<uint8_t>("quarterOfYear", {2}, {tp1.quarterOfYear, tp2.quarterOfYear});
  test.AddOutput<uint8_t>("halfOfYear", {2}, {tp1.halfOfYear, tp2.halfOfYear});
  test.AddOutput<uint8_t>("weekIso", {2}, {tp1.weekIso, tp2.weekIso});
  test.AddOutput<int32_t>("yearIso", {2}, {tp1.yearIso, tp2.yearIso});
  test.AddOutput<std::string>("monthLabel", {2}, {tp1.monthLabel, tp2.monthLabel});
  test.AddOutput<std::string>("amPmLabel", {2}, {tp1.amPmLabel, tp2.amPmLabel});
  test.AddOutput<std::string>("dayOfWeekLabel", {2}, {tp1.dayOfWeekLabel, tp2.dayOfWeekLabel});
  test.AddOutput<std::string>("holidayName", {2}, {tp1.holidayName, tp2.holidayName});
  test.AddOutput<uint8_t>("isPaidTimeOff", {2}, {tp1.isPaidTimeOff, tp2.isPaidTimeOff});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, DateTimeTransformer_future_2025_june_30) {
  const time_t date = 1751241600;
  const auto date_tp = std::chrono::system_clock::from_time_t(date);

  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  // Add state input
  auto stream = GetStream();
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("Date", {1}, {date});

  dft::DateTimeTransformer dt("", "");
  dft::TimePoint tp = dt.execute(date_tp);
  ASSERT_EQ(tp.year, 2025);
  ASSERT_EQ(tp.month, dft::TimePoint::JUNE);
  ASSERT_EQ(tp.day, 30);
  ASSERT_EQ(tp.hour, 0);
  ASSERT_EQ(tp.minute, 0);
  ASSERT_EQ(tp.second, 0);
  ASSERT_EQ(tp.dayOfWeek, dft::TimePoint::MONDAY);
  ASSERT_EQ(tp.dayOfYear, 180);
  ASSERT_EQ(tp.quarterOfYear, 2);
  ASSERT_EQ(tp.weekOfMonth, 4);
  ASSERT_EQ(tp.amPm, 0);
  ASSERT_EQ(tp.hour12, 0);
  ASSERT_EQ(tp.dayOfQuarter, 91);
  ASSERT_EQ(tp.halfOfYear, 1);
  ASSERT_EQ(tp.weekIso, 27);
  ASSERT_EQ(tp.yearIso, 2025);
  ASSERT_EQ(tp.monthLabel, "June");
  ASSERT_EQ(tp.amPmLabel, "am");
  ASSERT_EQ(tp.dayOfWeekLabel, "Monday");
  ASSERT_EQ(tp.holidayName, "");
  ASSERT_EQ(tp.isPaidTimeOff, 0);

  test.AddOutput<int32_t>("year", {1}, {tp.year});
  test.AddOutput<uint8_t>("month", {1}, {tp.month});
  test.AddOutput<uint8_t>("day", {1}, {tp.day});
  test.AddOutput<uint8_t>("hour", {1}, {tp.hour});
  test.AddOutput<uint8_t>("minute", {1}, {tp.minute});
  test.AddOutput<uint8_t>("second", {1}, {tp.second});
  test.AddOutput<uint8_t>("amPm", {1}, {tp.amPm});
  test.AddOutput<uint8_t>("hour12", {1}, {tp.hour12});
  test.AddOutput<uint8_t>("dayOfWeek", {1}, {tp.dayOfWeek});
  test.AddOutput<uint8_t>("dayOfQuarter", {1}, {tp.dayOfQuarter});
  test.AddOutput<uint16_t>("dayOfYear", {1}, {tp.dayOfYear});
  test.AddOutput<uint16_t>("weekOfMonth", {1}, {tp.weekOfMonth});
  test.AddOutput<uint8_t>("quarterOfYear", {1}, {tp.quarterOfYear});
  test.AddOutput<uint8_t>("halfOfYear", {1}, {tp.halfOfYear});
  test.AddOutput<uint8_t>("weekIso", {1}, {tp.weekIso});
  test.AddOutput<int32_t>("yearIso", {1}, {tp.yearIso});
  test.AddOutput<std::string>("monthLabel", {1}, {tp.monthLabel});
  test.AddOutput<std::string>("amPmLabel", {1}, {tp.amPmLabel});
  test.AddOutput<std::string>("dayOfWeekLabel", {1}, {tp.dayOfWeekLabel});
  test.AddOutput<std::string>("holidayName", {1}, {tp.holidayName});
  test.AddOutput<uint8_t>("isPaidTimeOff", {1}, {tp.isPaidTimeOff});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(FeaturizersTests, DateTimeTransformer_Country_Canada) {
  std::string const dataDir(onnxruntime::featurizers::GetDateTimeTransformerDataDir());

  if(dataDir.empty()) {
    GTEST_SKIP() <<
      "Skipping country-based tests, as the data directory could not be found. This likely indicates that\n"
      "the test is being invoked from a nuget installation, which isn't a scenario that is supported by\n"
      "featurizers (featurizers will only be used via the Python ORT wrappers and data is installed as\n"
      "part of the wheel).\n";
  }

  const time_t date = 157161600;
  const auto date_tp = std::chrono::system_clock::from_time_t(date);

  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  // Add state input
  auto stream = GetStream("Canada");
  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("Date", {1}, {date});

  dft::DateTimeTransformer dt("Canada", dataDir);
  dft::TimePoint tp = dt.execute(date_tp);

  ASSERT_EQ(tp.holidayName, "Christmas Day");

  test.AddOutput<int32_t>("year", {1}, {tp.year});
  test.AddOutput<uint8_t>("month", {1}, {tp.month});
  test.AddOutput<uint8_t>("day", {1}, {tp.day});
  test.AddOutput<uint8_t>("hour", {1}, {tp.hour});
  test.AddOutput<uint8_t>("minute", {1}, {tp.minute});
  test.AddOutput<uint8_t>("second", {1}, {tp.second});
  test.AddOutput<uint8_t>("amPm", {1}, {tp.amPm});
  test.AddOutput<uint8_t>("hour12", {1}, {tp.hour12});
  test.AddOutput<uint8_t>("dayOfWeek", {1}, {tp.dayOfWeek});
  test.AddOutput<uint8_t>("dayOfQuarter", {1}, {tp.dayOfQuarter});
  test.AddOutput<uint16_t>("dayOfYear", {1}, {tp.dayOfYear});
  test.AddOutput<uint16_t>("weekOfMonth", {1}, {tp.weekOfMonth});
  test.AddOutput<uint8_t>("quarterOfYear", {1}, {tp.quarterOfYear});
  test.AddOutput<uint8_t>("halfOfYear", {1}, {tp.halfOfYear});
  test.AddOutput<uint8_t>("weekIso", {1}, {tp.weekIso});
  test.AddOutput<int32_t>("yearIso", {1}, {tp.yearIso});
  test.AddOutput<std::string>("monthLabel", {1}, {tp.monthLabel});
  test.AddOutput<std::string>("amPmLabel", {1}, {tp.amPmLabel});
  test.AddOutput<std::string>("dayOfWeekLabel", {1}, {tp.dayOfWeekLabel});
  test.AddOutput<std::string>("holidayName", {1}, {tp.holidayName});
  test.AddOutput<uint8_t>("isPaidTimeOff", {1}, {tp.isPaidTimeOff});

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

}  // namespace test
}  // namespace onnxruntime
