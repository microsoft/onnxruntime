// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/TimeSeriesImputerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

inline std::chrono::system_clock::time_point GetTimePoint(std::chrono::system_clock::time_point tp, int unitsToAdd, std::string = "days") {
  return tp + std::chrono::minutes(unitsToAdd * (60 * 24));
}

inline int64_t GetTimeSecs(std::chrono::system_clock::time_point tp) {
  using namespace std::chrono;
  return time_point_cast<seconds>(tp).time_since_epoch().count();
}

using InputType = std::tuple<
    std::chrono::system_clock::time_point,
    std::vector<std::string>,
    std::vector<nonstd::optional<std::string>>>;

using TransformedType = std::vector<
    std::tuple<
        bool,
        std::chrono::system_clock::time_point,
        std::vector<std::string>,
        std::vector<nonstd::optional<std::string>>>>;

namespace {
std::vector<uint8_t> GetStream(const std::vector<std::vector<InputType>>& training_batches,
                               const std::vector<NS::TypeId>& col_to_impute_data_types,
                               bool suppress_error, NS::Featurizers::Components::TimeSeriesImputeStrategy impute_strategy) {
  using TSImputerEstimator = NS::Featurizers::TimeSeriesImputerEstimator;

  NS::AnnotationMapsPtr const pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  TSImputerEstimator estimator(pAllColumnAnnotations, col_to_impute_data_types, suppress_error, impute_strategy);

  NS::TestHelpers::Train<TSImputerEstimator, InputType>(estimator, training_batches);
  TSImputerEstimator::TransformerUniquePtr pTransformer(estimator.create_transformer());

  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

static void AddInputs(OpTester& test, const std::vector<std::vector<InputType>>& training_batches,
                      const std::vector<InputType>& inference_batches, const std::vector<NS::TypeId>& cols_to_impute_data_types,
                      bool suppress_error, NS::Featurizers::Components::TimeSeriesImputeStrategy impute_strategy) {
  auto stream = GetStream(
      training_batches,
      cols_to_impute_data_types,
      suppress_error,
      impute_strategy);

  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  std::vector<int64_t> times;
  std::vector<std::string> keys;
  std::vector<std::string> data;

  using namespace std::chrono;
  for (const auto& infb : inference_batches) {
    times.push_back(time_point_cast<seconds>(std::get<0>(infb)).time_since_epoch().count());
    keys.insert(keys.end(), std::get<1>(infb).cbegin(), std::get<1>(infb).cend());
    std::transform(std::get<2>(infb).cbegin(), std::get<2>(infb).cend(), std::back_inserter(data),
                   [](const nonstd::optional<std::string>& opt) -> std::string {
                     if (opt.has_value()) return *opt;
                     return std::string();
                   });
  }

  // Should have equal amount of keys per row
  ASSERT_TRUE(keys.size() % times.size() == 0);
  ASSERT_TRUE(data.size() % times.size() == 0);
  test.AddInput<int64_t>("Times", {static_cast<int64_t>(times.size())}, times);
  test.AddInput<std::string>("Keys", {static_cast<int64_t>(times.size()), static_cast<int64_t>(keys.size() / times.size())}, keys);
  test.AddInput<std::string>("Data", {static_cast<int64_t>(times.size()), static_cast<int64_t>(data.size() / times.size())}, data);
}

void AddOutputs(OpTester& test, const std::initializer_list<bool>& added, const std::initializer_list<std::chrono::system_clock::time_point>& times,
                const std::vector<std::string>& keys, const std::vector<std::string>& data) {
  ASSERT_TRUE(keys.size() % times.size() == 0);
  ASSERT_TRUE(data.size() % times.size() == 0);

  std::vector<int64_t> times_int64;
  std::transform(times.begin(), times.end(), std::back_inserter(times_int64), GetTimeSecs);

  test.AddOutput<bool>("Added", {static_cast<int64_t>(added.size())}, added);
  test.AddOutput<int64_t>("ImputedTimes", {static_cast<int64_t>(times.size())}, times_int64);
  test.AddOutput<std::string>("ImputedKeys", {static_cast<int64_t>(times.size()), static_cast<int64_t>(keys.size() / times.size())}, keys);
  test.AddOutput<std::string>("ImputedData", {static_cast<int64_t>(times.size()), static_cast<int64_t>(data.size() / times.size())}, data);
}

TEST(FeaturizersTests, RowImputation_1_grain_no_gaps) {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto tp_0 = GetTimePoint(now, 0);
  auto tp_1 = GetTimePoint(now, 1);
  auto tp_2 = GetTimePoint(now, 2);
  auto tuple_1 = std::make_tuple(tp_0, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_2 = std::make_tuple(tp_1, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_3 = std::make_tuple(tp_2, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"15.0", nonstd::optional<std::string>{}});

  std::vector<InputType> inference_batches = {tuple_1,
                                             tuple_2,
                                             tuple_3};

  OpTester test("TimeSeriesImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  AddInputs(test, {inference_batches}, inference_batches,
            {NS::TypeId::Float64, NS::TypeId::Float64}, false, NS::Featurizers::Components::TimeSeriesImputeStrategy::Forward);
  AddOutputs(test, {false, false, false}, {tp_0, tp_1, tp_2},
             {"a", "a", "a"}, {"14.5", "18", "14.5", "12", "15.0", "12"});

  test.Run();
}

TEST(FeaturizersTests, RowImputation_1_grain_2_gaps) {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto tp_0 = GetTimePoint(now, 0);
  auto tp_1 = GetTimePoint(now, 1);
  auto tp_2 = GetTimePoint(now, 2);
  auto tp_3 = GetTimePoint(now, 3);

  auto tuple_0 = std::make_tuple(tp_0, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_1 = std::make_tuple(tp_1, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_3 = std::make_tuple(tp_3, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "15.0"});

  OpTester test("TimeSeriesImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  AddInputs(test, {{tuple_0, tuple_1}}, {tuple_0, tuple_3},
            {NS::TypeId::Float64, NS::TypeId::Float64}, false, NS::Featurizers::Components::TimeSeriesImputeStrategy::Forward);

  AddOutputs(test, {false, true, true, false}, {tp_0, tp_1, tp_2, tp_3},
             {"a", "a", "a", "a"}, {"14.5", "18", "14.5", "18", "14.5", "18", "14.5", "15.0"});
  test.Run();
}

TEST(FeaturizersTests, RowImputation_2_grains_no_gaps_input_interleaved) {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto tp_0 = GetTimePoint(now, 0);
  auto tp_1 = GetTimePoint(now, 1);
  auto tp_5 = GetTimePoint(now, 5);
  auto tp_6 = GetTimePoint(now, 6);

  auto tuple_0 = std::make_tuple(tp_0, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_5 = std::make_tuple(tp_5, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_5_inf = std::make_tuple(GetTimePoint(now, 5), std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{"114.5", "118"});
  auto tuple_1 = std::make_tuple(tp_1, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_6 = std::make_tuple(tp_6, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_6_inf = std::make_tuple(GetTimePoint(now, 6), std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "112"});

  OpTester test("TimeSeriesImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  AddInputs(test, {{tuple_0, tuple_5, tuple_1, tuple_6}}, {tuple_0, tuple_5_inf, tuple_1, tuple_6_inf},
            {NS::TypeId::Float64, NS::TypeId::Float64}, false, NS::Featurizers::Components::TimeSeriesImputeStrategy::Forward);

  AddOutputs(test, {false, false, false, false}, {tp_0, tp_5, tp_1, tp_6},
             {"a", "b", "a", "b"}, {"14.5", "18", "114.5", "118", "14.5", "12", "114.5", "112"});
  test.Run();
}

TEST(FeaturizersTests, RowImputation_2_grains_1_gap_input_interleaved) {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto tp_0 = GetTimePoint(now, 0);
  auto tp_1 = GetTimePoint(now, 1);
  auto tp_2 = GetTimePoint(now, 2);
  auto tp_5 = GetTimePoint(now, 5);
  auto tp_6 = GetTimePoint(now, 6);
  auto tp_7 = GetTimePoint(now, 7);

  auto tuple_0 = std::make_tuple(tp_0, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_2 = std::make_tuple(GetTimePoint(now, 2), std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_5 = std::make_tuple(tp_5, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_5_inf = std::make_tuple(tp_5, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{"114.5", "118"});
  auto tuple_1 = std::make_tuple(tp_1, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_6 = std::make_tuple(tp_6, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_7 = std::make_tuple(GetTimePoint(now, 7), std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "112"});

  OpTester test("TimeSeriesImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  AddInputs(test, {{tuple_0, tuple_5, tuple_1, tuple_6}}, {tuple_0, tuple_5_inf, tuple_2, tuple_7},
            {NS::TypeId::Float64, NS::TypeId::Float64}, false, NS::Featurizers::Components::TimeSeriesImputeStrategy::Forward);
  AddOutputs(test, {false, false, true, false, true, false}, {tp_0, tp_5, tp_1, tp_2, tp_6, tp_7},
             {"a", "b", "a", "a", "b", "b"}, {"14.5", "18", "114.5", "118", "14.5", "18", "14.5", "12", "114.5", "118", "114.5", "112"});
  test.Run();
}

TEST(FeaturizersTests, RowImputation_2_grains_1_gap_input_interleaved_add_additional_imputed_columns) {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto tp_0 = GetTimePoint(now, 0);
  auto tp_1 = GetTimePoint(now, 1);
  auto tp_2 = GetTimePoint(now, 2);
  auto tp_5 = GetTimePoint(now, 5);
  auto tp_6 = GetTimePoint(now, 6);
  auto tp_7 = GetTimePoint(now, 7);

  auto tuple_0 = std::make_tuple(tp_0, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_2 = std::make_tuple(GetTimePoint(now, 2), std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_5 = std::make_tuple(tp_5, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_5_inf = std::make_tuple(tp_5, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{"114.5", "118"});
  auto tuple_1 = std::make_tuple(tp_1, std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_6 = std::make_tuple(tp_6, std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_7 = std::make_tuple(GetTimePoint(now, 7), std::vector<std::string>{"b"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "112"});

  OpTester test("TimeSeriesImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  AddInputs(test, {{tuple_0, tuple_5, tuple_1, tuple_6}}, {tuple_0, tuple_5_inf, tuple_2, tuple_7},
            {NS::TypeId::Float64, NS::TypeId::Float64}, false, NS::Featurizers::Components::TimeSeriesImputeStrategy::Forward);
  test.AddInput<int64_t>("Input_1", {4, 1}, {1, 2, 3, 4});
  test.AddInput<float>("Input_2", {4, 1}, {1, 2, 3, 4});
  test.AddInput<std::string>("Input_3", {4, 1}, {"1", "2", "3", "4"});
  test.AddInput<bool>("Input_4", {4, 1}, {false, true, true, true});
  AddOutputs(test, {false, false, true, false, true, false}, {tp_0, tp_5, tp_1, tp_2, tp_6, tp_7},
             {"a", "b", "a", "a", "b", "b"}, {"14.5", "18", "114.5", "118", "14.5", "18", "14.5", "12", "114.5", "118", "114.5", "112"});
  test.AddOutput<int64_t>("Output_1", {6, 1}, {1, 2, 0, 3, 0, 4});
  test.AddOutput<float>("Output_2", {6, 1}, {1, 2, std::numeric_limits<float>::quiet_NaN(), 3, std::numeric_limits<float>::quiet_NaN(), 4});
  test.AddOutput<std::string>("Output_3", {6, 1}, {"1", "2", "", "3", "", "4"});
  test.AddOutput<bool>("Output_4", {6, 1}, {false, true, false, true, false, true});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
