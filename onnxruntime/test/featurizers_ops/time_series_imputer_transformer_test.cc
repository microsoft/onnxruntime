// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/TimeSeriesImputerFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

inline std::chrono::system_clock::time_point GetTimePoint(std::chrono::system_clock::time_point tp, int unitsToAdd, std::string = "days") {
  return tp + std::chrono::minutes(unitsToAdd * (60 * 24));
}

inline int64_t GetTimeInt(std::chrono::system_clock::time_point tp, int unitsToAdd) {
  using namespace std::chrono;
  return duration_cast<seconds>(GetTimePoint(tp, unitsToAdd).time_since_epoch()).count();
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

std::vector<uint8_t> GetStream(const std::vector<std::vector<InputType>>& trainingBatches,
                               std::vector<NS::TypeId> colsToImputeDataTypes,
                               bool supressError, NS::Featurizers::Components::TimeSeriesImputeStrategy tsImputeStrategy) {
  using KeyT = std::vector<std::string>;
  using ColsToImputeT = std::vector<nonstd::optional<std::string>>;
  using InputBatchesType = std::vector<std::vector<InputType>>;
  using TSImputerEstimator = NS::Featurizers::TimeSeriesImputerEstimator;

  NS::AnnotationMapsPtr const pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  TSImputerEstimator estimator(pAllColumnAnnotations, colsToImputeDataTypes, supressError, tsImputeStrategy);

  NS::TestHelpers::Train<TSImputerEstimator, InputType>(estimator, trainingBatches);
  TSImputerEstimator::TransformerUniquePtr pTransformer(estimator.create_transformer());

  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

static void AddInputs (OpTester& test, const std::vector<InputType>& inferenceBatches) {

  auto stream = GetStream(
      {inferenceBatches},
      {NS::TypeId::Float64, NS::TypeId::Float64},
      false,
      NS::Featurizers::Components::TimeSeriesImputeStrategy::Forward);

  auto dim = static_cast<int64_t>(stream.size());
  test.AddInput<uint8_t>("State", {dim}, stream);

  std::vector<int64_t> times;
  std::vector<std::string> keys;
  std::vector<std::string> data;

  for (const auto& infb : inferenceBatches) {
    times.push_back(std::get<0>(infb).time_since_epoch().count());
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

void AddOutputs(OpTester& test, const std::initializer_list<bool>& added, const std::initializer_list<int64_t>& times,
  const std::vector<std::string>& keys, const std::vector<std::string>& data) {

  ASSERT_TRUE(keys.size() % times.size() == 0);
  ASSERT_TRUE(data.size() % times.size() == 0);
  test.AddOutput<bool>("Added", {static_cast<int64_t>(added.size())}, added);
  test.AddOutput<int64_t>("ImputedTimes", {static_cast<int64_t>(times.size())}, times);
  test.AddOutput<std::string>("ImputedKeys", {static_cast<int64_t>(times.size()), static_cast<int64_t>(keys.size() / times.size())}, keys);
  test.AddOutput<std::string>("ImputedData", {static_cast<int64_t>(times.size()), static_cast<int64_t>(data.size() / times.size())}, data);
}

TEST(FeaturizersTests, RowImputation_1_grain_no_gaps) {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  auto tuple_1 = std::make_tuple(GetTimePoint(now, 0), std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"14.5", "18"});
  auto tuple_2 = std::make_tuple(GetTimePoint(now, 1), std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{nonstd::optional<std::string>{}, "12"});
  auto tuple_3 = std::make_tuple(GetTimePoint(now, 2), std::vector<std::string>{"a"}, std::vector<nonstd::optional<std::string>>{"15.0", nonstd::optional<std::string>{}});

  std::vector<InputType> inferenceBatches = {tuple_1,
                                             tuple_2,
                                             tuple_3};

  OpTester test("TimeSeriesImputerTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  AddInputs(test, inferenceBatches);
  AddOutputs(test, {false, false, false}, {GetTimeInt(now, 0), GetTimeInt(now, 1), GetTimeInt(now, 2)},
             {"a", "a", "a"}, {"14.5", "18", "14.5", "12", "15.0", "12"});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
