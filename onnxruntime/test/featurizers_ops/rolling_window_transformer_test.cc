#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/AnalyticalRollingWindowFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {

using InputType = std::tuple<std::vector<std::string> const &, int32_t const &>;
using EstimatorT = NS::Featurizers::GrainedAnalyticalRollingWindowEstimator<int32_t>;

std::vector<uint8_t> GetStream(EstimatorT& estimator, const std::vector<InputType>& trainingBatches) {
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, RollingWindow_Transformer_Grained_Mean_1_grain_window_size_1_horizon_1) {
  //parameter setting
  EstimatorT      estimator(NS::CreateTestAnnotationMapsPtr(1), NS::Featurizers::AnalyticalRollingWindowCalculation::Mean, 1, 1);
  using GrainType = std::vector<std::string>;
  using GrainedInputType = std::tuple<GrainType, int32_t>;
  GrainType const grain({"one"});
  GrainedInputType const tup1 = std::make_tuple(grain, 1);
  std::vector<std::tuple<std::vector<std::string> const &, std::int32_t const &>> const training_batch = {tup1};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("RollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {3, 1}, {"one", "one", "one"});
  test.AddInput<int32_t>("Target", {3}, {1, 2, 3});
  test.AddOutput<double>("Output", {3, 1}, {NS::Traits<double>::CreateNullValue(), 1.0, 2.0});

  test.Run();
}

TEST(FeaturizersTests, RollingWindow_Transformer_Grained_Mean_1_grain_window_size_2_horizon_2_min_window_size_2) {
  //parameter setting
  EstimatorT      estimator(NS::CreateTestAnnotationMapsPtr(1), NS::Featurizers::AnalyticalRollingWindowCalculation::Mean, 2, 2, 2);
  using GrainType = std::vector<std::string>;
  using GrainedInputType = std::tuple<GrainType, int32_t>;
  GrainType const grain({"one"});
  GrainedInputType const tup1 = std::make_tuple(grain, 1);
  std::vector<std::tuple<std::vector<std::string> const &, std::int32_t const &>> const training_batch = {tup1};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("RollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {4, 1}, {"one", "one", "one", "one"});
  test.AddInput<int32_t>("Target", {4}, {1, 2, 3, 4});
  test.AddOutput<double>("Output", {4, 2}, {NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            1.5,
                                            1.5,
                                            2.5});

  test.Run();
}

TEST(FeaturizersTests, RollingWindow_Transformer_Grained_Mean_2_grain_window_size_2_horizon_2_min_window_size_2) {
  EstimatorT      estimator(NS::CreateTestAnnotationMapsPtr(1), NS::Featurizers::AnalyticalRollingWindowCalculation::Mean, 2, 2);
  using GrainType = std::vector<std::string>;
  using GrainedInputType = std::tuple<GrainType, std::int32_t>;
  const GrainType grainOne({"one"});
  const GrainType grainTwo({"two"});
  const GrainedInputType tup1 = std::make_tuple(grainOne, 1);
  const GrainedInputType tup2 = std::make_tuple(grainTwo, 1);
  const std::vector<std::tuple<std::vector<std::string> const &, std::int32_t const &>> training_batch = {tup1, tup2};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("RollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {4, 1}, {"one", "two", "one", "two"});
  test.AddInput<int32_t>("Target", {4}, {1, 1, 2, 2});
  test.AddOutput<double>("Output", {4, 2}, {NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            NS::Traits<double>::CreateNullValue(),
                                            1.0,
                                            NS::Traits<double>::CreateNullValue(),
                                            1.0});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
