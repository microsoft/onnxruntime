#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/AnalyticalRollingWindowFeaturizer.h"
#include "Featurizers/SimpleRollingWindowFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {

using InputType = std::tuple<std::vector<std::string> const &, double const &>;

template<typename EstimatorT>
std::vector<uint8_t> GetStream(EstimatorT& estimator, const std::vector<InputType>& trainingBatches) {
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, AnalyticalRollingWindow_Transformer_Grained_Mean_1_grain_window_size_1_horizon_1) {
  using EstimatorT = NS::Featurizers::GrainedAnalyticalRollingWindowEstimator<double>;
  using GrainType = std::vector<std::string>;
  using GrainedInputType = EstimatorT::InputType;

  EstimatorT      estimator(NS::CreateTestAnnotationMapsPtr(1), NS::Featurizers::AnalyticalRollingWindowCalculation::Mean, 1, 1);

  const GrainType grain({"one"});
  const double value1(static_cast<double>(10));
  const GrainedInputType tup1(grain, value1);
  const std::vector<std::tuple<std::vector<std::string> const &, double const &>> training_batch = {tup1};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("AnalyticalRollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {3, 1}, {"one", "one", "one"});
  test.AddInput<double>("Target", {3}, {1, 2, 3});

  test.AddOutput<double>("Output", {3, 1,  1}, {NS::Traits<double>::CreateNullValue(), 1.0, 2.0});

  test.Run();
}

TEST(FeaturizersTests, AnalyticalRollingWindow_Transformer_Grained_Mean_1_grain_window_size_2_horizon_2_min_window_size_2) {
  using EstimatorT = NS::Featurizers::GrainedAnalyticalRollingWindowEstimator<double>;
  using GrainType = std::vector<std::string>;
  using GrainedInputType = EstimatorT::InputType;

  EstimatorT      estimator(NS::CreateTestAnnotationMapsPtr(1), NS::Featurizers::AnalyticalRollingWindowCalculation::Mean, 2, 2, 2);

  const GrainType grain({"one"});
  const double value1(static_cast<double>(1));
  const GrainedInputType tup1(grain, value1);
  const std::vector<std::tuple<std::vector<std::string> const &, double const &>> training_batch = {tup1};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("AnalyticalRollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {4, 1}, {"one", "one", "one", "one"});
  test.AddInput<double>("Target", {4}, {1, 2, 3, 4});

  test.AddOutput<double>("Output", {4, 1, 2}, {NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               1.5,
                                               1.5,
                                               2.5});

  test.Run();
}

TEST(FeaturizersTests, AnalyticalRollingWindow_Transformer_Grained_Mean_2_grain_window_size_2_horizon_2_min_window_size_2) {
  using EstimatorT = NS::Featurizers::GrainedAnalyticalRollingWindowEstimator<double>;
  using GrainType = std::vector<std::string>;
  using GrainedInputType = EstimatorT::InputType;

  EstimatorT      estimator(NS::CreateTestAnnotationMapsPtr(1), NS::Featurizers::AnalyticalRollingWindowCalculation::Mean, 2, 2);

  const GrainType grainOne({"one"});
  const GrainType grainTwo({"two"});
  const double value1(static_cast<double>(1));
  const double value2(static_cast<double>(1));
  const GrainedInputType tup1(grainOne, value1);
  const GrainedInputType tup2(grainTwo, value2);
  const std::vector<std::tuple<std::vector<std::string> const &, double const &>> training_batch = {tup1, tup2};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("AnalyticalRollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {4, 1}, {"one", "two", "one", "two"});
  test.AddInput<double>("Target", {4}, {1, 1, 2, 2});

  test.AddOutput<double>("Output", {4, 1, 2}, {NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               NS::Traits<double>::CreateNullValue(),
                                               1.0,
                                               NS::Traits<double>::CreateNullValue(),
                                               1.0});

  test.Run();
}

TEST(FeaturizersTests, SimpleRollingWindow_Transformer_Grained_Min_1_grain_window_size_1_horizon_1) {
  using EstimatorT = NS::Featurizers::GrainedSimpleRollingWindowEstimator<double>;
  using GrainType = std::vector<std::string>;
  using GrainedInputType = EstimatorT::InputType;

  EstimatorT      estimator(NS::CreateTestAnnotationMapsPtr(1), NS::Featurizers::SimpleRollingWindowCalculation::Min, 1, 1);

  const GrainType grainOne({"one"});
  const double value1(static_cast<double>(1));
  const GrainedInputType tup1(grainOne, value1);
  const std::vector<std::tuple<std::vector<std::string> const &, double const &>> training_batch = {tup1};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("SimpleRollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {3, 1}, {"one", "one", "one"});
  test.AddInput<double>("Target", {3}, {1, 2, 3});

  test.AddOutput<double>("Output", {3, 1, 1}, {NS::Traits<double>::CreateNullValue(), 1, 2});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
