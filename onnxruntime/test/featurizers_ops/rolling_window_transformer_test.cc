#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/AnalyticalRollingWindowFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {

using InputType = std::tuple<std::vector<std::string> const &, std::int32_t const &>;
using EstimatorT = NS::Featurizers::GrainedAnalyticalRollingWindowEstimator<std::int32_t>;
using AnalyticalRollingWindowCalculation = NS::Featurizers::AnalyticalRollingWindowCalculation;

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
  NS::AnnotationMapsPtr                   pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  NS::Featurizers::GrainedAnalyticalRollingWindowEstimator<int32_t>      estimator(pAllColumnAnnotations, 1, AnalyticalRollingWindowCalculation::Mean, 1);
  using GrainType = std::vector<std::string>;
  using GrainedInputType = std::tuple<GrainType, int32_t>;
  GrainType const grain({"one"});
  InputType const tup1 = std::make_tuple(grain, 1);
  std::vector<InputType> const training_batch = {tup1};

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("RollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {3, 1}, {"one", "one", "one"});
  test.AddInput<int32_t>("Target", {3}, {1, 2, 3});
  test.AddOutput<double>("Output", {3, 1}, {NS::Traits<double>::CreateNullValue(), 1.0, 2.0});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
