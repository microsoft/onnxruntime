#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/ShortGrainDropperFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {

using InputType = std::vector<std::string>;
using EstimatorT = NS::Featurizers::ShortGrainDropperEstimator<std::numeric_limits<size_t>::max()>;

std::vector<uint8_t> GetStream(EstimatorT& estimator, const std::vector<std::vector<InputType>>& trainingBatches) {
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, ShortGrainDropperTransformer_Has_CV) {

  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0, 4);

  std::vector<std::vector<std::vector<std::string>>> trainingBatches = NS::TestHelpers::make_vector<std::vector<std::vector<std::string>>>(
    NS::TestHelpers::make_vector<std::vector<std::string>>(
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "e"),
      NS::TestHelpers::make_vector<std::string>("a", "e"),
      NS::TestHelpers::make_vector<std::string>("a", "f")
    )
  );

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ShortGrainDropperTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {6, 2}, {"a", "b", "a", "c", "a", "d", "a", "e", "a", "f", "a", "g"});
  test.AddOutput<bool>("Output", {6}, {false, true, true, true, true, false});

  test.Run();
}

TEST(FeaturizersTests, ShortGrainDropperTransformer_No_CV) {

  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0, 3);

  std::vector<std::vector<std::vector<std::string>>> trainingBatches = NS::TestHelpers::make_vector<std::vector<std::vector<std::string>>>(
    NS::TestHelpers::make_vector<std::vector<std::string>>(
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "e"),
      NS::TestHelpers::make_vector<std::string>("a", "e"),
      NS::TestHelpers::make_vector<std::string>("a", "f")
    )
  );

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ShortGrainDropperTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {6, 2}, {"a", "b", "a", "c", "a", "d", "a", "e", "a", "f", "a", "g"});
  test.AddOutput<bool>("Output", {6}, {false, false, true, true, true, false});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
