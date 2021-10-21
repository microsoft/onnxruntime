#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/CountVectorizerFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

using IndexMapType = std::unordered_map<std::string, std::uint32_t>;
using AnalyzerMethod = NS::Featurizers::Components::AnalyzerMethod;

namespace onnxruntime {
namespace test {
namespace {

using InputType = std::string;
using EstimatorT = NS::Featurizers::CountVectorizerEstimator<std::numeric_limits<size_t>::max()>;

std::vector<uint8_t> GetStream(EstimatorT& estimator, const std::vector<std::vector<InputType>>& trainingBatches) {
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

// string - without binary with decorator, analyze word, maxdf = 1, mindf = 0, topk = null, empty vocabulary, ngram_min = 1, ngram_max = 1"
TEST(FeaturizersTests, CountVectorizerTransformer_string_nobinary_with_decorator) {

  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0, true, AnalyzerMethod::Word, "",
                       1.0, 0, nonstd::optional<std::uint32_t>(), 1, 1, false);

  auto trainingBatches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>("oraNge apple oranGE grape"),
      NS::TestHelpers::make_vector<InputType>("grApe caRrOt carrot apple"),
      NS::TestHelpers::make_vector<InputType>("peach Banana orange banana"));

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("CountVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {1}, {"banana grape grape apple apple apple orange"});
  test.AddOutput<uint32_t>("Output", {6}, {3, 1, 0, 2, 1, 0});

  test.Run();
}

// string - with binary with decorator, analyze word, maxdf = 1, mindf = 0, topk = null, empty vocabulary, ngram_min = 1, ngram_max = 1
TEST(FeaturizersTests, CountVectorizerTransformer_string_withbinary_withdecorator) {

  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0, false, AnalyzerMethod::Word, "",
                       1.0, 0, nonstd::optional<std::uint32_t>(), 1, 1, true);

  auto trainingBatches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>("orange apple orange grape"),
      NS::TestHelpers::make_vector<InputType>("grape carrot carrot apple"),
      NS::TestHelpers::make_vector<InputType>("peach banana orange banana"));

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("CountVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);

  test.AddInput<std::string>("Input", {1}, {"banana grape grape apple apple apple orange"});
  test.AddOutput<uint32_t>("Output", {6}, {1, 1, 0, 1, 1, 0});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
