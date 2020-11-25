#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/TfidfVectorizerFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

using IndexMapType = std::unordered_map<std::string, std::uint32_t>;
using AnalyzerMethod = NS::Featurizers::Components::AnalyzerMethod;

namespace onnxruntime {
namespace test {
namespace {

using InputType = std::string;
using TransformedType = NS::Featurizers::SparseVectorEncoding<std::float_t>;
using EstimatorT = NS::Featurizers::TfidfVectorizerEstimator<>;

std::vector<uint8_t> GetStream(EstimatorT& estimator, const std::vector<std::vector<InputType>>& trainingBatches) {
  NS::TestHelpers::Train<EstimatorT, InputType>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}
}  // namespace

TEST(FeaturizersTests, TfidfVectorizerTransformer_string_standard_1_with_decorator) {
  EstimatorT estimator(
      NS::CreateTestAnnotationMapsPtr(1),
      0,
      true,
      AnalyzerMethod::Word,
      "");

  auto trainingBatches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>("this is THE first document"),
      NS::TestHelpers::make_vector<InputType>("this DOCUMENT is the second document"),
      NS::TestHelpers::make_vector<InputType>("and this is the THIRD one"),
      NS::TestHelpers::make_vector<InputType>("IS this THE first document"));

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("TfidfVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Input", {1}, {"THIS is the FIRST document"});
  test.AddOutput<float>("Output", {9}, {0.f, 0.469791f, 0.580286f, 0.384085f, 0.f, 0.f, 0.384085f, 0.f, 0.384085f});
  test.Run();
}

TEST(FeaturizersTests, TfidfVectorizerTransformer_string_standard_1_no_decorator) {

   EstimatorT estimator(
      NS::CreateTestAnnotationMapsPtr(1),
      0,
      false,
      AnalyzerMethod::Word,
      "");

  auto trainingBatches = NS::TestHelpers::make_vector<std::vector<InputType>>(
      NS::TestHelpers::make_vector<InputType>("this is the first document"),
      NS::TestHelpers::make_vector<InputType>("this document is the second document"),
      NS::TestHelpers::make_vector<InputType>("and this is the third one"),
      NS::TestHelpers::make_vector<InputType>("is this the first document"));

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("TfidfVectorizerTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);

  test.AddInput<std ::string>("Input", {1}, {"this is the first document"});
  test.AddOutput<float>("Output", {9}, {0.f, 0.469791f, 0.580286f, 0.384085f, 0.f, 0.f, 0.384085f, 0.f, 0.384085f});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
