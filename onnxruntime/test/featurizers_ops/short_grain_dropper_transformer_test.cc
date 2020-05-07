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

TEST(FeaturizersTests, ShortGrainDropperTransformer_Min_4) {

  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0, 4);

  std::vector<std::vector<std::vector<std::string>>> trainingBatches = NS::TestHelpers::make_vector<std::vector<std::vector<std::string>>>(
    NS::TestHelpers::make_vector<std::vector<std::string>>(
      NS::TestHelpers::make_vector<std::string>("a", "b"), //false
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "c"), //false
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "d"), //true
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "e"), //true
      NS::TestHelpers::make_vector<std::string>("a", "e"),
      NS::TestHelpers::make_vector<std::string>("a", "f")  //true
    )
  );

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ShortGrainDropperTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("GrainInput", {6, 2}, {"a", "b", "a", "c", "a", "d", "a", "e", "a", "f", "a", "g"});
  test.AddInput<std::string>("Non_GrainInput_1", {6, 2}, {"c", "c", "d", "d", "e", "e", "e", "e", "e", "e", "e", "e"});
  test.AddInput<double>("Non_GrainInput_2", {6, 1}, {1, 2, 3, 4, 5, 6});

  test.AddOutput<std::string>("GrainOutput", {2, 2}, {"a", "b", "a", "c"});
  test.AddOutput<std::string>("Non_GrainOutput_1", {2, 2}, {"c", "c", "d", "d"});
  test.AddOutput<double>("Non_GrainOutput_2", {2, 1}, {1, 2});

  test.Run();
}

TEST(FeaturizersTests, ShortGrainDropperTransformer_Min_3) {

  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0, 3);

  std::vector<std::vector<std::vector<std::string>>> trainingBatches = NS::TestHelpers::make_vector<std::vector<std::vector<std::string>>>(
    NS::TestHelpers::make_vector<std::vector<std::string>>(
      NS::TestHelpers::make_vector<std::string>("a", "b"), //false
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "c"), //false
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "d"), //false
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "e"), //true
      NS::TestHelpers::make_vector<std::string>("a", "e"),
      NS::TestHelpers::make_vector<std::string>("a", "f")  //true
    )
  );

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ShortGrainDropperTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("GrainInput", {6, 2}, {"a", "b", "a", "c", "a", "d", "a", "e", "a", "f", "a", "g"});
  test.AddInput<std::string>("Non_GrainInput_1", {6, 2}, {"c", "c", "d", "d", "e", "e", "e", "e", "e", "e", "e", "e"});
  test.AddInput<double>("Non_GrainInput_2", {6, 1}, {1, 2, 3, 4, 5, 6});

  test.AddOutput<std::string>("GrainOutput", {3, 2}, {"a", "b", "a", "c", "a", "d"});
  test.AddOutput<std::string>("Non_GrainOutput_1", {3, 2}, {"c", "c", "d", "d", "e", "e"});
  test.AddOutput<double>("Non_GrainOutput_2", {3, 1}, {1, 2, 3});

  test.Run();
}

TEST(FeaturizersTests, ShortGrainDropperTransformer_Min_2) {

  EstimatorT estimator(NS::CreateTestAnnotationMapsPtr(1), 0, 2);

  std::vector<std::vector<std::vector<std::string>>> trainingBatches = NS::TestHelpers::make_vector<std::vector<std::vector<std::string>>>(
    NS::TestHelpers::make_vector<std::vector<std::string>>(
      NS::TestHelpers::make_vector<std::string>("a", "b"), //false
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "b"),
      NS::TestHelpers::make_vector<std::string>("a", "c"), //false
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "c"),
      NS::TestHelpers::make_vector<std::string>("a", "d"), //false
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "d"),
      NS::TestHelpers::make_vector<std::string>("a", "e"), //false
      NS::TestHelpers::make_vector<std::string>("a", "e"),
      NS::TestHelpers::make_vector<std::string>("a", "f")  //true
    )
  );

  auto stream = GetStream(estimator, trainingBatches);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ShortGrainDropperTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("GrainInput", {6, 2}, {"a", "b", "a", "c", "a", "d", "a", "e", "a", "f", "a", "g"});
  test.AddInput<std::string>("Non_GrainInput_1", {6, 2}, {"c", "c", "d", "d", "e", "e", "e", "e", "e", "e", "e", "e"});
  test.AddInput<double>("Non_GrainInput_2", {6, 1}, {1, 2, 3, 4, 5, 6});

  test.AddOutput<std::string>("GrainOutput", {4, 2}, {"a", "b", "a", "c", "a", "d", "a", "e"});
  test.AddOutput<std::string>("Non_GrainOutput_1", {4, 2}, {"c", "c", "d", "d", "e", "e", "e", "e"});
  test.AddOutput<double>("Non_GrainOutput_2", {4, 1}, {1, 2, 3, 4});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
