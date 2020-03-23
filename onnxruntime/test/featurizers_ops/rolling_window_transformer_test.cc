#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/AnalyticalRollingWindowFeaturizer.h"
#include "Featurizers/../Archive.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {

using InputType = int32_t;
using TransformerT = NS::Featurizers::AnalyticalRollingWindowTransformer<InputType>;

std::vector<uint8_t> GetTransformerStream(TransformerT& transformer) {
  NS::Archive ar;
  transformer.save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, RollingWindow_Transformer_Draft) {
  //parameter setting
  TransformerT transformer(1, NS::Featurizers::AnalyticalRollingWindowCalculation::Mean, 2);

  auto stream = GetTransformerStream(transformer);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("RollingWindowTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {4, 1}, {"a", "a", "a", "a"});
  test.AddInput<int32_t>("Target", {4}, {1, 2, 3, 4});
  test.AddOutput<double>("Output", {4, 2}, {NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(), 1.0, 1.0, 2.0, 2.0, 3.0});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
