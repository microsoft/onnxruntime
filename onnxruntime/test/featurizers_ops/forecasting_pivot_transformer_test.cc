// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/../Archive.h"
#include "Featurizers/ForecastingPivotFeaturizer.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

namespace {

template <typename T>
std::vector<uint8_t> GetStream() {
  NS::Featurizers::ForecastingPivotTransformer<T> transformer;
  NS::Archive ar;
  transformer.save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, ForecastingPivotTransformer) {
  auto stream = GetStream<float>();
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ForecastingPivotTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<double>("Tensor1", {2, 3, 4}, {1, 6, 3, 9,
                                              2, 4, 5, 8,
                                              NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), 7, 10,
                                              1, 6, 3, 9,
                                              2, 4, 5, 8,
                                              NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), 7, 10});
  test.AddInput<double>("Tensor2", {2, 2, 4}, {2, NS::Traits<float>::CreateNullValue(), 5, 6,
                                              2, NS::Traits<float>::CreateNullValue(), 3, 4,
                                              2, NS::Traits<float>::CreateNullValue(), 5, 6,
                                              2, NS::Traits<float>::CreateNullValue(), 3, 4});
  test.AddOutput<double>("Output", {2, 5}, {3, 5, 7, 5, 3,
                                              9, 8, 10, 6, 4});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
