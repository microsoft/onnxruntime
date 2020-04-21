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
  using MatrixT = NS::RowMajMatrix<typename NS::Traits<T>::nullable_type>;
  using InputType = std::vector<Eigen::Map<MatrixT>>;
  NS::Featurizers::ForecastingPivotTransformer<std::tuple<typename InputType::iterator, typename InputType::iterator>> transformer;
  NS::Archive ar;
  transformer.save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, ForecastingPivotTransformer_2_Inputs) {
  auto stream = GetStream<float>();
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ForecastingPivotTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddAttribute<int64_t>("num_pivot_columns", static_cast<int64_t>(2));
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<double>("Input_1", {2, 3, 4}, {1, 6, 3, 9,
                                              2, 4, 5, 8,
                                              NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), 7, 10,
                                              1, 6, 3, 9,
                                              2, 4, 5, 8,
                                              NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), 7, 10});
  test.AddInput<double>("Input_2", {2, 2, 4}, {2, NS::Traits<float>::CreateNullValue(), 5, 6,
                                              2, NS::Traits<float>::CreateNullValue(), 3, 4,
                                              2, NS::Traits<float>::CreateNullValue(), 5, 6,
                                              2, NS::Traits<float>::CreateNullValue(), 3, 4});
  test.AddOutput<double>("Output_1", {4, 1}, {3, 9, 3, 9});
  test.AddOutput<double>("Output_2", {4, 1}, {5, 8, 5, 8});
  test.AddOutput<double>("Output_3", {4, 1}, {7, 10, 7, 10});
  test.AddOutput<double>("Output_4", {4, 1}, {5, 6, 5, 6});
  test.AddOutput<double>("Output_5", {4, 1}, {3, 4, 3, 4});
  //horizon output
  test.AddOutput<uint32_t>("Output_6", {4, 1}, {2, 1, 2, 1});

  test.Run();
}

TEST(FeaturizersTests, ForecastingPivotTransformer_4_Inputs) {
  auto stream = GetStream<float>();
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("ForecastingPivotTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddAttribute<int64_t>("num_pivot_columns", static_cast<int64_t>(2));
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<double>("Input_1", {2, 3, 4}, {1, 6, 3, 9,
                                              2, 4, 5, 8,
                                              NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), 7, 10,
                                              1, 6, 3, 9,
                                              2, 4, 5, 8,
                                              NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), 7, 10});
  test.AddInput<double>("Input_2", {2, 2, 4}, {2, NS::Traits<float>::CreateNullValue(), 5, 6,
                                              2, NS::Traits<float>::CreateNullValue(), 3, 4,
                                              2, NS::Traits<float>::CreateNullValue(), 5, 6,
                                              2, NS::Traits<float>::CreateNullValue(), 3, 4});

  test.AddInput<std::string>("Input_3", {2, 1, 4}, {"7", "7", "7", "7",
                                               "9", "9", "9", "9"});
  test.AddInput<double>("Input_4", {2, 4}, {-7, -7, -7, -7,
                                               -9, -9, -9, -9});

  //pivot output
  test.AddOutput<double>("Output_1", {4, 1}, {3, 9, 3, 9});
  test.AddOutput<double>("Output_2", {4, 1}, {5, 8, 5, 8});
  test.AddOutput<double>("Output_3", {4, 1}, {7, 10, 7, 10});
  test.AddOutput<double>("Output_4", {4, 1}, {5, 6, 5, 6});
  test.AddOutput<double>("Output_5", {4, 1}, {3, 4, 3, 4});

  //non-pivot output
  test.AddOutput<std::string>("Output_6", {4, 4}, {"7", "7", "7", "7",
                                                   "7", "7", "7", "7",
                                                   "9", "9", "9", "9",
                                                   "9", "9", "9", "9"});
  test.AddOutput<double>("Output_7", {4, 4}, {-7, -7, -7, -7,
                                              -7, -7, -7, -7,
                                              -9, -9, -9, -9,
                                              -9, -9, -9, -9});
  //horizon output
  test.AddOutput<uint32_t>("Output_8", {4, 1}, {2, 1, 2, 1});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
