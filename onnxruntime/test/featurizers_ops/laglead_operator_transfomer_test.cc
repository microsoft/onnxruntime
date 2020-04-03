// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/../Archive.h"
#include "Featurizers/LagLeadOperatorFeaturizer.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {
namespace {

template<typename EstimatorT, typename InputTupleType>
std::vector<uint8_t> GetStream(EstimatorT& estimator, const std::vector<InputTupleType>& trainingBatches) {
  NS::TestHelpers::Train<EstimatorT, InputTupleType>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, Grained_LagLead_2_grain_horizon_2_lead_1_lead_2_long) {
  using InputType = float;
  using GrainType = std::vector<std::string>;
  NS::AnnotationMapsPtr                                            pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  NS::Featurizers::GrainedLagLeadOperatorEstimator<InputType>      estimator(pAllColumnAnnotations, 2, {1, 2});

  using GrainedInputType = typename NS::Featurizers::GrainedLagLeadOperatorEstimator<InputType>::InputType;

  const GrainType grain1({"one"});
  const InputType value1(static_cast<InputType>(10));
  const InputType value2(static_cast<InputType>(11));
  const InputType value3(static_cast<InputType>(12));
  const GrainedInputType tup1(grain1, value1);
  const GrainedInputType tup2(grain1, value2);
  const GrainedInputType tup3(grain1, value3);

  const GrainType grain2({"two"});
  const InputType value4(static_cast<InputType>(20));
  const InputType value5(static_cast<InputType>(21));
  const InputType value6(static_cast<InputType>(22));
  const GrainedInputType tup4(grain2, value4);
  const GrainedInputType tup5(grain2, value5);
  const GrainedInputType tup6(grain2, value6);

  auto training_batch = NS::TestHelpers::make_vector<std::tuple<GrainType const &, InputType const &>>(tup1, tup2, tup3, tup4, tup5, tup6);

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("LagLeadOperatorTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {6, 1}, {"one", "one", "one", "two", "two", "two"});
  test.AddInput<float>("Target", {6}, {10, 11, 12, 20, 21, 22});
  test.AddOutput<std::string>("OutputGrains", {6, 1}, {"one", "two", "one", "one", "two", "two"});
  test.AddOutput<float>("Output", {6, 2, 2}, {10, 11, 11, 12,
                                               20, 21, 21, 22,
                                               11, 12, 12, NS::Traits<float>::CreateNullValue(),
                                               12, NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(),
                                               21, 22, 22, NS::Traits<float>::CreateNullValue(),
                                               22, NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue(), NS::Traits<float>::CreateNullValue()});

  test.Run();
}

TEST(FeaturizersTests, Grained_LagLead_2_grain_horizon_2_lead_1_lead_2_double) {
  using InputType = double;
  using GrainType = std::vector<std::string>;
  NS::AnnotationMapsPtr                                            pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  NS::Featurizers::GrainedLagLeadOperatorEstimator<InputType>      estimator(pAllColumnAnnotations, 2, {1, 2});

  using GrainedInputType = typename NS::Featurizers::GrainedLagLeadOperatorEstimator<InputType>::InputType;

  const GrainType grain1({"one"});
  const InputType value1(static_cast<InputType>(10));
  const InputType value2(static_cast<InputType>(11));
  const InputType value3(static_cast<InputType>(12));
  const GrainedInputType tup1(grain1, value1);
  const GrainedInputType tup2(grain1, value2);
  const GrainedInputType tup3(grain1, value3);

  const GrainType grain2({"two"});
  const InputType value4(static_cast<InputType>(20));
  const InputType value5(static_cast<InputType>(21));
  const InputType value6(static_cast<InputType>(22));
  const GrainedInputType tup4(grain2, value4);
  const GrainedInputType tup5(grain2, value5);
  const GrainedInputType tup6(grain2, value6);

  auto training_batch = NS::TestHelpers::make_vector<std::tuple<GrainType const &, InputType const &>>(tup1, tup2, tup3, tup4, tup5, tup6);

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("LagLeadOperatorTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {6, 1}, {"one", "one", "one", "two", "two", "two"});
  test.AddInput<double>("Target", {6}, {10, 11, 12, 20, 21, 22});
  test.AddOutput<std::string>("OutputGrains", {6, 1}, {"one", "two", "one", "one", "two", "two"});
  test.AddOutput<double>("Output", {6, 2, 2}, {10, 11, 11, 12,
                                               20, 21, 21, 22,
                                               11, 12, 12, NS::Traits<double>::CreateNullValue(),
                                               12, NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(),
                                               21, 22, 22, NS::Traits<double>::CreateNullValue(),
                                               22, NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue()});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
