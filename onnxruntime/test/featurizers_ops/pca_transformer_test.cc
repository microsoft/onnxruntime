// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/PCAFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

template <typename MatrixT>
std::vector<uint8_t> GetStream(const MatrixT& training_batches) {
  using EstimatorT = NS::Featurizers::PCAEstimator<MatrixT>;
  NS::AnnotationMapsPtr const pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  EstimatorT estimator(pAllColumnAnnotations, 0);

  std::vector<std::vector<MatrixT>> trainingBatches = NS::TestHelpers::make_vector<std::vector<MatrixT>>(
      NS::TestHelpers::make_vector<MatrixT>(training_batches));

  NS::TestHelpers::Train<EstimatorT, MatrixT>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

template <typename T>
void PCATransformerTestColMajorStandard() {
  using Matrix = Eigen::Map<NS::RowMajMatrix<T>>;

  // Row major order, for training
  T data[] = {
      -1, -1,
      -2, -1,
      -3, -2,
      1, 1,
      2, 1,
      3, 2};

  Matrix matrix(data, 6, 2);

  auto stream = GetStream<Matrix>(matrix);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("PCATransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<T>("Input", {6, 2}, {-1, -1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2});

  test.AddOutput<T>("Output", {6, 2},
                    {-0.2935787f, -1.3834058f,
                     0.2513348f, -2.2218980f,
                     -0.0422439f, -3.6053038f,
                     0.2935787f, 1.3834058f,
                     -0.2513348f, 2.2218980f,
                     0.0422439f, 3.6053038f});

  test.Run();
}

TEST(FeaturizersTests, PCATransformer_double) {
  PCATransformerTestColMajorStandard<double>();
}

TEST(FeaturizersTests, PCATransformer_float) {
  PCATransformerTestColMajorStandard<float>();
}

}  // namespace test
}  // namespace onnxruntime
