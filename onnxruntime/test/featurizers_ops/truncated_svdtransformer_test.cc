// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/TruncatedSVDFeaturizer.h"
#include "Featurizers/TestHelpers.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {

template <typename MatrixT>
std::vector<uint8_t> GetStream(const MatrixT& training_matrix) {
  using EstimatorT = NS::Featurizers::TruncatedSVDEstimator<MatrixT>;
  NS::AnnotationMapsPtr const pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  EstimatorT estimator(pAllColumnAnnotations, 0);

  std::vector<std::vector<MatrixT>> trainingBatches = NS::TestHelpers::make_vector<std::vector<MatrixT>>(
      NS::TestHelpers::make_vector<MatrixT>(training_matrix));

  NS::TestHelpers::Train<EstimatorT, MatrixT>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

template <typename T>
void TruncatedSVDTransformerTestRowMajStandard() {
  using Matrix = Eigen::Map<const NS::RowMajMatrix<T>>;

  // Row major order
  const T data[] = {
      -1, -1, 0,
      0, -2, -1,
      -3, 0, -2};

  Matrix matrix(data, 3, 3);
  auto stream = GetStream<Matrix>(matrix);
  auto dim = static_cast<int64_t>(stream.size());

  OpTester test("TruncatedSVDTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<T>("Input", {3, 3}, {-1, -1, 0, 0, -2, -1, -3, 0, -2});
  test.AddOutput<T>("Output", {3, 3}, {1.009107f, 0.626315f, 0.767745f, 0.965105f, 1.995869f, -0.291682f, 3.529165f, -0.724887f, -0.139759f});
  test.Run();
}

TEST(FeaturizersTests, TruncatedSVDTransformer_double) {
  TruncatedSVDTransformerTestRowMajStandard<double>();
}

TEST(FeaturizersTests, TruncatedSVDTransformer_float) {
  TruncatedSVDTransformerTestRowMajStandard<float>();
}

}  // namespace test
}  // namespace onnxruntime