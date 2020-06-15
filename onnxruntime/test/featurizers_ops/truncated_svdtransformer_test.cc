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
  //Hardcode the seed = 42
  EstimatorT estimator(pAllColumnAnnotations, 0, static_cast<unsigned int>(42));

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
  std::vector<T> output = {-1.009107f, 0.626315f, -0.767745f, -0.965105f, 1.995869f, 0.291682f, -3.529165f, -0.724887f, 0.139759f};
  test.AddOutput<T>("Output", {3, 3},
                    output);

  Matrix verify_matrix(output.data(), 3, 3);

  // Custom verification function is necessary since the matrix output will vary from
  // platform to platform enough so we choose to check max STD deviation.
  OpTester::CustomOutputVerifierFn ver_fn = [&verify_matrix](const std::vector<OrtValue>& fetches, const std::string& provider) {
    std::cout << "Verifying TruncatedSVDTransformerTestRowMajStandard:" << provider << std::endl;
    const float eps = 0.0001f;
    ASSERT_TRUE(fetches.size() == 1);
    const auto& fetch = fetches.at(0);
    const auto& tensor = fetch.Get<Tensor>();
    ASSERT_EQ(tensor.Shape().NumDimensions(), 2);
    ASSERT_EQ(tensor.Shape().Size(), 9);
    Matrix output_matrix(tensor.Data<T>(), 3, 3);
    ASSERT_LT((output_matrix.cwiseProduct(output_matrix) - verify_matrix.cwiseProduct(verify_matrix)).norm(), eps);
  };

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNupharExecutionProvider}, nullptr, {}, ORT_SEQUENTIAL, ver_fn);
}

TEST(FeaturizersTests, TruncatedSVDTransformer_double) {
  TruncatedSVDTransformerTestRowMajStandard<double>();
}

TEST(FeaturizersTests, TruncatedSVDTransformer_float) {
  TruncatedSVDTransformerTestRowMajStandard<float>();
}

}  // namespace test
}  // namespace onnxruntime