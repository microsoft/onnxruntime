// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <random>
#include "core/framework/tensor.h"
#include "core/providers/cpu/nn/layer_norm_helper.h"
#include "core/session/inference_session.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

template <typename T>
class RMSNormalizationOpTest : public ::testing::Test {
};
using RMSNormalizationOpTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(RMSNormalizationOpTest, RMSNormalizationOpTestTypes);

TYPED_TEST(RMSNormalizationOpTest, RMSNorm) {
// prevents test from running on non-BF16-supporting hardware
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);

  vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> input_dims{1, 2, 3};
  test.AddInput<TypeParam>("X", input_dims, GetTypedArray<TypeParam>(input));

  vector<float> scale = {1.F, 1.F, 1.F};
  vector<int64_t> scale_dims = {6};
  test.AddInput<TypeParam>("scale", scale_dims, GetTypedArray<TypeParam>(scale), true);

  vector<float> expected_output = {0.4629f, 0.9258f, 1.3887f, 0.7895f, 0.9869f, 1.1843f};
  test.AddOutput<TypeParam>("Y", input_dims, GetTypedArray<TypeParam>(expected_output));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
    {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
     kNnapiExecutionProvider, kQnnExecutionProvider});
}

TYPED_TEST(RMSNormalizationOpTest, RMSNorm_Scale) {
// prevents test from running on non-BF16-supporting hardware
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);

  vector<float> input = {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f};
  std::vector<int64_t> input_dims{2, 2, 2};
  test.AddInput<TypeParam>("X", input_dims, GetTypedArray<TypeParam>(input));

  vector<float> scale = {-0.6953f, 5.1824f};
  vector<int64_t> scale_dims = {2};
  test.AddInput<TypeParam>("scale", scale_dims, GetTypedArray<TypeParam>(scale), true);

  vector<float> expected_output = {0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f};
  test.AddOutput<TypeParam>("Y", input_dims, GetTypedArray<TypeParam>(expected_output));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
    {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
     kNnapiExecutionProvider, kQnnExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime