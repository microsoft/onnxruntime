#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

// All tests in this file are for the CPU provider and
// CUDA provider

TEST(RMSNormalizationOpTest, RMSNorm) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{1, 2, 3};
  test.AddInput<float>("X", input_dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  vector<int64_t> scale_dims = {3};
  test.AddInput<float>("scale", scale_dims, {1.F, 1.F, 1.F});
  test.AddOutput<float>("Y", input_dims, {0.4629f, 0.9258f, 1.3887f, 0.7895f, 0.9869f, 1.1843f});
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_float16) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{1, 2, 3};
  test.AddInput<MLFloat16>("X", input_dims, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  vector<int64_t> scale_dims = {3};
  test.AddInput<MLFloat16>("scale", scale_dims, ToFloat16({1.F, 1.F, 1.F}));
  test.AddOutput<MLFloat16>("Y", input_dims, ToFloat16({0.4629f, 0.9258f, 1.3887f, 0.7895f, 0.9869f, 1.1843f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{2, 2, 2};
  test.AddInput<float>("X", input_dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  vector<int64_t> scale_dims = {2};
  test.AddInput<float>("scale", scale_dims, {-0.6953f, 5.1824f});
  test.AddOutput<float>("Y", input_dims, {0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f});
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Float16) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{2, 2, 2};
  test.AddInput<MLFloat16>("X", input_dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  vector<int64_t> scale_dims = {2};
  test.AddInput<MLFloat16>("scale", scale_dims, ToFloat16({-0.6953f, 5.1824f}));
  test.AddOutput<MLFloat16>("Y", input_dims, ToFloat16({0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
