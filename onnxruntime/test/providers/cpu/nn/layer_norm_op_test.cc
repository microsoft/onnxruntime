// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <random>
#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

static void CheckTensor(const Tensor& expected_tensor, const Tensor& output_tensor, double rtol, double atol) {
  ORT_ENFORCE(expected_tensor.Shape() == output_tensor.Shape(),
              "Expected output shape [" + expected_tensor.Shape().ToString() +
                  "] did not match run output shape [" +
                  output_tensor.Shape().ToString() + "]");

  ASSERT_TRUE(expected_tensor.DataType() == DataTypeImpl::GetType<float>()) << "Compare with non float number is not supported yet. ";
  auto expected = expected_tensor.Data<float>();
  auto output = output_tensor.Data<float>();
  for (auto i = 0; i < expected_tensor.Shape().Size(); ++i) {
    const auto expected_value = expected[i], actual_value = output[i];
    // TODO enable these checks for non-finite values
    /*if (std::isnan(expected_value)) {
      ASSERT_TRUE(std::isnan(actual_value)) << "value mismatch at index " << i << "; expected is NaN, actual is not NaN";
    } else if (std::isinf(expected_value)) {
      ASSERT_EQ(expected_value, actual_value) << "value mismatch at index " << i;
    } else*/
    {
      double diff = fabs(expected_value - actual_value);
      ASSERT_TRUE(diff <= (atol + rtol * fabs(expected_value))) << "value mismatch at index " << i << "; expected: " << expected_value << ", actual: " << actual_value;
    }
  }
}

class LayerNormOpTester : public OpTester {
 public:
  LayerNormOpTester(const char* op, int opset_version = 9, const char* domain = onnxruntime::kOnnxDomain) : OpTester(op, opset_version, domain) {}
  void CompareWithCPU(double rtol, double atol);
};

void LayerNormOpTester::CompareWithCPU(double rtol, double atol) {
#ifndef NDEBUG
  run_called_ = true;
#endif
  if (DefaultCudaExecutionProvider() == nullptr) {
    return;
  }
  auto p_model = BuildGraph();
  auto& graph = p_model->MainGraph();

  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    return;
  }

  // Hookup the inputs and outputs
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  // Run the model
  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  InferenceSession cpu_session_object{so};

  // first run with cpu
  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::istringstream str(s1);
  status = cpu_session_object.Load(str);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = cpu_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;

  std::vector<MLValue> cpu_fetches;
  status = cpu_session_object.Run(run_options, feeds, output_names, &cpu_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
    return;
  }

  // run with CUDA
  auto cuda_execution_provider = DefaultCudaExecutionProvider();

  InferenceSession cuda_session_object{so};
  EXPECT_TRUE(cuda_session_object.RegisterExecutionProvider(std::move(cuda_execution_provider)).IsOK());

  auto p_model_2 = BuildGraph();
  auto& graph_2 = p_model_2->MainGraph();

  status = graph_2.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    return;
  }

  std::string s2;
  p_model_2->ToProto().SerializeToString(&s2);
  std::istringstream str2(s2);

  status = cuda_session_object.Load(str2);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = cuda_session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Initialize failed with status: " << status.ErrorMessage();
    return;
  }

  std::vector<MLValue> cuda_fetches;
  status = cuda_session_object.Run(run_options, feeds, output_names, &cuda_fetches);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  //compare
  ASSERT_TRUE(cpu_fetches.size() == cuda_fetches.size());
  for (auto i = 0; i < cpu_fetches.size(); i++) {
    if (cpu_fetches[i].IsTensor() && cuda_fetches[i].IsTensor()) {
      VLOGS_DEFAULT(1) << "Checking tensor " << i;
      CheckTensor(cpu_fetches[i].Get<Tensor>(), cuda_fetches[i].Get<Tensor>(), rtol, atol);
    }
  }
}

static void TestLayerNorm(const std::vector<int64_t>& X_dims,
                   const std::vector<int64_t>& scale_dims,
                   const std::vector<int64_t>& B_dims,
                   const std::vector<int64_t>& Y_dims,
                   const std::vector<int64_t>& mean_dims,
                   const std::vector<int64_t>& var_dims,
                   optional<float> epsilon,
                   int64_t axis = -1,
                   int64_t keep_dims = 1) {
  LayerNormOpTester test("LayerNormalization");
  test.AddAttribute("axis", axis);
  test.AddAttribute("keep_dims", keep_dims);
  if (epsilon.has_value()) {
    test.AddAttribute("epsilon", epsilon.value());
  }

  int64_t X_size = std::accumulate(X_dims.cbegin(), X_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t scale_size = std::accumulate(scale_dims.cbegin(), scale_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t B_size = std::accumulate(B_dims.cbegin(), B_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t Y_size = std::accumulate(Y_dims.cbegin(), Y_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t mean_size = std::accumulate(mean_dims.cbegin(), mean_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  int64_t var_size = std::accumulate(var_dims.cbegin(), var_dims.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});

  // create rand inputs
  std::vector<float> X_data(X_size, 1.0f);
  std::vector<float> scale_data(scale_size, 1.0f);
  std::vector<float> B_data(B_size, 2.0f);
  std::vector<float> Y_data(Y_size);
  std::vector<float> mean_data(mean_size);
  std::vector<float> var_data(var_size);

  FillRandom<float>(X_data, 0.0f, 1.0f);
  FillRandom<float>(scale_data, 0.0f, 1.0f);
  FillRandom<float>(B_data, 0.0f, 1.0f);

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("scale", scale_dims, scale_data, true);
  test.AddInput<float>("B", B_dims, B_data, true);

  test.AddOutput<float>("output", Y_dims, Y_data);
  test.AddOutput<float>("mean", mean_dims, mean_data);
  test.AddOutput<float>("var", var_dims, var_data);

  test.CompareWithCPU(1e-3, 1e-3);
}

TEST(LayerNormTest, BERTLayerNorm) {
  float epsilon = 1e-05f;
  std::vector<int64_t> X_dims{4, 512, 128};
  std::vector<int64_t> scale_dims{128};
  std::vector<int64_t> B_dims{128};
  std::vector<int64_t> Y_dims{4, 512, 128};
  std::vector<int64_t> mean_dims{4, 512, 1};
  std::vector<int64_t> var_dims{4, 512, 1};
  TestLayerNorm(X_dims, scale_dims, B_dims, Y_dims, mean_dims, var_dims, epsilon);
}
}
}  // namespace onnxruntime
