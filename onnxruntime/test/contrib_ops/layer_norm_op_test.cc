// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <random>
#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
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
    if (std::isnan(expected_value)) {
      ASSERT_TRUE(std::isnan(actual_value)) << "value mismatch at index " << i << "; expected is NaN, actual is not NaN";
    } else if (std::isinf(expected_value)) {
      ASSERT_EQ(expected_value, actual_value) << "value mismatch at index " << i;
    } else {
      double diff = fabs(expected_value - actual_value);
      ASSERT_TRUE(diff <= (atol + rtol * fabs(expected_value))) << "value mismatch at index " << i << "; expected: " << expected_value << ", actual: " << actual_value;
    }
  }
}

class LayerNormOpTester : public OpTester {
 public:
  LayerNormOpTester(const char* op,
                    const std::vector<int64_t>& X_dims,
                    const std::vector<int64_t>& scale_dims,
                    const std::vector<int64_t>& B_dims,
                    const std::vector<int64_t>& Y_dims,
                    float epsilon,
                    int64_t axis = -1,
                    int64_t keep_dims = 1,
                    int opset_version = 1,
                    const char* domain = onnxruntime::kOnnxDomain) : OpTester(op, opset_version, domain),
                                                                     X_dims_(X_dims),
                                                                     scale_dims_(scale_dims),
                                                                     B_dims_(B_dims),
                                                                     Y_dims_(Y_dims),
                                                                     epsilon_(epsilon),
                                                                     axis_(axis),
                                                                     keep_dims_(keep_dims) {
    Init();
  }
  void Init() {
    AddAttribute("axis", axis_);
    AddAttribute("keep_dims", keep_dims_);
    AddAttribute("epsilon", epsilon_);

    int64_t X_size = std::accumulate(X_dims_.cbegin(), X_dims_.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    int64_t scale_size = std::accumulate(scale_dims_.cbegin(), scale_dims_.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    int64_t B_size = std::accumulate(B_dims_.cbegin(), B_dims_.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    int64_t Y_size = std::accumulate(Y_dims_.cbegin(), Y_dims_.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});

    // create rand inputs
    X_data_.resize(X_size, 1.f);
    scale_data_.resize(scale_size, 1.f);
    B_data_.resize(B_size, 2.f);
    Y_data_.resize(Y_size);

    FillRandom<float>(X_data_, 0.0f, 1.0f);
    FillRandom<float>(scale_data_, 0.0f, 1.0f);
    FillRandom<float>(B_data_, 0.0f, 1.0f);

    AddInput<float>("X", X_dims_, X_data_);
    AddInput<float>("scale", scale_dims_, scale_data_, true);
    AddInput<float>("B", B_dims_, B_data_, true);

    AddOutput<float>("output", Y_dims_, Y_data_);
  }
  void Run() {
#ifndef NDEBUG
    run_called_ = true;
#endif
    std::vector<MLValue> cpu_fetches;
    std::vector<MLValue> cuda_fetches;
    std::vector<MLValue> subgraph_fetches;
    ComputeWithCPU(cpu_fetches);
    ComputeWithCUDA(cuda_fetches);
    ComputeOriSubgraphWithCPU(subgraph_fetches);

    // Compare CPU with original subgraph
    ASSERT_TRUE(cpu_fetches.size() == subgraph_fetches.size());
    for (size_t i = 0; i < cpu_fetches.size(); i++) {
      if (cpu_fetches[i].IsTensor() && subgraph_fetches[i].IsTensor()) {
        VLOGS_DEFAULT(1) << "Checking tensor " << i;
        CheckTensor(subgraph_fetches[i].Get<Tensor>(), cpu_fetches[i].Get<Tensor>(), 1e-3, 1e-3);
      }
    }

    // Compare GPU with original subgraph
    if (DefaultCudaExecutionProvider()) {
      ASSERT_TRUE(cuda_fetches.size() == subgraph_fetches.size());
      for (size_t i = 0; i < cuda_fetches.size(); i++) {
        if (cuda_fetches[i].IsTensor() && subgraph_fetches[i].IsTensor()) {
          VLOGS_DEFAULT(1) << "Checking tensor " << i;
          CheckTensor(subgraph_fetches[i].Get<Tensor>(), cuda_fetches[i].Get<Tensor>(), 1e-3, 1e-3);
        }
      }
    }
  }

 private:
  void ComputeWithCPU(std::vector<MLValue>& cpu_fetches);
  void ComputeWithCUDA(std::vector<MLValue>& cuda_fetches);
  void ComputeOriSubgraphWithCPU(std::vector<MLValue>& subgraph_fetches);

 private:
  std::vector<int64_t> X_dims_;
  std::vector<int64_t> scale_dims_;
  std::vector<int64_t> B_dims_;
  std::vector<int64_t> Y_dims_;

  std::vector<float> X_data_;
  std::vector<float> scale_data_;
  std::vector<float> B_data_;
  std::vector<float> Y_data_;

  float epsilon_;
  int64_t axis_;
  int64_t keep_dims_;
};

void LayerNormOpTester::ComputeWithCPU(std::vector<MLValue>& cpu_fetches) {
  auto p_model = BuildGraph();
  auto& graph = p_model->MainGraph();

  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status;

  // Hookup the inputs and outputs
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;

  // run with LayerNormalization
  InferenceSession layernorm_session_object{so};
  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::istringstream str(s1);
  ASSERT_TRUE((status = layernorm_session_object.Load(str)).IsOK()) << status;
  ASSERT_TRUE((status = layernorm_session_object.Initialize()).IsOK()) << status;
  ASSERT_TRUE((status = layernorm_session_object.Run(run_options, feeds, output_names, &cpu_fetches)).IsOK());
}

void LayerNormOpTester::ComputeWithCUDA(std::vector<MLValue>& cuda_fetches) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    return;
  }

  auto p_model = BuildGraph();
  auto& graph = p_model->MainGraph();

  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status;

  // Hookup the inputs and outputs
  std::unordered_map<std::string, MLValue> feeds;
  std::vector<std::string> output_names;
  FillFeedsAndOutputNames(feeds, output_names);

  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;

  auto cuda_execution_provider = DefaultCudaExecutionProvider();
  InferenceSession cuda_session_object{so};
  EXPECT_TRUE(cuda_session_object.RegisterExecutionProvider(std::move(cuda_execution_provider)).IsOK());

  std::string s;
  p_model->ToProto().SerializeToString(&s);
  std::istringstream str2(s);

  EXPECT_TRUE((status = cuda_session_object.Load(str2)).IsOK()) << status;
  EXPECT_TRUE((status = cuda_session_object.Initialize()).IsOK()) << status;
  EXPECT_TRUE((status = cuda_session_object.Run(run_options, feeds, output_names, &cuda_fetches)).IsOK()) << status;
}

void LayerNormOpTester::ComputeOriSubgraphWithCPU(std::vector<MLValue>& subgraph_fetches) {
  NameMLValMap feeds;
  OrtValue ml_value;
  std::vector<std::string> output_names{"Y"};

  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), X_dims_, X_data_, &ml_value);
  feeds.insert(std::make_pair("X", ml_value));
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), scale_dims_, scale_data_, &ml_value);
  feeds.insert(std::make_pair("Scale", ml_value));
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), B_dims_, B_data_, &ml_value);
  feeds.insert(std::make_pair("B", ml_value));

  SessionOptions so;
  so.session_logid = op_;
  so.session_log_verbosity_level = 1;

  RunOptions run_options;
  run_options.run_tag = op_;
  run_options.run_log_verbosity_level = 1;

  Status status;
  InferenceSession subgraph_session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE((status = subgraph_session_object.Load("testdata/layernorm.onnx")).IsOK()) << status;
  ASSERT_TRUE((status = subgraph_session_object.Initialize()).IsOK()) << status;
  ASSERT_TRUE((status = subgraph_session_object.Run(run_options, feeds, output_names, &subgraph_fetches)).IsOK()) << status;
}

TEST(LayerNormTest, BERTLayerNorm) {
  float epsilon = 1e-12f;
  std::vector<int64_t> X_dims{4, 128};
  std::vector<int64_t> scale_dims{128};
  std::vector<int64_t> B_dims{128};
  std::vector<int64_t> Y_dims{4, 128};
  LayerNormOpTester test("LayerNormalization", X_dims, scale_dims, B_dims, Y_dims, epsilon, -1, 1);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
