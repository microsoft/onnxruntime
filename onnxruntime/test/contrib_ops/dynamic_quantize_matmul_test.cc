// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

template <typename T>
class DynamicQuantizeMatMulOpTester : public OpTester {
 public:
  DynamicQuantizeMatMulOpTester(const char* op,
                                const std::vector<int64_t>& A_dims,
                                const std::vector<int64_t>& B_dims,
                                const std::vector<int64_t>& Y_dims,
                                string&& model_file,
                                int opset_version = 1,
                                const char* domain = onnxruntime::kMSDomain) : OpTester(op, opset_version, domain),
                                                                               A_dims_(A_dims),
                                                                               B_dims_(B_dims),
                                                                               Y_dims_(Y_dims),
                                                                               model_file_(model_file) {
    Init();
  }
  void Init() {
    // create rand inputs
    RandomValueGenerator random{};
    A_data_ = random.Uniform<float>(A_dims_, -1.0f, 1.0f);

    std::vector<int> tmp_B_data = random.Uniform<int32_t>(B_dims_, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    std::transform(tmp_B_data.begin(), tmp_B_data.end(), std::back_inserter(B_data_), [](int32_t v) -> T {
      return static_cast<T>(v);
    });

    B_zero_point_ = {static_cast<T>(random.Uniform<int32_t>({1}, std::numeric_limits<T>::min(), std::numeric_limits<T>::max())[0])};
    B_scale_ = random.Uniform<float>({1}, -0.1f, 0.1f);

    const int64_t Y_size = std::accumulate(Y_dims_.cbegin(), Y_dims_.cend(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
    Y_data_.resize(Y_size);

    AddInput<float>("A", A_dims_, A_data_);
    AddInput<T>("B", B_dims_, B_data_);
    AddInput<float>("b_scale", {1}, B_scale_);
    AddInput<T>("b_zero_point", {1}, B_zero_point_);

    AddOutput<float>("Y", Y_dims_, Y_data_);
  }
  void Run() {
#ifndef NDEBUG
    // run_called_ to true to avoid a complaining in the destructor of OpTester
    run_called_ = true;
#endif
    std::vector<MLValue> cpu_fetches;
    std::vector<MLValue> subgraph_fetches;
    Compute(cpu_fetches);
    ComputeOriginalSubgraph(subgraph_fetches);

    // Compare CPU with original subgraph on the output(0) - Y
    ASSERT_TRUE(cpu_fetches.size() >= subgraph_fetches.size());
    for (size_t i = 0; i < subgraph_fetches.size(); i++) {
      if (cpu_fetches[i].IsTensor() && subgraph_fetches[i].IsTensor()) {
        VLOGS_DEFAULT(1) << "Checking tensor " << i;
        CheckTensor(subgraph_fetches[i].Get<Tensor>(), cpu_fetches[i].Get<Tensor>(), 1e-3, 1e-3);
      }
    }
  }

 private:
  void Compute(std::vector<MLValue>& cpu_fetches) {
    auto p_model = BuildGraph();
    auto& graph = p_model->MainGraph();

    Status status = graph.Resolve();
    ASSERT_TRUE(status.IsOK()) << status;

    // Hookup the inputs and outputs
    NameMLValMap feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names);

    SessionOptions so;
    so.session_logid = op_;
    so.session_log_verbosity_level = 1;

    RunOptions run_options;
    run_options.run_tag = op_;
    run_options.run_log_verbosity_level = 1;

    // run with DynamicQuantizeMatMul
    InferenceSession session_object{so, GetEnvironment()};
    std::string s1;
    p_model->ToProto().SerializeToString(&s1);
    std::istringstream str(s1);
    ASSERT_TRUE((status = session_object.Load(str)).IsOK()) << status;
    ASSERT_TRUE((status = session_object.Initialize()).IsOK()) << status;
    ASSERT_TRUE((status = session_object.Run(run_options, feeds, output_names, &cpu_fetches)).IsOK());
  }

  void ComputeOriginalSubgraph(std::vector<MLValue>& subgraph_fetches) {
    NameMLValMap feeds;
    OrtValue ml_value;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names);

    SessionOptions so;
    so.session_logid = op_;
    so.session_log_verbosity_level = 1;
    so.graph_optimization_level = TransformerLevel::Level1;

    RunOptions run_options;
    run_options.run_tag = op_;
    run_options.run_log_verbosity_level = 1;

    Status status;
    InferenceSession subgraph_session_object{so, GetEnvironment()};
    ASSERT_TRUE((status = subgraph_session_object.Load(model_file_)).IsOK()) << status;
    ASSERT_TRUE((status = subgraph_session_object.Initialize()).IsOK()) << status;
    ASSERT_TRUE((status = subgraph_session_object.Run(run_options, feeds, output_names, &subgraph_fetches)).IsOK()) << status;
  }

 private:
  std::vector<int64_t> A_dims_;
  std::vector<int64_t> B_dims_;
  std::vector<int64_t> Y_dims_;

  std::vector<float> A_data_;
  std::vector<T> B_data_;
  std::vector<float> B_scale_;
  std::vector<T> B_zero_point_;
  std::vector<float> Y_data_;

  std::string model_file_;
};

TEST(DynamicQuantizeMatMul, Int8_test) {
#ifdef MLAS_SUPPORTS_GEMM_U8X8
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  std::vector<int64_t> Y_dims{4, 128};
  DynamicQuantizeMatMulOpTester<int8_t> test("DynamicQuantizeMatMul",
                                             A_dims,
                                             B_dims,
                                             Y_dims,
                                             "testdata/dynamic_quantize_matmul_int8.onnx");
  test.Run();
#endif
}

TEST(DynamicQuantizeMatMul, UInt8_test) {
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  std::vector<int64_t> Y_dims{4, 128};
  DynamicQuantizeMatMulOpTester<uint8_t> test("DynamicQuantizeMatMul",
                                              A_dims,
                                              B_dims,
                                              Y_dims,
                                              "testdata/dynamic_quantize_matmul_uint8.onnx");
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
