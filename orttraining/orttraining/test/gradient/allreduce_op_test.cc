// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "core/framework/execution_providers.h"
#include "test/util/include/default_providers.h"
#include "core/session/environment.h"
#include "orttraining/models/runner/training_runner.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

namespace onnxruntime {
namespace test {

TEST(AllreduceTest, HorovodCPUAllreduceTest) {
  OpTester allreduce_test("HorovodAllReduce", 9, onnxruntime::kOnnxDomain);
  if (training::MPIContext::GetInstance().GetWorldRank() == 0){
   allreduce_test.AddInput<float>("G", {3}, {4, 5, 6});
  }
  else if(training::MPIContext::GetInstance().GetWorldRank() == 1) {
   allreduce_test.AddInput<float>("G", {3}, {7, 8, 9});     
  }

  allreduce_test.AddOutput<float>("G_new", {3}, {11.f, 13.f, 15.f});
  allreduce_test.AddOutput<bool>("Ready", {}, {true});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());

  allreduce_test.Run(OpTester::ExpectResult::kExpectSuccess/*expect_result*/, ""/*expected_failure_string*/,
                     {}/*excluded_provider_types*/, nullptr/*run_options*/, &providers/*execution_providers*/,
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/, {}/*custom_output_verifier*/,
                     {}/*resolve_options*/);
}

TEST(AllreduceTest, HorovodCPUAdasumAllreduceTest) {
  OpTester allreduce_test("HorovodAllReduce", 9, onnxruntime::kOnnxDomain);
  if (training::MPIContext::GetInstance().GetWorldRank() == 0){
   allreduce_test.AddInput<float>("G", {3}, {4, 5, 6});
  }
  else if(training::MPIContext::GetInstance().GetWorldRank() == 1) {
   allreduce_test.AddInput<float>("G", {3}, {7, 8, 9});
  }

  allreduce_test.AddOutput<float>("G_new", {3}, {5.6301f, 6.5235f, 7.4169f});
  allreduce_test.AddOutput<bool>("Ready", {}, {true});
  allreduce_test.AddAttribute("reduce_op", static_cast<int64_t>(2));

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());

  allreduce_test.Run(OpTester::ExpectResult::kExpectSuccess/*expect_result*/, ""/*expected_failure_string*/,
                     {}/*excluded_provider_types*/, nullptr/*run_options*/, &providers/*execution_providers*/,
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/, {}/*custom_output_verifier*/,
                     {}/*resolve_options*/);
}

TEST(AllreduceTest, CPUAdasumAllreduceTestReduceTwoTensors) {
  OpTester allreduce_test("AdasumAllReduce", 1, onnxruntime::kMSDomain);
  // Alternating inputs to test symmetry
  std::vector<float> grad_1 = {4.0f, 5.0f, 6.0f};
  std::vector<float> grad_2 = {7.0f, 8.0f, 9.0f};
  if (training::MPIContext::GetInstance().GetWorldRank() == 0){
   allreduce_test.AddInput<float>("G1", {3}, grad_1);
   allreduce_test.AddInput<float>("G2", {3}, grad_2);
  }
  else if(training::MPIContext::GetInstance().GetWorldRank() == 1) {
   allreduce_test.AddInput<float>("G1", {3}, grad_2);
   allreduce_test.AddInput<float>("G2", {3}, grad_1);
  }

  std::vector<float> output_grad = {5.6301f, 6.5235f, 7.4169f};

  allreduce_test.AddOutput<float>("G_new1", {3}, output_grad);
  allreduce_test.AddOutput<float>("G_new2", {3}, output_grad);
  allreduce_test.AddAttribute("reduce_algo", static_cast<int64_t>(0));

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());

  allreduce_test.Run(OpTester::ExpectResult::kExpectSuccess/*expect_result*/, ""/*expected_failure_string*/,
                     {}/*excluded_provider_types*/, nullptr/*run_options*/, &providers/*execution_providers*/,
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/, {}/*custom_output_verifier*/,
                     {}/*resolve_options*/);
}

TEST(AllreduceTest, CPUAdasumAllreduceTestReduceTwoTensorsFP16) {
  OpTester allreduce_test("AdasumAllReduce", 1, onnxruntime::kMSDomain);
  // Alternating inputs to test symmetry
  std::vector<float> grad_1 = {5.6301f, 6.5235f, 7.4169f};
  std::vector<float> grad_2 = {7.0f, 8.0f, 9.0f};

  std::vector<MLFloat16> grad_1_half(3);
  std::vector<MLFloat16> grad_2_half(3);

  ConvertFloatToMLFloat16(grad_1.data(), grad_1_half.data(), 3);
  ConvertFloatToMLFloat16(grad_2.data(), grad_2_half.data(), 3);

  if (training::MPIContext::GetInstance().GetWorldRank() == 0){
   allreduce_test.AddInput<MLFloat16>("G1", {3}, grad_1_half);
   allreduce_test.AddInput<MLFloat16>("G2", {3}, grad_2_half);
  }
  else if(training::MPIContext::GetInstance().GetWorldRank() == 1) {
   allreduce_test.AddInput<MLFloat16>("G1", {3}, grad_2_half);
   allreduce_test.AddInput<MLFloat16>("G2", {3}, grad_1_half);
  }

  std::vector<float> output_grad = {6.32478f, 7.2628f, 8.2009f};

  std::vector<MLFloat16> output_grad_half(3);

  ConvertFloatToMLFloat16(output_grad.data(), output_grad_half.data(), 3);

  allreduce_test.AddOutput<MLFloat16>("G_new1", {3}, output_grad_half);
  allreduce_test.AddOutput<MLFloat16>("G_new2", {3}, output_grad_half);

  allreduce_test.AddAttribute("reduce_algo", static_cast<int64_t>(0));

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());

  allreduce_test.Run(OpTester::ExpectResult::kExpectSuccess/*expect_result*/, ""/*expected_failure_string*/,
                     {}/*excluded_provider_types*/, nullptr/*run_options*/, &providers/*execution_providers*/,
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/, {}/*custom_output_verifier*/,
                     {}/*resolve_options*/);
}

TEST(AllreduceTest, CPUAdasumAllreduceTestFailTensorCountMismatch) {
  OpTester allreduce_test("AdasumAllReduce", 1, onnxruntime::kMSDomain);
  if (training::MPIContext::GetInstance().GetWorldRank() == 0){
   allreduce_test.AddInput<float>("G1", {3}, {4, 5, 6});
  }
  else if(training::MPIContext::GetInstance().GetWorldRank() == 1) {
   allreduce_test.AddInput<float>("G1", {3}, {7, 8, 9});
   allreduce_test.AddInput<float>("G2", {3}, {4, 5, 6});
  }

  allreduce_test.AddOutput<float>("G_new1", {3}, {5.6301f, 6.5235f, 7.4169f});
  allreduce_test.AddOutput<float>("G_new2", {3}, {5.6301f, 6.5235f, 7.4169f});
  allreduce_test.AddAttribute("reduce_algo", static_cast<int64_t>(0));

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());

  allreduce_test.Run(OpTester::ExpectResult::kExpectFailure/*expect_result*/, ""/*expected_failure_string*/,
                     {}/*excluded_provider_types*/, nullptr/*run_options*/, &providers/*execution_providers*/,
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/, {}/*custom_output_verifier*/,
                     {}/*resolve_options*/);
}

void build_allreduce_graph(Graph& graph, int64_t reduce_op = 1/*SUM*/) {
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  
  // BOOL tensor.
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  // Input tensor
  auto& allreduce_input_arg = graph.GetOrCreateNodeArg("input_t", &float_tensor);
  inputs.push_back(&allreduce_input_arg);

  // Output tensor and ready signal
  auto& output_arg_1 = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg_1);
  auto& output_arg_ready_tensor = graph.GetOrCreateNodeArg("node_1_out_ready", &bool_tensor);
  outputs.push_back(&output_arg_ready_tensor);

  // Attribute
  ONNX_NAMESPACE::AttributeProto reduce_op_attr;
  reduce_op_attr.set_name("reduce_op");
  reduce_op_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  reduce_op_attr.set_i(reduce_op);

  auto& hvd_allreduce_node =  graph.AddNode("node_1", "HorovodAllReduce", "node 1.", inputs, outputs);
  hvd_allreduce_node.AddAttribute("reduce_op", reduce_op_attr);
  
  inputs.clear();
  inputs.push_back(&output_arg_1);
  inputs.push_back(&output_arg_ready_tensor);
  auto& barrier_output_tensor = graph.GetOrCreateNodeArg("barrier_output_t", &float_tensor);
  auto& barrier_output_ready = graph.GetOrCreateNodeArg("barrier_output_ready", &bool_tensor);
  outputs.clear();
  outputs.push_back(&barrier_output_tensor);
  outputs.push_back(&barrier_output_ready);
  graph.AddNode("node_2", "HorovodBarrier", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
}
#ifdef USE_CUDA
std::unique_ptr<IExecutionProvider> create_cuda_execution_provider() {
  CUDAExecutionProviderInfo info;
  OrtDevice::DeviceId device_id = static_cast<OrtDevice::DeviceId>(training::MPIContext::GetInstance().GetLocalRank());
  size_t cuda_mem_limit = std::numeric_limits<size_t>::max();
  cuda_mem_limit = static_cast<size_t>(1 * 1024 * 1024 * 1024);

  info.device_id = device_id;
  info.cuda_mem_limit = cuda_mem_limit;
  info.arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo;
  return onnxruntime::make_unique<CUDAExecutionProvider>(info);
}

TEST(AllreduceTest, HorovodGPUAdasumAllreduceTest) {
  onnxruntime::Model model("allreduce_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  build_allreduce_graph(graph, 2/*reduce_op*/);
  
  std::string model_file_name = "GPUAdasumAllreduceTest.onnx";
  auto status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "AllreduceTest.HorovodGPUAdasumAllreduceTest";
  
  onnxruntime::InferenceSession session_object{so, GetEnvironment()};
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  
  auto test_cuda_ep = create_cuda_execution_provider();
 
  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = onnxruntime::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(test_cuda_ep)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());
  std::vector<int64_t> dims_allreduce_input = {1, 3};
  std::vector<float> values_allreduce_input;

  if(training::MPIContext::GetInstance().GetWorldRank() == 0) {
    values_allreduce_input.push_back(4.f);
    values_allreduce_input.push_back(5.f);
    values_allreduce_input.push_back(6.f);
  }
  else {
    values_allreduce_input.push_back(7.f);
    values_allreduce_input.push_back(8.f);
    values_allreduce_input.push_back(9.f);

  }
  OrtValue ml_value_input_t;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_allreduce_input, values_allreduce_input, &ml_value_input_t);
  
  NameMLValMap feeds;
  feeds.insert(std::make_pair("input_t", ml_value_input_t));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("barrier_output_t");
  output_names.push_back("barrier_output_ready");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_allreduce = {1, 3};
  std::vector<float> expected_values_allreduce = {11, 13, 15};

  std::vector<int64_t> expected_dims_allreduce_ready = {};
  bool expected_values_allreduce_ready = true;
  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  
  ASSERT_EQ(2u, fetches.size());
  
  // Verify tensor data
  auto& actual_output_tensor = fetches[0].Get<Tensor>();
  TensorShape expected_shape(expected_dims_allreduce);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_tensor.Shape()));

  const std::vector<float> found(actual_output_tensor.template Data<float>(),
                             actual_output_tensor.template Data<float>() + expected_values_allreduce.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_NEAR((double)expected_values_allreduce[i], (double)found[i], 1e-4f);

  // Verify ready tensor
  auto& actual_output_ready_tensor = fetches[1].Get<Tensor>();
  TensorShape expected_ready_shape(expected_dims_allreduce_ready);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_ready_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_ready_tensor.Shape()));

  const bool found_ready = actual_output_ready_tensor.template Data<bool>();
  ASSERT_EQ(expected_values_allreduce_ready, found_ready);
  
}
#endif
}  // namespace test
}  // namespace onnxruntime
