// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef USE_MPI
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

TEST(AllreduceTest, CPUAdasumAllreduceTestReduceTwoTensors) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

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
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/,
                     {}/*resolve_options*/);
}

TEST(AllreduceTest, CPUAdasumAllreduceTestReduceTwoTensorsFP16) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }
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
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/,
                     {}/*resolve_options*/);
}

TEST(AllreduceTest, CPUAdasumAllreduceTestFailTensorCountMismatch) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

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
                     ExecutionMode::ORT_SEQUENTIAL/*execution_mode*/,
                     {}/*resolve_options*/);
}

const std::string lr_string = "ETA";
const std::string grad_divisor_string = "grad_divisor";
const std::string update_count_string = "Update_Count";
const std::string weight_suffix_string = "_W";
const std::string m1_suffix_string = "_Moment_1";
const std::string m2_suffix_string = "_Moment_2";
const std::string update_count_out_string = "_Update_Count_Out";
const std::string m1_out_suffix_string = "_Moment_1_Out";
const std::string m2_out_suffix_string = "_Moment_2_Out";
const std::string weight_out_suffix_string = "_W_Out";
const std::string gradient_out_suffix_string = "_optimizer_Out";
const std::string allreduce_output_suffix_string = "_allreduce_out";

void build_optimizer_node(Graph& graph,
                          onnx::TensorProto_DataType element_type,
                          int num_of_elements,
                          std::string& original_input_name,
                          onnxruntime::NodeArg* input_gradient,
                          onnxruntime::NodeArg* output_arg,
                          training::AdasumReductionType adasum_reduce_type) {
  std::vector<onnxruntime::NodeArg*> optimizer_inputs;
  std::vector<onnxruntime::NodeArg*> optimizer_outputs;
  // learning rate input
  ONNX_NAMESPACE::TypeProto eta_tensor;
  auto eta_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  eta_tensor.mutable_tensor_type()->set_elem_type(eta_type);
  eta_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& eta_input_arg = graph.GetOrCreateNodeArg(lr_string, &eta_tensor);
  optimizer_inputs.push_back(&eta_input_arg);
  // Update count input
  ONNX_NAMESPACE::TypeProto uc_tensor;
  auto uc_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  uc_tensor.mutable_tensor_type()->set_elem_type(uc_type);
  uc_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& uc_input_arg = graph.GetOrCreateNodeArg(update_count_string, &uc_tensor);
  optimizer_inputs.push_back(&uc_input_arg);
  // Weight input
  ONNX_NAMESPACE::TypeProto weight_tensor;
  weight_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  weight_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_of_elements);
  auto& weight_input_arg = graph.GetOrCreateNodeArg(original_input_name + weight_suffix_string, &weight_tensor);
  optimizer_inputs.push_back(&weight_input_arg);
  // Gradient input
  optimizer_inputs.push_back(input_gradient);
  // Moment1 input
  ONNX_NAMESPACE::TypeProto m1_tensor;
  m1_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  m1_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_of_elements);
  auto& m1_input_arg = graph.GetOrCreateNodeArg(original_input_name + m1_suffix_string, &m1_tensor);
  optimizer_inputs.push_back(&m1_input_arg);
  // Moment2 input
  ONNX_NAMESPACE::TypeProto m2_tensor;
  m2_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  m2_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_of_elements);
  auto& m2_input_arg = graph.GetOrCreateNodeArg(original_input_name + m2_suffix_string, &m2_tensor);
  optimizer_inputs.push_back(&m2_input_arg);

  // Update Count output
  ONNX_NAMESPACE::TypeProto uc_out_tensor;
  uc_out_tensor.mutable_tensor_type()->set_elem_type(uc_type);
  uc_out_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& uc_output_arg = graph.GetOrCreateNodeArg(input_gradient->Name() + update_count_out_string, &uc_out_tensor);
  optimizer_outputs.push_back(&uc_output_arg);

  // Moment 1 output
  ONNX_NAMESPACE::TypeProto m1_out_tensor;
  m1_out_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  m1_out_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_of_elements);
  auto& m1_output_arg = graph.GetOrCreateNodeArg(original_input_name + m1_out_suffix_string, &m1_out_tensor);
  optimizer_outputs.push_back(&m1_output_arg);

  // Moment 1 output
  ONNX_NAMESPACE::TypeProto m2_out_tensor;
  m2_out_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  m2_out_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_of_elements);
  auto& m2_output_arg = graph.GetOrCreateNodeArg(original_input_name + m2_out_suffix_string, &m2_out_tensor);
  optimizer_outputs.push_back(&m2_output_arg);

  // Weight and gradient output
  if (adasum_reduce_type == training::AdasumReductionType::None) {
    ONNX_NAMESPACE::TypeProto weight_out_tensor;
    weight_out_tensor.mutable_tensor_type()->set_elem_type(element_type);
    weight_out_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_of_elements);
    auto& weight_out_arg = graph.GetOrCreateNodeArg(output_arg->Name(), &weight_out_tensor);
    optimizer_outputs.push_back(&weight_out_arg);

    auto& gradient_out_arg = graph.GetOrCreateNodeArg("", nullptr);
    optimizer_outputs.push_back(&gradient_out_arg);
  }
  else {
    auto& weight_out_arg = graph.GetOrCreateNodeArg("", nullptr);
    optimizer_outputs.push_back(&weight_out_arg);
    ONNX_NAMESPACE::TypeProto gradient_out_tensor;
    gradient_out_tensor.mutable_tensor_type()->set_elem_type(element_type);
    gradient_out_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(num_of_elements);
    auto& gradient_out_arg = graph.GetOrCreateNodeArg(output_arg->Name(), &gradient_out_tensor);
    optimizer_outputs.push_back(&gradient_out_arg);
  }

  auto& optimizer_node =  graph.AddNode(input_gradient->Name() + "_adam_optimizer", "AdamOptimizer", "Adam optimizer.", optimizer_inputs, optimizer_outputs,
                                    nullptr/*attributes*/, kMSDomain);

  ONNX_NAMESPACE::AttributeProto bias_correction_attribute, weight_decay_mode_attribute;

  bias_correction_attribute.set_name("do_bias_correction");
  bias_correction_attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  bias_correction_attribute.set_i(0);

  weight_decay_mode_attribute.set_name("weight_decay_mode");
  weight_decay_mode_attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  weight_decay_mode_attribute.set_i(0);

  optimizer_node.AddAttribute("do_bias_correction", bias_correction_attribute);
  optimizer_node.AddAttribute("weight_decay_mode", weight_decay_mode_attribute);
}

using AllreduceGraphConfigVector = std::vector<std::tuple<std::string/*input name*/,
                                                          std::string/*output name*/,
                                                          int/*number of elements*/>>;

void build_allreduce_graph(Graph& graph, AllreduceGraphConfigVector& config,
                           training::AdasumReductionType adasum_reduce_type = training::AdasumReductionType::None,
                           bool build_optimizer=false,
                           bool half_precision=false) {

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  auto element_type = half_precision ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 :
                                      ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

  // Tensor proto.
  ONNX_NAMESPACE::TypeProto float_tensor;

  for (size_t i = 0; i < config.size(); i++) {
    float_tensor.mutable_tensor_type()->set_elem_type(element_type);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(std::get<2>(config[i]));
    
    // Input gradient tensor defined tests
    auto& allreduce_input_arg = graph.GetOrCreateNodeArg(std::get<0>(config[i]), &float_tensor);
    inputs.push_back(&allreduce_input_arg);

    // Output tensor defined by tests
    ONNX_NAMESPACE::TypeProto output_float_tensor;
    auto output_type = build_optimizer ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT :
                                      ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    output_float_tensor.mutable_tensor_type()->set_elem_type(output_type);
    output_float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(std::get<2>(config[i]));
    auto& output_arg_1 = graph.GetOrCreateNodeArg(std::get<1>(config[i]), &output_float_tensor);
    outputs.push_back(&output_arg_1);
  }
  std::string allreduce_op_name = adasum_reduce_type == training::AdasumReductionType::None ?
                                  "NcclAllReduce" : "AdasumAllReduce";

  // If using hierarchical reduction, nccl allreduce will be used before adasum to get sum on local ranks.
  if (adasum_reduce_type == training::AdasumReductionType::GpuHierarchicalReduction) {
    std::string level_1_allreduce = "NcclAllReduce";
    std::vector<onnxruntime::NodeArg*> level_1_inputs;
    std::vector<onnxruntime::NodeArg*> level_1_outputs;
    for (size_t i = 0; i < config.size(); i++) {
      // Set graph input as input to the level 1 allreduce node
      level_1_inputs.push_back(inputs[i]);
      // Output tensor
      auto& level_1_output_arg = graph.GetOrCreateNodeArg(std::get<0>(config[i]) + "_level_1_out", &float_tensor);
    
      level_1_outputs.push_back(&level_1_output_arg);
    }
    auto& level_1_allreduce_node =  graph.AddNode("node_level_1", level_1_allreduce,
                                                  "level 1 allreduce.", level_1_inputs, level_1_outputs,
                                                  nullptr/*attributes*/, kMSDomain);
    ONNX_NAMESPACE::AttributeProto level_1_group_type_attribute;

    level_1_group_type_attribute.set_name("group_type");
    level_1_group_type_attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    level_1_group_type_attribute.set_i(2/*node local data parallel*/);
    level_1_allreduce_node.AddAttribute("group_type", level_1_group_type_attribute);
    // Set inputs of next node to be outputs of level 1 reduction node.
    inputs.clear();
    inputs = std::move(level_1_outputs);

    if (build_optimizer) {
      ONNX_NAMESPACE::TypeProto scale_tensor;
      std::vector<onnxruntime::NodeArg*> scale_grad_inputs;
      std::vector<onnxruntime::NodeArg*> scale_grad_outputs;
      auto scale_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
      scale_tensor.mutable_tensor_type()->set_elem_type(scale_type);
      scale_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      auto& scale_input_arg = graph.GetOrCreateNodeArg(grad_divisor_string, &scale_tensor);

      for (size_t i = 0; i < config.size(); i ++) {
        scale_grad_inputs.clear();
        scale_grad_inputs.push_back(&scale_input_arg);
        scale_grad_inputs.push_back(inputs[i]);
        // Output tensor
        auto& scale_grad_output_arg = graph.GetOrCreateNodeArg(inputs[i]->Name() + "_scaled", &float_tensor);
        scale_grad_outputs.push_back(&scale_grad_output_arg);
        auto& scaled_grad_node =  graph.AddNode(std::get<0>(config[i]) + "_scaled_grad", "MixedPrecisionScale",
                                              "scale grad", scale_grad_inputs, {&scale_grad_output_arg},
                                              nullptr/*attributes*/, kMSDomain);
        ONNX_NAMESPACE::AttributeProto scale_attribute;
        scale_attribute.set_name("to");
        scale_attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
        scale_attribute.set_i(static_cast<int64_t>(element_type));
        scaled_grad_node.AddAttribute("to", scale_attribute);
      }
      // Set inputs of next node to be outputs of scale node.
      inputs.clear();
      inputs = std::move(scale_grad_outputs);
    }
  }

  // Build optimizer before reduction for Adasum
  if(build_optimizer && adasum_reduce_type != training::AdasumReductionType::None) {
    std::vector<onnxruntime::NodeArg*> intermediate_output;
    for (size_t i = 0; i < config.size(); i++) {
      // Adasum computes on the updated gradients of optimizer, so suffix tensor name with gradient out.
      auto& optimizer_output_arg = graph.GetOrCreateNodeArg(std::get<0>(config[i]) + gradient_out_suffix_string, &float_tensor);
      build_optimizer_node(graph, element_type, std::get<2>(config[i]), std::get<0>(config[i]),
                           inputs[i], &optimizer_output_arg, adasum_reduce_type);
      intermediate_output.push_back(&optimizer_output_arg);
    }
    // Set inputs to next node to be outputs of optimizer.
    inputs.clear();
    inputs = std::move(intermediate_output);
  }

  std::vector<onnxruntime::NodeArg*> allreduce_outputs;
  if (build_optimizer) {
    // If build_optimizer, outputs of allreduce need to be appended with allreduce suffix string.
    for (size_t i = 0; i < config.size(); i++) {
      auto& allreduce_output_arg = graph.GetOrCreateNodeArg(std::get<0>(config[i]) + allreduce_output_suffix_string, &float_tensor);
      allreduce_outputs.push_back(&allreduce_output_arg);
    }
  }
  else {
    // If not build_optimizer, outputs of allreduce are graph outputs.
    allreduce_outputs = std::move(outputs);
  }

  auto& allreduce_node =  graph.AddNode("node_allreduce", allreduce_op_name, "node allreduce.", inputs, allreduce_outputs,
                                        nullptr/*attributes*/, kMSDomain);
  if (adasum_reduce_type != training::AdasumReductionType::None) {
    // Attribute
    ONNX_NAMESPACE::AttributeProto adasum_reduction_type_attribute;
    adasum_reduction_type_attribute.set_name("reduce_algo");
    adasum_reduction_type_attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    adasum_reduction_type_attribute.set_i(static_cast<int64_t>(adasum_reduce_type));
    allreduce_node.AddAttribute("reduce_algo", adasum_reduction_type_attribute);
  }
  else {
    // Attribute
    ONNX_NAMESPACE::AttributeProto group_type_attribute;
    group_type_attribute.set_name("group_type");
    group_type_attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    group_type_attribute.set_i(0/*data parallel*/);
    allreduce_node.AddAttribute("group_type", group_type_attribute);
  }

  if(build_optimizer) {
    if (adasum_reduce_type == training::AdasumReductionType::None) {
      // If build_optimizer and regular allreduce, inputs are outputs of previous allreduce node. 
      inputs.clear();
      inputs = std::move(allreduce_outputs);
      for (size_t i = 0; i< config.size(); i++) {
        build_optimizer_node(graph, element_type, std::get<2>(config[i]), std::get<0>(config[i]),
                             inputs[i], outputs[i], adasum_reduce_type);
      }
    }
    else {
      // If build_optimizer and Adasum allreduce, outputs of Adasum nodes need to be added back to original weights. 
      std::vector<NodeArg*> weight_update_input_args;
      std::vector<NodeArg*> weight_update_output_args;
      std::string accumulator_op = "InPlaceAccumulator";
      for (size_t i = 0; i < config.size(); i++) {
        weight_update_input_args.clear();
        weight_update_output_args.clear();
        // weight input
        ONNX_NAMESPACE::TypeProto weight_input_tensor;
        weight_input_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        weight_input_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(std::get<2>(config[i]));
        auto& weight_input_arg = graph.GetOrCreateNodeArg(std::get<0>(config[i]) + weight_suffix_string, &weight_input_tensor);
        weight_update_input_args.push_back(&weight_input_arg);

        // reduced gradient input
        weight_update_input_args.push_back(allreduce_outputs[i]);

        // final output
        ONNX_NAMESPACE::TypeProto updated_weight_out_tensor;
        updated_weight_out_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        updated_weight_out_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(std::get<2>(config[i]));
        auto& updated_weight_out_arg = graph.GetOrCreateNodeArg(std::get<1>(config[i]), &updated_weight_out_tensor);
        weight_update_output_args.push_back(&updated_weight_out_arg);

        graph.AddNode(std::get<0>(config[i]) + "_node_update_weight", accumulator_op, "node update weight.",
                      weight_update_input_args, {&updated_weight_out_arg},
                      nullptr/*attributes*/, kMSDomain);
      }
    }
  }

  auto status = graph.Resolve();
  if (!status.IsOK()) {
    std::cout<<"Status not OK. Error: "<<status.ErrorMessage()<<std::endl;
  }
  ASSERT_TRUE(status.IsOK());
}
#ifdef USE_CUDA
std::unique_ptr<IExecutionProvider> create_cuda_execution_provider() {
  CUDAExecutionProviderInfo info;
  OrtDevice::DeviceId device_id = static_cast<OrtDevice::DeviceId>(training::MPIContext::GetInstance().GetLocalRank());
  size_t gpu_mem_limit = std::numeric_limits<size_t>::max();
  gpu_mem_limit = static_cast<size_t>(1 * 1024 * 1024 * 1024);

  info.device_id = device_id;
  info.gpu_mem_limit = gpu_mem_limit;
  info.arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo;
  return std::make_unique<CUDAExecutionProvider>(info);
}

TEST(AllreduceTest, GPUHierarchicalAdasumAllreduceOptimizerTest) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

  training::DistributedRunConfig config = {training::MPIContext::GetInstance().GetWorldRank(),// world rank
                                          training::MPIContext::GetInstance().GetWorldSize(),// world size
                                          training::MPIContext::GetInstance().GetLocalRank(),// local rank
                                          training::MPIContext::GetInstance().GetLocalSize(),// local size
                                          training::MPIContext::GetInstance().GetWorldSize(),// data parallel group
                                          };
  training::DistributedRunContext::CreateInstance(config);

  std::string input_gradient_string = "input_t";
  std::string output_gradient_string = "output_t";
  AllreduceGraphConfigVector adasum_graph_configs;

  // Learning rate
  float eta = 0.5f;

  // Update Count
  int64_t update_count = 1;

  // Gradient scale divisor
  float scale = 1.f / training::MPIContext::GetInstance().GetLocalSize();

  // Gradients
  std::vector<int64_t> dims_allreduce_input = {3};
  std::vector<float> values_allreduce_input;

  if(training::MPIContext::GetInstance().GetWorldRank() == 0) {
    values_allreduce_input.push_back(3.f);
    values_allreduce_input.push_back(4.f);
    values_allreduce_input.push_back(5.f);
  }
  else {
    values_allreduce_input.push_back(5.f);
    values_allreduce_input.push_back(6.f);
    values_allreduce_input.push_back(7.f);

  }

  // Weights
  std::vector<float> values_weight_input;
  values_weight_input.push_back(1.f);
  values_weight_input.push_back(2.f);
  values_weight_input.push_back(3.f);

  // M1
  std::vector<float> values_m1_input;
  values_m1_input.push_back(0.1f);
  values_m1_input.push_back(0.2f);
  values_m1_input.push_back(0.3f);

  // M2
  std::vector<float> values_m2_input;
  values_m2_input.push_back(0.4f);
  values_m2_input.push_back(0.5f);
  values_m2_input.push_back(0.6f);

  onnxruntime::Model model("adasum_optimizer_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto adasum_graph_config = std::tuple<std::string, std::string, int>(input_gradient_string,
                                                                        output_gradient_string,
                                                                        dims_allreduce_input[0]);
  adasum_graph_configs.push_back(adasum_graph_config);             

  build_allreduce_graph(graph, adasum_graph_configs, training::AdasumReductionType::GpuHierarchicalReduction, true/*build_optimizer*/,
                        false/*half_precision*/);
  
  std::string model_file_name = "GPUHierarchicalAdasumAllreduceOptimizerTest.onnx";
  auto status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "AllreduceTest.GPUHierarchicalAdasumAllreduceOptimizerTest";
  
  onnxruntime::InferenceSession session_object{so, GetEnvironment()};
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  
  auto test_cuda_ep = create_cuda_execution_provider();
 
  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(test_cuda_ep)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  NameMLValMap feeds;
  // Learning rate inputs
  OrtValue ml_value_eta;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), {1}, {eta}, &ml_value_eta);
  feeds.insert(std::make_pair(lr_string, ml_value_eta));

  // Grad scale inputs
  OrtValue ml_value_scale;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), {1}, {scale}, &ml_value_scale);
  feeds.insert(std::make_pair(grad_divisor_string, ml_value_scale));

  // Update count inputs
  OrtValue ml_value_uc;
  CreateMLValue<int64_t>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), {1}, {update_count}, &ml_value_uc);
  feeds.insert(std::make_pair(update_count_string, ml_value_uc));

  // Gradient inputs
  OrtValue ml_value_input_t;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_allreduce_input, values_allreduce_input, &ml_value_input_t);
  feeds.insert(std::make_pair(input_gradient_string, ml_value_input_t));

  // Weight inputs
  OrtValue ml_value_weight_inputs;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                       dims_allreduce_input,
                       values_weight_input,
                       &ml_value_weight_inputs);
  feeds.insert(std::make_pair(input_gradient_string + weight_suffix_string, ml_value_weight_inputs));

  // M1 inputs
  OrtValue ml_value_m1_inputs;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                       dims_allreduce_input,
                       values_m1_input,
                       &ml_value_m1_inputs);
  feeds.insert(std::make_pair(input_gradient_string + m1_suffix_string, ml_value_m1_inputs));

  // M2 inputs
  OrtValue ml_value_m2_inputs;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                       dims_allreduce_input,
                       values_m2_input,
                       &ml_value_m2_inputs);
  feeds.insert(std::make_pair(input_gradient_string + m2_suffix_string, ml_value_m2_inputs));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back(output_gradient_string);

  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_allreduce = {3};
  std::vector<float> expected_values_allreduce = {0.6199f, 1.5305f, 2.4542f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!status.IsOK()) {
    std::cout<<"Status not OK. Error: "<<status.ErrorMessage()<<std::endl;
  }
  ASSERT_TRUE(status.IsOK());
  
  ASSERT_EQ(1u, fetches.size());
  
  // Verify tensor data
  auto& actual_output_tensor = fetches[0].Get<Tensor>();
  TensorShape expected_shape(expected_dims_allreduce);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_tensor.Shape()));

  const std::vector<float> found(actual_output_tensor.template Data<float>(),
                             actual_output_tensor.template Data<float>() + expected_values_allreduce.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_NEAR((double)expected_values_allreduce[i], (double)found[i], 1e-4f);

  if(training::MPIContext::GetInstance().GetWorldRank() == 0)
   std::remove(model_file_name.c_str());
}

TEST(AllreduceTest, GPUHierarchicalAdasumAllreduceOptimizerFP16Test) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

  training::DistributedRunConfig config = {training::MPIContext::GetInstance().GetWorldRank(),// world rank
                                          training::MPIContext::GetInstance().GetWorldSize(),// world size
                                          training::MPIContext::GetInstance().GetLocalRank(),// local rank
                                          training::MPIContext::GetInstance().GetLocalSize(),// local size
                                          training::MPIContext::GetInstance().GetWorldSize(),// data parallel group
                                          };
  training::DistributedRunContext::CreateInstance(config);

  std::string input_gradient_string = "input_t";
  std::string output_gradient_string = "output_t";
  AllreduceGraphConfigVector adasum_graph_configs;

  // Learning rate
  float eta = 0.5f;

  // Update Count
  int64_t update_count = 1;

  // Gradient scale divisor
  float scale = 1.f / training::MPIContext::GetInstance().GetLocalSize();

  // Gradients
  std::vector<int64_t> dims_allreduce_input = {3};
  std::vector<float> values_allreduce_input;

  if(training::MPIContext::GetInstance().GetWorldRank() == 0) {
    values_allreduce_input.push_back(3.f);
    values_allreduce_input.push_back(4.f);
    values_allreduce_input.push_back(5.f);
  }
  else {
    values_allreduce_input.push_back(5.f);
    values_allreduce_input.push_back(6.f);
    values_allreduce_input.push_back(7.f);

  }

  std::vector<MLFloat16> values_allreduce_input_half(dims_allreduce_input[0]);

  ConvertFloatToMLFloat16(values_allreduce_input.data(), values_allreduce_input_half.data(), dims_allreduce_input[0]);

  // Weights
  std::vector<float> values_weight_input;
  values_weight_input.push_back(1.f);
  values_weight_input.push_back(2.f);
  values_weight_input.push_back(3.f);

  // std::vector<MLFloat16> values_weight_input_half(dims_allreduce_input[0]);

  // ConvertFloatToMLFloat16(values_weight_input.data(), values_weight_input_half.data(), dims_allreduce_input[0]);

  // M1
  std::vector<float> values_m1_input;
  values_m1_input.push_back(0.1f);
  values_m1_input.push_back(0.2f);
  values_m1_input.push_back(0.3f);

  // M2
  std::vector<float> values_m2_input;
  values_m2_input.push_back(0.4f);
  values_m2_input.push_back(0.5f);
  values_m2_input.push_back(0.6f);

  onnxruntime::Model model("adasum_optimizer_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto adasum_graph_config = std::tuple<std::string, std::string, int>(input_gradient_string,
                                                                        output_gradient_string,
                                                                        dims_allreduce_input[0]);
  adasum_graph_configs.push_back(adasum_graph_config);             

  build_allreduce_graph(graph, adasum_graph_configs, training::AdasumReductionType::GpuHierarchicalReduction, true/*build_optimizer*/,
                        true/*half_precision*/);
  
  std::string model_file_name = "GPUHierarchicalAdasumAllreduceOptimizerFP16Test.onnx";
  auto status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "AllreduceTest.GPUHierarchicalAdasumAllreduceOptimizerFP16Test";
  
  onnxruntime::InferenceSession session_object{so, GetEnvironment()};
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  
  auto test_cuda_ep = create_cuda_execution_provider();
 
  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(test_cuda_ep)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  NameMLValMap feeds;
  // Learning rate inputs
  OrtValue ml_value_eta;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), {1}, {eta}, &ml_value_eta);
  feeds.insert(std::make_pair(lr_string, ml_value_eta));

  // Grad scale inputs
  OrtValue ml_value_scale;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), {1}, {scale}, &ml_value_scale);
  feeds.insert(std::make_pair(grad_divisor_string, ml_value_scale));

  // Update count inputs
  OrtValue ml_value_uc;
  CreateMLValue<int64_t>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), {1}, {update_count}, &ml_value_uc);
  feeds.insert(std::make_pair(update_count_string, ml_value_uc));

  // Gradient inputs
  OrtValue ml_value_input_t;
  CreateMLValue<MLFloat16>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                           dims_allreduce_input,
                           values_allreduce_input_half,
                           &ml_value_input_t);
  feeds.insert(std::make_pair(input_gradient_string, ml_value_input_t));

  // Weight inputs
  OrtValue ml_value_weight_inputs;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                       dims_allreduce_input,
                       values_weight_input,
                       &ml_value_weight_inputs);
  feeds.insert(std::make_pair(input_gradient_string + weight_suffix_string, ml_value_weight_inputs));

  // M1 inputs
  OrtValue ml_value_m1_inputs;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                       dims_allreduce_input,
                       values_m1_input,
                       &ml_value_m1_inputs);
  feeds.insert(std::make_pair(input_gradient_string + m1_suffix_string, ml_value_m1_inputs));

  // M2 inputs
  OrtValue ml_value_m2_inputs;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                       dims_allreduce_input,
                       values_m2_input,
                       &ml_value_m2_inputs);
  feeds.insert(std::make_pair(input_gradient_string + m2_suffix_string, ml_value_m2_inputs));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back(output_gradient_string);

  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_allreduce = {3};
  std::vector<float> expected_values_allreduce = {0.6199f, 1.5305f, 2.4542f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!status.IsOK()) {
    std::cout<<"Status not OK. Error: "<<status.ErrorMessage()<<std::endl;
  }
  ASSERT_TRUE(status.IsOK());
  
  ASSERT_EQ(1u, fetches.size());
  
  // Verify tensor data
  auto& actual_output_tensor = fetches[0].Get<Tensor>();
  TensorShape expected_shape(expected_dims_allreduce);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_tensor.Shape()));

  const std::vector<float> found(actual_output_tensor.template Data<float>(),
                             actual_output_tensor.template Data<float>() + expected_values_allreduce.size());

  for (size_t i = 0; i < found.size(); i++)
    ASSERT_NEAR((double)expected_values_allreduce[i], (double)found[i], 1e-4f);

  if(training::MPIContext::GetInstance().GetWorldRank() == 0)
   std::remove(model_file_name.c_str());
}

TEST(AllreduceTest, GPUHierarchicalAdasumAllreduceTest) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

  training::DistributedRunConfig config = {training::MPIContext::GetInstance().GetWorldRank(),// world rank
                                          training::MPIContext::GetInstance().GetWorldSize(),// world size
                                          training::MPIContext::GetInstance().GetLocalRank(),// local rank
                                          training::MPIContext::GetInstance().GetLocalSize(),// local size
                                          training::MPIContext::GetInstance().GetWorldSize(),// data parallel group
                                          };
  training::DistributedRunContext::CreateInstance(config);

  std::vector<int64_t> dims_allreduce_input = {3};
  std::vector<float> values_allreduce_input;
  std::string input_gradient_string = "input_t";
  std::string output_gradient_string = "output_t";
  AllreduceGraphConfigVector adasum_graph_configs;

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

  onnxruntime::Model model("adasum_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto adasum_graph_config = std::tuple<std::string, std::string, int>(input_gradient_string,
                                                                            output_gradient_string,
                                                                            dims_allreduce_input[0]);
  adasum_graph_configs.push_back(adasum_graph_config);             
  build_allreduce_graph(graph, adasum_graph_configs, training::AdasumReductionType::GpuHierarchicalReduction);
  
  std::string model_file_name = "GPUHierarchicalAdasumAllreduceTest.onnx";
  auto status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "AllreduceTest.GPUHierarchicalAdasumAllreduceTest";
  
  onnxruntime::InferenceSession session_object{so, GetEnvironment()};
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  
  auto test_cuda_ep = create_cuda_execution_provider();
 
  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(test_cuda_ep)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());
  OrtValue ml_value_input_t;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_allreduce_input, values_allreduce_input, &ml_value_input_t);
  
  NameMLValMap feeds;
  feeds.insert(std::make_pair(input_gradient_string, ml_value_input_t));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back(output_gradient_string);
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_allreduce = {3};
  std::vector<float> expected_values_allreduce = {11, 13, 15};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!status.IsOK()) {
    std::cout<<"Status not OK. Error: "<<status.ErrorMessage()<<std::endl;
  }
  ASSERT_TRUE(status.IsOK());
  
  ASSERT_EQ(1u, fetches.size());
  
  // Verify tensor data
  auto& actual_output_tensor = fetches[0].Get<Tensor>();
  TensorShape expected_shape(expected_dims_allreduce);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_tensor.Shape()));

  const std::vector<float> found(actual_output_tensor.template Data<float>(),
                             actual_output_tensor.template Data<float>() + expected_values_allreduce.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_NEAR((double)expected_values_allreduce[i], (double)found[i], 1e-4f);

  if(training::MPIContext::GetInstance().GetWorldRank() == 0)
   std::remove(model_file_name.c_str());
}

TEST(AllreduceTest, GPUHierarchicalAdasumFP16AllreduceTest) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

  training::DistributedRunConfig config = {training::MPIContext::GetInstance().GetWorldRank(),// world rank
                                          training::MPIContext::GetInstance().GetWorldSize(),// world size
                                          training::MPIContext::GetInstance().GetLocalRank(),// local rank
                                          training::MPIContext::GetInstance().GetLocalSize(),// local size
                                          training::MPIContext::GetInstance().GetWorldSize(),// data parallel group
                                          };
  training::DistributedRunContext::CreateInstance(config);

  std::vector<int64_t> dims_allreduce_input = {4};
  std::vector<float> values_allreduce_input;
  std::string input_gradient_string = "input_t";
  std::string output_gradient_string = "output_t";
  AllreduceGraphConfigVector adasum_graph_configs;

  if(training::MPIContext::GetInstance().GetWorldRank() == 0) {
    values_allreduce_input.push_back(4.f);
    values_allreduce_input.push_back(5.f);
    values_allreduce_input.push_back(6.f);
    values_allreduce_input.push_back(7.f);
  }
  else {
    values_allreduce_input.push_back(8.f);
    values_allreduce_input.push_back(9.f);
    values_allreduce_input.push_back(10.f);
    values_allreduce_input.push_back(11.f);
  }

  std::vector<MLFloat16> values_allreduce_input_half(4);

  ConvertFloatToMLFloat16(values_allreduce_input.data(), values_allreduce_input_half.data(), 4);

  onnxruntime::Model model("adasum_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto adasum_graph_config = std::tuple<std::string, std::string, int>(input_gradient_string,
                                                                       output_gradient_string,
                                                                       dims_allreduce_input[0]);
  adasum_graph_configs.push_back(adasum_graph_config);             
  build_allreduce_graph(graph, adasum_graph_configs, training::AdasumReductionType::GpuHierarchicalReduction,
                        false/*build_optimizer*/,
                        true/*half_precision*/);
  
  std::string model_file_name = "GPUHierarchicalAdasumFP16AllreduceTest.onnx";
  auto status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "AllreduceTest.GPUHierarchicalAdasumFP16AllreduceTest";
  
  onnxruntime::InferenceSession session_object{so, GetEnvironment()};
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  
  auto test_cuda_ep = create_cuda_execution_provider();
 
  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(test_cuda_ep)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());
  OrtValue ml_value_input_t;
  CreateMLValue<MLFloat16>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                           dims_allreduce_input, values_allreduce_input_half, &ml_value_input_t);
  
  NameMLValMap feeds;
  feeds.insert(std::make_pair(input_gradient_string, ml_value_input_t));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back(output_gradient_string);
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_allreduce = {4};
  std::vector<float> expected_values_allreduce = {12, 14, 16, 18};

  std::vector<MLFloat16> expected_values_allreduce_half(4);

  ConvertFloatToMLFloat16(expected_values_allreduce.data(), expected_values_allreduce_half.data(), 4);

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!status.IsOK()) {
    std::cout<<"Status not OK. Error: "<<status.ErrorMessage()<<std::endl;
  }
  ASSERT_TRUE(status.IsOK());
  
  ASSERT_EQ(1u, fetches.size());
  
  // Verify tensor data
  auto& actual_output_tensor = fetches[0].Get<Tensor>();
  TensorShape expected_shape(expected_dims_allreduce);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_tensor.Shape()));

  const std::vector<MLFloat16> found_half(actual_output_tensor.template Data<MLFloat16>(),
                             actual_output_tensor.template Data<MLFloat16>() + expected_values_allreduce_half.size());
  std::vector<float> found(found_half.size());
  ConvertMLFloat16ToFloat(found_half.data(), found.data(), found.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_NEAR((double)expected_values_allreduce[i], (double)found[i], 1e-3f);
  
  if(training::MPIContext::GetInstance().GetWorldRank() == 0)
   std::remove(model_file_name.c_str());
}

TEST(AllreduceTest, GPUAdasumAllreduceTest) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

  training::DistributedRunConfig config = {training::MPIContext::GetInstance().GetWorldRank(),// world rank
                                          training::MPIContext::GetInstance().GetWorldSize(),// world size
                                          training::MPIContext::GetInstance().GetLocalRank(),// local rank
                                          training::MPIContext::GetInstance().GetLocalSize(),// local size
                                          training::MPIContext::GetInstance().GetWorldSize(),// data parallel group
                                          };
  training::DistributedRunContext::CreateInstance(config);

  std::vector<int64_t> dims_allreduce_input = {4};
  std::vector<float> values_allreduce_input;
  std::string input_gradient_string = "input_t";
  std::string output_gradient_string = "output_t";
  AllreduceGraphConfigVector adasum_graph_configs;

  if(training::MPIContext::GetInstance().GetWorldRank() == 0) {
    values_allreduce_input.push_back(4.f);
    values_allreduce_input.push_back(5.f);
    values_allreduce_input.push_back(6.f);
    values_allreduce_input.push_back(7.f);
  }
  else {
    values_allreduce_input.push_back(8.f);
    values_allreduce_input.push_back(9.f);
    values_allreduce_input.push_back(10.f);
    values_allreduce_input.push_back(11.f);
  }

  onnxruntime::Model model("adasum_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto adasum_graph_config = std::tuple<std::string, std::string, int>(input_gradient_string,
                                                                       output_gradient_string,
                                                                       dims_allreduce_input[0]);
  adasum_graph_configs.push_back(adasum_graph_config);             

  build_allreduce_graph(graph, adasum_graph_configs, training::AdasumReductionType::CpuReduction);
  
  std::string model_file_name = "GPUAdasumAllreduceTest.onnx";
  auto status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "AllreduceTest.GPUAdasumAllreduceTest";
  
  onnxruntime::InferenceSession session_object{so, GetEnvironment()};
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  
  auto test_cuda_ep = create_cuda_execution_provider();
 
  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(test_cuda_ep)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());
  OrtValue ml_value_input_t;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                       dims_allreduce_input,
                       values_allreduce_input,
                       &ml_value_input_t);
  
  NameMLValMap feeds;
  feeds.insert(std::make_pair(input_gradient_string, ml_value_input_t));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back(output_gradient_string);
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_allreduce = {4};
  std::vector<float> expected_values_allreduce = {6.2643, 7.1228, 7.9812, 8.8397};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!status.IsOK()) {
    std::cout<<"Status not OK. Error: "<<status.ErrorMessage()<<std::endl;
  }
  ASSERT_TRUE(status.IsOK());
  
  ASSERT_EQ(1u, fetches.size());
  
  // Verify tensor data
  auto& actual_output_tensor = fetches[0].Get<Tensor>();
  TensorShape expected_shape(expected_dims_allreduce);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_tensor.Shape()));

  const std::vector<float> found(actual_output_tensor.template Data<float>(),
                             actual_output_tensor.template Data<float>() + expected_values_allreduce.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_NEAR((double)expected_values_allreduce[i], (double)found[i], 1e-4f);
  
  if(training::MPIContext::GetInstance().GetWorldRank() == 0)
   std::remove(model_file_name.c_str());
}

TEST(AllreduceTest, GPUAdasumFP16AllreduceTest) {
  if (training::MPIContext::GetInstance().GetWorldSize() != 2) {
    return;
  }

  training::DistributedRunConfig config = {training::MPIContext::GetInstance().GetWorldRank(),// world rank
                                          training::MPIContext::GetInstance().GetWorldSize(),// world size
                                          training::MPIContext::GetInstance().GetLocalRank(),// local rank
                                          training::MPIContext::GetInstance().GetLocalSize(),// local size
                                          training::MPIContext::GetInstance().GetWorldSize(),// data parallel group
                                          };
  training::DistributedRunContext::CreateInstance(config);

  std::vector<int64_t> dims_allreduce_input = {4};
  std::vector<float> values_allreduce_input;
  std::string input_gradient_string = "input_t";
  std::string output_gradient_string = "output_t";
  AllreduceGraphConfigVector adasum_graph_configs;

  if(training::MPIContext::GetInstance().GetWorldRank() == 0) {
    values_allreduce_input.push_back(4.f);
    values_allreduce_input.push_back(5.f);
    values_allreduce_input.push_back(6.f);
    values_allreduce_input.push_back(7.f);
  }
  else {
    values_allreduce_input.push_back(8.f);
    values_allreduce_input.push_back(9.f);
    values_allreduce_input.push_back(10.f);
    values_allreduce_input.push_back(11.f);
  }

  std::vector<MLFloat16> values_allreduce_input_half(4);

  ConvertFloatToMLFloat16(values_allreduce_input.data(), values_allreduce_input_half.data(), 4);

  onnxruntime::Model model("adasum_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto adasum_graph_config = std::tuple<std::string, std::string, int>(input_gradient_string,
                                                                       output_gradient_string,
                                                                       dims_allreduce_input[0]);
  adasum_graph_configs.push_back(adasum_graph_config);             

  build_allreduce_graph(graph, adasum_graph_configs, training::AdasumReductionType::CpuReduction, true/*half_precision*/);
  
  std::string model_file_name = "GPUAdasumFP16AllreduceTest.onnx";
  auto status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "AllreduceTest.GPUAdasumFP16AllreduceTest";
  
  onnxruntime::InferenceSession session_object{so, GetEnvironment()};
  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  
  auto test_cuda_ep = create_cuda_execution_provider();
 
  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(test_cuda_ep)).IsOK());

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());
  OrtValue ml_value_input_t;
  CreateMLValue<MLFloat16>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault),
                           dims_allreduce_input, values_allreduce_input_half, &ml_value_input_t);
  
  NameMLValMap feeds;
  feeds.insert(std::make_pair(input_gradient_string, ml_value_input_t));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("node_1_out_1");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_allreduce = {4};
  std::vector<float> expected_values_allreduce = {6.2643, 7.1228, 7.9812, 8.8397};

  std::vector<MLFloat16> expected_values_allreduce_half(4);

  ConvertFloatToMLFloat16(expected_values_allreduce.data(), expected_values_allreduce_half.data(), 4);

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!status.IsOK()) {
    std::cout<<"Status not OK. Error: "<<status.ErrorMessage()<<std::endl;
  }
  ASSERT_TRUE(status.IsOK());
  
  ASSERT_EQ(1u, fetches.size());
  
  // Verify tensor data
  auto& actual_output_tensor = fetches[0].Get<Tensor>();
  TensorShape expected_shape(expected_dims_allreduce);
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape),
            *reinterpret_cast<const std::vector<int64_t>*>(&actual_output_tensor.Shape()));

  const std::vector<MLFloat16> found_half(actual_output_tensor.template Data<MLFloat16>(),
                             actual_output_tensor.template Data<MLFloat16>() + expected_values_allreduce_half.size());
  std::vector<float> found(found_half.size());
  ConvertMLFloat16ToFloat(found_half.data(), found.data(), found.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_NEAR((double)expected_values_allreduce[i], (double)found[i], 5e-3);
  
  if(training::MPIContext::GetInstance().GetWorldRank() == 0)
   std::remove(model_file_name.c_str());
}

#endif
}  // namespace test
}  // namespace onnxruntime
#endif // USE_MPI