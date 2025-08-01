// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms

#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <algorithm>
#include <random>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/api_asserts.h"
#include "core/graph/basic_types.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/model_saving_options.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
namespace test {
#ifdef _WIN32

Utils::NvTensorRtRtxEpInfo Utils::nv_tensorrt_rtx_ep_info;

void Utils::GetEp(Ort::Env& env, const std::string& ep_name, const OrtEpDevice*& ep_device) {
  const OrtApi& c_api = Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices;
  ASSERT_ORTSTATUS_OK(c_api.GetEpDevices(env, &ep_devices, &num_devices));

  auto it = std::find_if(ep_devices, ep_devices + num_devices,
                         [&c_api, &ep_name](const OrtEpDevice* ep_device) {
                           // NV TensorRT RTX EP uses registration name as ep name
                           return c_api.EpDevice_EpName(ep_device) == ep_name;
                         });

  if (it == ep_devices + num_devices) {
    ep_device = nullptr;
  } else {
    ep_device = *it;
  }
}

void Utils::RegisterAndGetNvTensorRtRtxEp(Ort::Env& env, RegisteredEpDeviceUniquePtr& registered_ep) {
  const OrtApi& c_api = Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(c_api.RegisterExecutionProviderLibrary(env,
                                                             nv_tensorrt_rtx_ep_info.registration_name.c_str(),
                                                             nv_tensorrt_rtx_ep_info.library_path.c_str()));
  const OrtEpDevice* nv_tensorrt_rtx_ep = nullptr;
  GetEp(env, nv_tensorrt_rtx_ep_info.registration_name, nv_tensorrt_rtx_ep);
  ASSERT_NE(nv_tensorrt_rtx_ep, nullptr);

  registered_ep = RegisteredEpDeviceUniquePtr(nv_tensorrt_rtx_ep, [&env, c_api](const OrtEpDevice* /*ep*/) {
    c_api.UnregisterExecutionProviderLibrary(env, nv_tensorrt_rtx_ep_info.registration_name.c_str());
  });
}
#endif  // _WIN32

void CreateBaseModel(const PathString& model_name,
                     std::string graph_name,
                     std::vector<int> dims,
                     bool add_fast_gelu,
                     ONNX_NAMESPACE::TensorProto_DataType dtype,
                     const PathString& external_initializer_file) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(dtype);

  for (auto dim : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  ONNX_NAMESPACE::TypeProto dyn_float_tensor;
  dyn_float_tensor.mutable_tensor_type()->set_elem_type(dtype);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);

  auto& output_arg_2 = graph.GetOrCreateNodeArg("node_2_out_1", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg_2);

  if (add_fast_gelu) {
    auto& output_arg_3 = graph.GetOrCreateNodeArg("node_3_out_1", &dyn_float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_3);

    graph.AddNode("node_3", "FastGelu", "node 3.", inputs, outputs,
                  /* attributes */ nullptr, kMSDomain);

    inputs.clear();
    inputs.push_back(&output_arg_3);
  }

  ONNX_NAMESPACE::TypeProto float_scalar;
  float_scalar.mutable_tensor_type()->set_elem_type(dtype);
  float_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& input_scalar = graph.GetOrCreateNodeArg("S", &float_scalar);
  inputs.push_back(&input_scalar);

  auto& output_arg_4 = graph.GetOrCreateNodeArg("O", &dyn_float_tensor);

  outputs.clear();
  outputs.push_back(&output_arg_4);
  graph.AddNode("node_5", "Add", "node 5.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  if (!external_initializer_file.empty()) {
    ModelSavingOptions save_options(128);
    status = Model::SaveWithExternalInitializers(model, model_name, external_initializer_file, save_options);
  } else {
    status = Model::Save(model, model_name);
  }
  ASSERT_TRUE(status.IsOK());
}

// Helper to create large initializers
ONNX_NAMESPACE::TensorProto CreateLargeWeight(
    const std::string& name,
    ONNX_NAMESPACE::TensorProto_DataType dtype,
    const std::vector<int64_t>& shape,
    float scale = 0.02f) {
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(dtype);
  for (auto d : shape) tensor.add_dims(d);
  // Here we fill with random floats, but for real data, use your trained weights.
  size_t total_size = 1;
  for (int64_t d : shape) total_size *= d;
  if (dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    std::vector<float> data(total_size);
    std::default_random_engine rng;
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto& v : data) v = dist(rng);
    tensor.set_raw_data(data.data(), total_size * sizeof(float));
  } else if (dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    std::vector<MLFloat16> data(total_size);
    std::default_random_engine rng;
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto& v : data) v = MLFloat16(dist(rng));
    tensor.set_raw_data(data.data(), total_size * sizeof(MLFloat16));
  } else {
    throw std::runtime_error("Unsupported data type for large weight");
  }
  return tensor;
}

// Helper to add a GroupQueryAttention node
onnxruntime::NodeArg& AddGroupQueryAttention(
    onnxruntime::Graph& graph,
    onnxruntime::NodeArg& query,
    onnxruntime::NodeArg& key,
    onnxruntime::NodeArg& value,
    int batch_size,
    int head_dim,
    int seq_len,
    int num_heads,
    int kv_num_heads,
    float scale,
    ONNX_NAMESPACE::TensorProto_DataType dtype,
    const std::string& node_name) {
  // KV cache
  ONNX_NAMESPACE::TypeProto key_type;
  key_type.mutable_tensor_type()->set_elem_type(dtype);
  key_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch_size);
  key_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(kv_num_heads);
  key_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(seq_len);
  key_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(head_dim);
  auto& past_key = graph.GetOrCreateNodeArg(node_name + "_past_key", &key_type);

  ONNX_NAMESPACE::TypeProto value_type;
  value_type.mutable_tensor_type()->set_elem_type(dtype);
  value_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch_size);
  value_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(kv_num_heads);
  value_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(seq_len);
  value_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(head_dim);
  auto& past_value = graph.GetOrCreateNodeArg(node_name + "_past_value", &value_type);

  // Output
  auto& output = graph.GetOrCreateNodeArg(node_name + "_output", nullptr);

  // Create required initializers for GroupQueryAttention
  ONNX_NAMESPACE::TensorProto seqlens_k_tensor;
  seqlens_k_tensor.set_name(node_name + "_seqlens_k");
  seqlens_k_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  seqlens_k_tensor.add_dims(2);
  seqlens_k_tensor.set_dims(0, batch_size);
  seqlens_k_tensor.set_dims(0, 1);
  seqlens_k_tensor.add_int32_data(seq_len - 1);  // seqlens_k = total_sequence_length - 1
  graph.AddInitializedTensor(seqlens_k_tensor);

  ONNX_NAMESPACE::TensorProto total_seq_len_tensor;
  total_seq_len_tensor.set_name(node_name + "_total_sequence_length");
  total_seq_len_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  total_seq_len_tensor.add_int32_data(seq_len);
  graph.AddInitializedTensor(total_seq_len_tensor);

  // Get the initializers that were created for this node
  auto* seqlens_k = graph.GetNodeArg(node_name + "_seqlens_k");
  auto* total_sequence_length = graph.GetNodeArg(node_name + "_total_sequence_length");

  auto& present_value = graph.GetOrCreateNodeArg(node_name + "_present_value", nullptr);
  auto& present_key = graph.GetOrCreateNodeArg(node_name + "_present_key", nullptr);

  // Inputs - GroupQueryAttention requires at least 7 inputs (query, key, value, past_key, past_value, seqlens_k, total_sequence_length)
  std::vector<onnxruntime::NodeArg*> inputs = {
      &query,                 // 0: query
      &key,                   // 1: key
      &value,                 // 2: value
      &past_key,              // 3: past_key (optional)
      &past_value,            // 4: past_value (optional)
      seqlens_k,              // 5: seqlens_k (required)
      total_sequence_length,  // 6: total_sequence_length (required)
                              // nullptr,                   // 7: cos_cache (optional)
                              // nullptr,                   // 8: sin_cache (optional)
                              // nullptr,                   // 9: position_ids (optional)
                              // nullptr,                   // 10: attention_bias (optional)
                              // nullptr                    // 11: head_sink (optional)
  };

  // Attributes
  NodeAttributes attrs;
  ONNX_NAMESPACE::AttributeProto attr_heads;
  attr_heads.set_name("num_heads");
  attr_heads.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_heads.set_i(num_heads);
  attrs["num_heads"] = attr_heads;
  ONNX_NAMESPACE::AttributeProto attr_kv_num_heads;
  attr_kv_num_heads.set_name("kv_num_heads");
  attr_kv_num_heads.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_kv_num_heads.set_i(kv_num_heads);
  attrs["kv_num_heads"] = attr_kv_num_heads;
  ONNX_NAMESPACE::AttributeProto attr_scale;
  attr_scale.set_name("scale");
  attr_scale.set_type(onnx::AttributeProto_AttributeType_FLOAT);
  attr_scale.set_f(scale);
  attrs["scale"] = attr_scale;

  // Register node
  graph.AddNode(
      node_name,
      "GroupQueryAttention",
      "GroupQueryAttention Node",
      inputs,
      {&output, &present_key, &present_value},
      &attrs,
      "com.microsoft");

  return output;
}

void CreateLargeLLMModel(const PathString& model_path, const PathString& external_data_path) {
  // Model parameters (example: 24 layers, 4096 hidden dim, 32 attention heads, 8 kv heads => GQA)
  int batch_size = 1;
  int num_layers = 32;
  int hidden_dim = 2048;
  int q_num_heads = 8;
  int kv_num_heads = 1;  // GQA: q_num_heads > kv_num_heads, and divisible.
  int seq_length = 128;  // Short, for demonstration.
  int vocab_size = 32000;
  auto dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;

  // Set up model/graph
  onnxruntime::Model model("LLM_With_GQA", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // Input
  ONNX_NAMESPACE::TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(dtype);
  input_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch_size);
  input_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(seq_length);
  input_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(hidden_dim);
  auto& input = graph.GetOrCreateNodeArg("input", &input_type);

  auto* current_arg = &input;

  // Repeated layers: [Attention + MLP]
  for (int l = 0; l < num_layers; ++l) {
    // KV cache - initialize with zeros for the first forward pass
    int head_dim = hidden_dim / q_num_heads;

    // Split Q, K, V
    auto& q_split = graph.GetOrCreateNodeArg("q_split_" + std::to_string(l), nullptr);
    auto& k_split = graph.GetOrCreateNodeArg("k_split_" + std::to_string(l), nullptr);
    auto& v_split = graph.GetOrCreateNodeArg("v_split_" + std::to_string(l), nullptr);
    constexpr bool split = false;
    if constexpr (split) {
      // Attention weights (Q, K, V projections)
      auto wqkv = CreateLargeWeight("wqkv_" + std::to_string(l),
                                    dtype, {hidden_dim, hidden_dim * 3});
      graph.AddInitializedTensor(wqkv);

      // Q = input @ wq, K = input @ wk, V = input @ wv
      auto& qkv_arg = graph.GetOrCreateNodeArg("qkv_" + std::to_string(l), nullptr);
      graph.AddNode("QKV_Linear_" + std::to_string(l), "MatMul", "", {current_arg, graph.GetNodeArg(wqkv.name())}, {&qkv_arg});

      NodeAttributes attrs_split;
      ONNX_NAMESPACE::AttributeProto attr_split_axis;
      attr_split_axis.set_name("axis");
      attr_split_axis.set_type(onnx::AttributeProto_AttributeType_INT);
      attr_split_axis.set_i(-1);
      attrs_split["axis"] = attr_split_axis;
      ONNX_NAMESPACE::AttributeProto attr_split_num_outputs;
      attr_split_num_outputs.set_name("num_outputs");
      attr_split_num_outputs.set_type(onnx::AttributeProto_AttributeType_INT);
      attr_split_num_outputs.set_i(3);
      attrs_split["num_outputs"] = attr_split_num_outputs;
      graph.AddNode("Q_Split_" + std::to_string(l), "Split", "", {&qkv_arg}, {&q_split, &k_split, &v_split}, &attrs_split);
    } else {
      // Attention weights (Q, K, V projections)
      auto wq = CreateLargeWeight("wq_" + std::to_string(l),
                                  dtype, {hidden_dim, hidden_dim});
      graph.AddInitializedTensor(wq);
      auto wk = CreateLargeWeight("wk_" + std::to_string(l),
                                  dtype, {hidden_dim, head_dim * kv_num_heads});
      graph.AddInitializedTensor(wk);
      auto wv = CreateLargeWeight("wv_" + std::to_string(l),
                                  dtype, {hidden_dim, head_dim * kv_num_heads});
      graph.AddInitializedTensor(wv);

      // Q = input @ wq, K = input @ wk, V = input @ wv
      graph.AddNode("Q_Linear_" + std::to_string(l), "MatMul", "", {current_arg, graph.GetNodeArg(wq.name())}, {&q_split});
      graph.AddNode("K_Linear_" + std::to_string(l), "MatMul", "", {current_arg, graph.GetNodeArg(wk.name())}, {&k_split});
      graph.AddNode("V_Linear_" + std::to_string(l), "MatMul", "", {current_arg, graph.GetNodeArg(wv.name())}, {&v_split});
    }
    // Reshape Q, K, V
    auto& q_reshaped = graph.GetOrCreateNodeArg("q_reshaped_" + std::to_string(l), nullptr);
    auto& k_reshaped = graph.GetOrCreateNodeArg("k_reshaped_" + std::to_string(l), nullptr);
    auto& v_reshaped = graph.GetOrCreateNodeArg("v_reshaped_" + std::to_string(l), nullptr);

    ONNX_NAMESPACE::TensorProto q_shape_tensor;
    q_shape_tensor.set_name("q_shape_" + std::to_string(l));
    q_shape_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    q_shape_tensor.add_dims(3);
    q_shape_tensor.add_int64_data(batch_size);
    q_shape_tensor.add_int64_data(seq_length);
    q_shape_tensor.add_int64_data(head_dim * q_num_heads);
    graph.AddInitializedTensor(q_shape_tensor);

    ONNX_NAMESPACE::TensorProto k_shape_tensor;
    k_shape_tensor.set_name("k_shape_" + std::to_string(l));
    k_shape_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    k_shape_tensor.add_dims(3);
    k_shape_tensor.add_int64_data(batch_size);
    k_shape_tensor.add_int64_data(seq_length);
    k_shape_tensor.add_int64_data(head_dim * kv_num_heads);
    graph.AddInitializedTensor(k_shape_tensor);

    ONNX_NAMESPACE::TensorProto v_shape_tensor;
    v_shape_tensor.set_name("v_shape_" + std::to_string(l));
    v_shape_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    v_shape_tensor.add_dims(3);
    v_shape_tensor.add_int64_data(batch_size);
    v_shape_tensor.add_int64_data(seq_length);
    v_shape_tensor.add_int64_data(head_dim * kv_num_heads);
    graph.AddInitializedTensor(v_shape_tensor);

    graph.AddNode("Q_Reshape_" + std::to_string(l), "Reshape", "", {&q_split, graph.GetNodeArg(q_shape_tensor.name())}, {&q_reshaped});
    graph.AddNode("K_Reshape_" + std::to_string(l), "Reshape", "", {&k_split, graph.GetNodeArg(k_shape_tensor.name())}, {&k_reshaped});
    graph.AddNode("V_Reshape_" + std::to_string(l), "Reshape", "", {&v_split, graph.GetNodeArg(v_shape_tensor.name())}, {&v_reshaped});

    // Replace standard attention with GQA
    auto& attn_out = AddGroupQueryAttention(
        graph, q_reshaped, k_reshaped, v_reshaped,
        batch_size, head_dim, seq_length, q_num_heads, kv_num_heads,
        1.0f, dtype,
        "GQA_" + std::to_string(l));

    // Add an MLP block: (Linear + Activation + Linear)
    auto w1 = CreateLargeWeight("mlp_w1_" + std::to_string(l), dtype, {hidden_dim, hidden_dim * 4});
    auto w2 = CreateLargeWeight("mlp_w2_" + std::to_string(l), dtype, {hidden_dim * 4, hidden_dim});
    graph.AddInitializedTensor(w1);
    graph.AddInitializedTensor(w2);

    auto& mlp_hidden = graph.GetOrCreateNodeArg("mlp_hidden_" + std::to_string(l), nullptr);
    graph.AddNode("MLP_1_" + std::to_string(l), "MatMul", "", {&attn_out, graph.GetNodeArg(w1.name())}, {&mlp_hidden});
    auto& relu_out = graph.GetOrCreateNodeArg("relu_" + std::to_string(l), nullptr);
    graph.AddNode("Relu_" + std::to_string(l), "Relu", "", {&mlp_hidden}, {&relu_out});
    auto& mlp_out = graph.GetOrCreateNodeArg("mlp_out_" + std::to_string(l), nullptr);
    graph.AddNode("MLP_2_" + std::to_string(l), "MatMul", "", {&relu_out, graph.GetNodeArg(w2.name())}, {&mlp_out});
    current_arg = &mlp_out;  // For next layer.
  }

  // Final projection to vocab
  auto w_logits = CreateLargeWeight("w_logits",
                                    dtype, {hidden_dim, vocab_size});
  graph.AddInitializedTensor(w_logits);
  auto& output = graph.GetOrCreateNodeArg("logits", nullptr);
  graph.AddNode("Output_Linear", "MatMul", "", {current_arg, graph.GetNodeArg(w_logits.name())}, {&output});

  // Validate, Write as large model with external data
  auto status = graph.Resolve();
  if (!status.IsOK()) throw std::runtime_error(status.ErrorMessage());

  onnxruntime::ModelSavingOptions save_options(128);
  status = onnxruntime::Model::SaveWithExternalInitializers(
      model, model_path, external_data_path, save_options);
  if (!status.IsOK()) throw std::runtime_error(status.ErrorMessage());
}

Ort::IoBinding generate_io_binding(
    Ort::Session& session,
    std::map<std::string, std::vector<int64_t>> shape_overwrites,
    OrtAllocator* allocator) {
  Ort::IoBinding binding(session);
  auto default_allocator = Ort::AllocatorWithDefaultOptions();
  if (allocator == nullptr) {
    allocator = default_allocator;
  }
  const OrtMemoryInfo* info;
  Ort::ThrowOnError(Ort::GetApi().AllocatorGetInfo(allocator, &info));
  Ort::MemoryInfo mem_info(info->name, info->alloc_type, info->device.Id(), info->mem_type);

  for (int input_idx = 0; input_idx < int(session.GetInputCount()); ++input_idx) {
    auto input_name = session.GetInputNameAllocated(input_idx, Ort::AllocatorWithDefaultOptions());
    auto full_tensor_info = session.GetInputTypeInfo(input_idx);
    auto tensor_info = full_tensor_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto type = tensor_info.GetElementType();
    if (shape_overwrites.find(input_name.get()) == shape_overwrites.end()) {
      for (auto& v : shape) {
        if (v == -1) {
          v = 1;
        }
      }
    } else {
      shape = shape_overwrites[input_name.get()];
    }
    auto input_value = Ort::Value::CreateTensor(allocator,
                                                shape.data(),
                                                shape.size(),
                                                type);
    binding.BindInput(input_name.get(), input_value);
  }

  for (int output_idx = 0; output_idx < int(session.GetOutputCount()); ++output_idx) {
    auto output_name = session.GetOutputNameAllocated(output_idx, Ort::AllocatorWithDefaultOptions());
    binding.BindOutput(output_name.get(), mem_info);
  }
  return binding;
}

}  // namespace test
}  // namespace onnxruntime
