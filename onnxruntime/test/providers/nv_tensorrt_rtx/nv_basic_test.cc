// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"

#include <onnxruntime_cxx_api.h>
#include <string>
#include <thread>
#include <filesystem>
#include <chrono>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {

std::string WideToUTF8(const std::wstring& wstr) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(wstr);
}

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with dynamic or non-dynamic input shape.
 * \param model_name - model name
 * \param graph_name - graph name
 * \param dims - input dimensions
 * \param add_fast_gelu - add FastGelu node which makes the whole model partition into TRT EP and CUDA EP subgraphs.
 *
 * input: "X", "Y" and "Z"
 *        you can specify input dimensions, for example (1, 3, 2), (1, 2) or (1, -1, -1)). Note: -1 means the dimension is dynamic.
 *        All three inputs have the same dimensions.
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *       Add (+ float scalar "S")
 *       /
 *     "O"
 *
 *     or
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *    FastGelu (This node will be placed on CUDA EP)
 *     /
 *     *       Add (+ float scalar "S")
 *    /
 *   "O"
 */
void CreateBaseModel(const PathString& model_name,
                     std::string graph_name,
                     std::vector<int> dims,
                     bool add_fast_gelu = false) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  for (auto dim : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  ONNX_NAMESPACE::TypeProto dyn_float_tensor;
  dyn_float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

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
  float_scalar.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& input_scalar = graph.GetOrCreateNodeArg("S", &float_scalar);
  inputs.push_back(&input_scalar);

  auto& output_arg_4 = graph.GetOrCreateNodeArg("O", &dyn_float_tensor);

  outputs.clear();
  outputs.push_back(&output_arg_4);
  graph.AddNode("node_5", "Add", "node 5.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  status = onnxruntime::Model::Save(model, model_name);
}

Ort::IoBinding generate_io_binding(Ort::Session& session, std::map<std::string, std::vector<int64_t>> shape_overwrites = {}) {
  Ort::IoBinding binding(session);
  auto allocator = Ort::AllocatorWithDefaultOptions();
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
    binding.BindOutput(output_name.get(), allocator.GetInfo());
  }
  return binding;
}

TEST(NvExecutionProviderTest, ContextEmbedAndReload) {
  PathString model_name = ORT_TSTR("nv_execution_provider_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_test_ctx.onnx");
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry("ep.context_enable", "1");
    so.AddConfigEntry("ep.context_file_path", WideToUTF8(model_name_ctx).c_str());
    so.AppendExecutionProvider("NvTensorRtRtx", {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry("ep.context_enable", "1");
    so.AppendExecutionProvider("NvTensorRtRtx", {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
}

TEST(NvExecutionProviderTest, ContextEmbedAndReloadDynamic) {
  PathString model_name = ORT_TSTR("nv_execution_provider_dyn_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_dyn_test_ctx.onnx");
  std::string graph_name = "test";
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry("ep.context_enable", "1");
    so.AddConfigEntry("ep.context_file_path", WideToUTF8(model_name_ctx).c_str());
    so.AppendExecutionProvider("NvTensorRtRtx", {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry("ep.context_enable", "1");
    so.AppendExecutionProvider("NvTensorRtRtx", {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    std::map<std::string, std::vector<int64_t>> shape_overwrites;
    shape_overwrites["X"] = {1, 5, 5};
    shape_overwrites["Y"] = {1, 5, 1};
    auto io_binding = generate_io_binding(session_object, shape_overwrites);
    session_object.Run(run_options, io_binding);
  }
}

TEST(NvExecutionProviderTest, ContextEmbedAndReloadDataDynamic) {
  PathString model_name = ORT_TSTR("nv_execution_provider_data_dyn_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_data_dyn_test_ctx.onnx");
  std::string graph_name = "test";
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims, true);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry("ep.context_enable", "1");
    so.AddConfigEntry("ep.context_file_path", WideToUTF8(model_name_ctx).c_str());
    so.AppendExecutionProvider("NvTensorRtRtx", {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry("ep.context_enable", "1");
    so.AppendExecutionProvider("NvTensorRtRtx", {});
    Ort::Session session_object(env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    std::map<std::string, std::vector<int64_t>> shape_overwrites;
    shape_overwrites["X"] = {1, 5, 5};
    shape_overwrites["Y"] = {1, 5, 5};
    auto io_binding = generate_io_binding(session_object, shape_overwrites);
    session_object.Run(run_options, io_binding);
  }
}

}  // namespace test
}  // namespace onnxruntime
