// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>

#include <random>
#include <unordered_map>
#include <vector>

#include <core/graph/onnx_protobuf.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>

extern OrtEnv* env;
extern const OrtApi* g_ort;

namespace {

struct DecodeBenchConfig {
  int64_t n;
  int64_t k;
  int64_t bits;
  int64_t block_size;
  int64_t accuracy_level;
};

template <typename T>
void AddTensorInitializer(ONNX_NAMESPACE::GraphProto& graph,
                          const std::string& name,
                          int32_t data_type,
                          const std::vector<int64_t>& dims,
                          const std::vector<T>& values) {
  auto* initializer = graph.add_initializer();
  initializer->set_name(name);
  initializer->set_data_type(data_type);
  for (int64_t dim : dims) {
    initializer->add_dims(dim);
  }

  initializer->set_raw_data(values.data(), values.size() * sizeof(T));
}

std::vector<DecodeBenchConfig> GetDecodeBenchConfigs() {
  // Each entry is {N, K, bits, block_size, accuracy_level} for a decode-style M=1 run.
  return {
      {5120, 3072, 4, 32, 4},
      {8192, 3072, 4, 32, 4},
      {3072, 8192, 4, 32, 4},
      {200064, 3072, 4, 32, 4},
  };
}

void AddMatMulNBitsNode(ONNX_NAMESPACE::GraphProto& graph,
                        const std::string& node_name,
                        const std::string& input_name,
                        const std::string& weight_name,
                        const std::string& scale_name,
                        const std::string& bias_name,
                        const std::string& output_name,
                        int64_t k,
                        int64_t n,
                        int64_t bits,
                        int64_t block_size,
                        int64_t accuracy_level) {
  auto* node = graph.add_node();
  node->set_name(node_name);
  node->set_op_type("MatMulNBits");
  node->set_domain("com.microsoft");
  node->add_input(input_name);
  node->add_input(weight_name);
  node->add_input(scale_name);
  node->add_input("");
  node->add_input("");
  node->add_input(bias_name);
  node->add_output(output_name);

  auto* attr_k = node->add_attribute();
  attr_k->set_name("K");
  attr_k->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_k->set_i(k);

  auto* attr_n = node->add_attribute();
  attr_n->set_name("N");
  attr_n->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_n->set_i(n);

  auto* attr_bits = node->add_attribute();
  attr_bits->set_name("bits");
  attr_bits->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_bits->set_i(bits);

  auto* attr_block = node->add_attribute();
  attr_block->set_name("block_size");
  attr_block->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_block->set_i(block_size);

  auto* attr_accuracy = node->add_attribute();
  attr_accuracy->set_name("accuracy_level");
  attr_accuracy->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_accuracy->set_i(accuracy_level);
}

std::vector<uint8_t> SerializeMatMulNBitsModel(const DecodeBenchConfig& config) {
  const int64_t k_blocks = (config.k + config.block_size - 1) / config.block_size;
  const int64_t blob_size = (config.block_size * config.bits) / 8;

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(10);

  auto* onnx_opset = model.add_opset_import();
  onnx_opset->set_domain("");
  onnx_opset->set_version(21);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("WebGpuMatMulNBitsDecode");

  auto* input = graph->add_input();
  input->set_name("A");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.k);

  auto* output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(config.n);

  std::vector<uint8_t> packed_b(static_cast<size_t>(config.n * k_blocks * blob_size), uint8_t{0x11});
  std::vector<Ort::Float16_t> scales(static_cast<size_t>(config.n * k_blocks), Ort::Float16_t(0.03125f));
  std::vector<Ort::Float16_t> bias(static_cast<size_t>(config.n), Ort::Float16_t(0.125f));

  AddTensorInitializer(*graph, "B", ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                       {config.n, k_blocks, blob_size}, packed_b);
  AddTensorInitializer(*graph, "scales", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                       {config.n, k_blocks}, scales);
  AddTensorInitializer(*graph, "bias", ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                       {config.n}, bias);

  AddMatMulNBitsNode(*graph,
                     "MatMulNBitsDecode",
                     "A",
                     "B",
                     "scales",
                     "bias",
                     "Y",
                     config.k,
                     config.n,
                     config.bits,
                     config.block_size,
                     config.accuracy_level);

  const auto serialized = model.SerializeAsString();
  return std::vector<uint8_t>(serialized.begin(), serialized.end());
}

static void BM_WebGpuMatMulNBitsDecode(benchmark::State& state) {
  const DecodeBenchConfig config{
      state.range(0),
      state.range(1),
      state.range(2),
      state.range(3),
      state.range(4),
  };

  if (config.k % config.block_size != 0) {
    state.SkipWithError("K must be divisible by block_size for this benchmark skeleton.");
    return;
  }

  std::vector<uint8_t> model_data = SerializeMatMulNBitsModel(config);

  Ort::SessionOptions session_options;
  session_options.DisableMemPattern();
  session_options.AppendExecutionProvider("WebGPU", std::unordered_map<std::string, std::string>{});

  OrtSession* raw_session = nullptr;
  OrtStatus* status = g_ort->CreateSessionFromArray(env, model_data.data(), model_data.size(), session_options, &raw_session);
  if (status != nullptr) {
    state.SkipWithError(g_ort->GetErrorMessage(status));
    g_ort->ReleaseStatus(status);
    return;
  }

  Ort::Session session{raw_session};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> input_shape{1, config.k};
  std::vector<Ort::Float16_t> activation(static_cast<size_t>(config.k));

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : activation) {
    value = Ort::Float16_t(dist(rng));
  }

  const char* input_names[] = {"A"};
  const char* output_names[] = {"Y"};

  auto input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(memory_info,
                                                                activation.data(),
                                                                activation.size(),
                                                                input_shape.data(),
                                                                input_shape.size());

  for (int i = 0; i < 10; ++i) {
    auto warmup_outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    benchmark::DoNotOptimize(warmup_outputs);
  }

  for (auto _ : state) {
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    benchmark::DoNotOptimize(outputs);
  }

  const double total_flops = 2.0 * static_cast<double>(config.n) * static_cast<double>(config.k);

  state.SetLabel("fp16_decode_bias");
  state.counters["TFLOPS"] = benchmark::Counter(
      total_flops,
      benchmark::Counter::kIsIterationInvariantRate);
}

void ApplyWebGpuMatMulNBitsDecodeArgs(benchmark::internal::Benchmark* benchmark) {
  for (const auto& config : GetDecodeBenchConfigs()) {
    benchmark->Args({config.n, config.k, config.bits, config.block_size, config.accuracy_level});
  }
}

BENCHMARK(BM_WebGpuMatMulNBitsDecode)
    ->Apply(ApplyWebGpuMatMulNBitsDecodeArgs)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

}  // namespace