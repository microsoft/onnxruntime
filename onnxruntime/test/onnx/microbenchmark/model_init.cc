// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>
#include <core/graph/model.h>
#include <core/framework/execution_providers.h>
#include <core/framework/kernel_registry_manager.h>
#include <core/framework/session_state.h>
#include <core/framework/graph_partitioner.h>
#include <core/providers/cpu/cpu_execution_provider.h>
#ifdef USE_CUDA
#include <core/providers/cuda/cuda_execution_provider.h>
#endif
#ifdef USE_MKLDNN
#include <core/providers/mkldnn/mkldnn_execution_provider.h>
#endif
#include <core/platform/env.h>
#include <core/graph/onnx_protobuf.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
using namespace google::protobuf::io;

constexpr const char* model_str =
    "ir_version: 4\n"
    "graph {\n"
    "  node {\n"
    "    input: \"X\"\n"
    "    input: \"X\"\n"
    "    output: \"Y\"\n"
    "    op_type: \"MatMul\"\n"
    "  }\n"
    "  name: \"test-model\"\n"
    "  input {\n"
    "    name: \"X\"\n"
    "    type {\n"
    "      tensor_type {\n"
    "        elem_type: 1\n"
    "        shape {\n"
    "          dim {\n"
    "            dim_value: 2\n"
    "          }\n"
    "          dim {\n"
    "            dim_value: 2\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  output {\n"
    "    name: \"Y\"\n"
    "    type {\n"
    "      tensor_type {\n"
    "        elem_type: 1\n"
    "        shape {\n"
    "          dim {\n"
    "            dim_value: 2\n"
    "          }\n"
    "          dim {\n"
    "            dim_value: 2\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
    "opset_import {\n"
    "  version: 8\n"
    "}";

using namespace onnxruntime;

#define BM_BREAK_IF_ERROR(expr)                                                 \
  do {                                                                          \
    auto _status = (expr);                                                      \
    if ((!_status.IsOK())) state.SkipWithError(_status.ErrorMessage().c_str()); \
  } while (0)

Status CreateModelFromStr(const char* str, std::unique_ptr<Model>* out) {
  ONNX_NAMESPACE::ModelProto mp;
  if (!google::protobuf::TextFormat::ParseFromString(str, &mp)) throw std::runtime_error("load model failed");
  *out = std::make_unique<Model>(mp);
  return Status::OK();
}

Status CreateExecutionProviders(std::unique_ptr<ExecutionProviders>* ret) {
  std::unique_ptr<ExecutionProviders> execution_providers = std::make_unique<ExecutionProviders>();
#ifdef USE_CUDA
  {
    CUDAExecutionProviderInfo epi;
    ORT_RETURN_IF_ERROR(
        execution_providers->Add(onnxruntime::kCudaExecutionProvider, std::make_unique<CUDAExecutionProvider>(epi)));
  }
#endif
#ifdef USE_MKLDNN
  {
    MKLDNNExecutionProviderInfo epi;
    ORT_RETURN_IF_ERROR(execution_providers->Add(onnxruntime::kMklDnnExecutionProvider,
                                                 std::make_unique<MKLDNNExecutionProvider>(epi)));
  }
#endif
  {
    CPUExecutionProviderInfo epi;
    ORT_RETURN_IF_ERROR(
        execution_providers->Add(onnxruntime::kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(epi)));
  }
  *ret = std::move(execution_providers);
  return Status::OK();
}

Status CreateKernelRegistryManagerFromModel(std::unique_ptr<KernelRegistryManager>* ret, Model* model) {
  std::unique_ptr<ExecutionProviders> execution_providers;
  ORT_RETURN_IF_ERROR(CreateExecutionProviders(&execution_providers));
  std::unique_ptr<KernelRegistryManager> kernel_registry_manager = std::make_unique<KernelRegistryManager>();
  ORT_RETURN_IF_ERROR(kernel_registry_manager->RegisterKernels(*execution_providers));
  SessionState s{*execution_providers, true};
  s.SetLogger(logging::LoggingManager::DefaultLogger());

  ORT_RETURN_IF_ERROR(model->MainGraph().Resolve());
  s.SetGraphViewer(std::make_unique<GraphViewer>(model->MainGraph()));
  GraphPartitioner partitioner(*kernel_registry_manager, *execution_providers);
  ORT_RETURN_IF_ERROR(partitioner.Partition(model->MainGraph(), s.ExportDll(), s.GetMutableFuncMgr()));
  *ret = std::move(kernel_registry_manager);
  return Status::OK();
}

static void SearchKernelRegistry_IMPL(benchmark::State& state, Model* model) {
  std::unique_ptr<KernelRegistryManager> kernel_registry_manager;
  auto st = CreateKernelRegistryManagerFromModel(&kernel_registry_manager, model);
  if (!st.IsOK()) throw std::runtime_error("failed");
  for (auto _ : state) {
    for (const auto& n : model->MainGraph().Nodes()) {
      const KernelCreateInfo* info;
      BM_BREAK_IF_ERROR(kernel_registry_manager->SearchKernelRegistry(n, &info));
      if (info == nullptr) state.SkipWithError("Search kernel failed");
    }
  }
}

static void BM_SearchKernelRegistry_SingleNodeModel(benchmark::State& state) {
  std::unique_ptr<Model> model;
  Status st = CreateModelFromStr(model_str, &model);
  if (!st.IsOK()) throw std::runtime_error("failed");
  SearchKernelRegistry_IMPL(state, model.get());
}

BENCHMARK(BM_SearchKernelRegistry_SingleNodeModel);

static void BM_SearchKernelRegistry_RealModel_tiny_yolo(benchmark::State& state) {
  std::shared_ptr<onnxruntime::Model> model;
  auto st = onnxruntime::Model::Load("../models/opset8/test_tiny_yolov2/model.onnx", model);
  SearchKernelRegistry_IMPL(state, model.get());
}

BENCHMARK(BM_SearchKernelRegistry_RealModel_tiny_yolo);

static void BM_SearchKernelRegistry_RealModel_inception_v4(benchmark::State& state) {
  std::shared_ptr<onnxruntime::Model> model;
  auto st = onnxruntime::Model::Load("../models/opset9/tf_inception_v4/model.onnx", model);
  SearchKernelRegistry_IMPL(state, model.get());
}

BENCHMARK(BM_SearchKernelRegistry_RealModel_inception_v4);

static void BM_PartitionModel_tiny_yolo(benchmark::State& state) {
  int fd;
  Status status = Env::Default().FileOpenRd("../models/opset8/test_tiny_yolov2/model.onnx", fd);
  if (!status.IsOK()) throw std::runtime_error("open test data failed");
  auto raw_input = std::unique_ptr<ZeroCopyInputStream>(std::make_unique<FileInputStream>(fd));
  auto coded_input = std::make_unique<CodedInputStream>(raw_input.get());

  ONNX_NAMESPACE::ModelProto model_proto;
  if (!model_proto.ParseFromCodedStream(coded_input.get())) throw std::runtime_error("open test data failed");
  std::unique_ptr<ExecutionProviders> execution_providers;
  BM_BREAK_IF_ERROR(CreateExecutionProviders(&execution_providers));
  std::unique_ptr<KernelRegistryManager> kernel_registry_manager = std::make_unique<KernelRegistryManager>();
  status = kernel_registry_manager->RegisterKernels(*execution_providers);
  if (!status.IsOK()) throw std::runtime_error("RegisterKernels failed");

  for (auto _ : state) {
    state.PauseTiming();
    std::shared_ptr<onnxruntime::Model> model = std::make_shared<onnxruntime::Model>(model_proto);
    SessionState s{*execution_providers, true};
    s.SetLogger(logging::LoggingManager::DefaultLogger());
    BM_BREAK_IF_ERROR(model->MainGraph().Resolve());
    s.SetGraphViewer(std::make_unique<GraphViewer>(model->MainGraph()));
    GraphPartitioner partitioner(*kernel_registry_manager, *execution_providers);
    state.ResumeTiming();
    BM_BREAK_IF_ERROR(partitioner.Partition(model->MainGraph(), s.ExportDll(), s.GetMutableFuncMgr()));
  }
}

BENCHMARK(BM_PartitionModel_tiny_yolo);

static void BM_PartitionModel_inception_v4(benchmark::State& state) {
  int fd;
  Status status = Env::Default().FileOpenRd("../models/opset9/tf_inception_v4/model.onnx", fd);
  if (!status.IsOK()) throw std::runtime_error("open test data failed");
  auto raw_input = std::unique_ptr<ZeroCopyInputStream>(std::make_unique<FileInputStream>(fd));
  auto coded_input = std::make_unique<CodedInputStream>(raw_input.get());

  ONNX_NAMESPACE::ModelProto model_proto;
  if (!model_proto.ParseFromCodedStream(coded_input.get())) throw std::runtime_error("open test data failed");
  std::unique_ptr<ExecutionProviders> execution_providers;
  BM_BREAK_IF_ERROR(CreateExecutionProviders(&execution_providers));
  std::unique_ptr<KernelRegistryManager> kernel_registry_manager = std::make_unique<KernelRegistryManager>();
  status = kernel_registry_manager->RegisterKernels(*execution_providers);
  if (!status.IsOK()) throw std::runtime_error("RegisterKernels failed");

  for (auto _ : state) {
    state.PauseTiming();
    std::shared_ptr<onnxruntime::Model> model = std::make_shared<onnxruntime::Model>(model_proto);
    SessionState s{*execution_providers, true};
    s.SetLogger(logging::LoggingManager::DefaultLogger());
    BM_BREAK_IF_ERROR(model->MainGraph().Resolve());
    s.SetGraphViewer(std::make_unique<GraphViewer>(model->MainGraph()));
    GraphPartitioner partitioner(*kernel_registry_manager, *execution_providers);
    state.ResumeTiming();
    BM_BREAK_IF_ERROR(partitioner.Partition(model->MainGraph(), s.ExportDll(), s.GetMutableFuncMgr()));
  }
}

BENCHMARK(BM_PartitionModel_inception_v4);
