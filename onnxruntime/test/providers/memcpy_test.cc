// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "../framework/test_utils.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"
#include "core/platform/path_lib.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "test/test_environment.h"
#include "asserts.h"

namespace onnxruntime {
namespace {
void PutAllNodesOnOneProvider(Graph& graph, const std::string& provider_type) {
  for (Node& n : graph.Nodes()) {
    n.SetExecutionProviderType(provider_type);
  }
}
}  // namespace

namespace test {
TEST(MemcpyTest, copy1) {
  concurrency::ThreadPool tp(&onnxruntime::Env::Default(), ThreadOptions(), ORT_TSTR("MemcpyTest"), 2, true);

  ExecutionProviders execution_providers;
  CPUExecutionProviderInfo epi;
  auto st = execution_providers.Add(onnxruntime::kCpuExecutionProvider,
                                    std::make_unique<CPUExecutionProvider>(epi));
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

  KernelRegistryManager kernel_registry_manager;
  ASSERT_STATUS_OK(kernel_registry_manager.RegisterKernels(execution_providers));

  ONNX_NAMESPACE::ModelProto mp;
  std::ifstream model_istream("testdata/matmul_1.onnx", std::ifstream::in | std::ifstream::binary);
  st = Model::Load(model_istream, &mp);
  ASSERT_STATUS_OK(st);

  Model model(mp, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  PutAllNodesOnOneProvider(model.MainGraph(), onnxruntime::kCpuExecutionProvider);

  DataTransferManager dtm;
  profiling::Profiler profiler;

  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;

  SessionState s(model.MainGraph(), execution_providers, &tp, nullptr, dtm,
                 DefaultLoggingManager().DefaultLogger(), profiler, sess_options);

  ASSERT_STATUS_OK(s.FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager));

  AllocatorPtr allocator = s.GetAllocator(execution_providers.Get(onnxruntime::kCpuExecutionProvider)->GetOrtDeviceByMemType(OrtMemTypeDefault));
  auto* data_type = DataTypeImpl::GetType<float>();
  OrtValue input;
  Tensor::InitOrtValue(data_type, TensorShape({3, 2}), std::move(allocator), input);
  float data[] = {1.f, 1.f, 0.f, 1.f, 1.f, 1.f};
  memcpy(input.GetMutable<Tensor>()->MutableDataRaw(), data, sizeof(data));

  OrtValue output;
  st = utils::CopyOneInputAcrossDevices(s, "X", input, output);
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
}
}  // namespace test
}  // namespace onnxruntime
