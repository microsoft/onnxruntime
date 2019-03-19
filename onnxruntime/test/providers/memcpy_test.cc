// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "../framework/test_utils.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include <core/framework/session_state_initializer.h>
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/utils.h"
#include "core/framework/path_lib.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace onnxruntime {
namespace {
void PutAllNodesOnOneProvider(Graph& graph, const std::string& provider_type) {
  for (Node& n : graph.Nodes()) {
    n.SetExecutionProviderType(provider_type);
  }
}
}  // namespace
TEST(MemcpyTest, copy1) {
  ExecutionProviders execution_providers;
  CPUExecutionProviderInfo epi;
  auto st = execution_providers.Add(onnxruntime::kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(epi));
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  SessionState s{execution_providers};
  s.SetLogger(logging::LoggingManager::DefaultLogger());
  KernelRegistryManager kernel_registry_manager;
  kernel_registry_manager.RegisterKernels(execution_providers);

  onnx::ModelProto mp;
  std::ifstream model_istream("testdata/matmul_1.pb", std::ifstream::in | std::ifstream::binary);
  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  const bool result = mp.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
  ASSERT_TRUE(result);

  Model model(mp);
  st = model.MainGraph().Resolve();
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  s.SetGraphViewer(std::make_unique<GraphViewer>(model.MainGraph()));
  PutAllNodesOnOneProvider(model.MainGraph(), onnxruntime::kCpuExecutionProvider);
  SessionStateInitializer session_initializer{ORT_TSTR(""), model.MainGraph(), s, execution_providers,
                                              kernel_registry_manager};
  st = session_initializer.CreatePlan(nullptr, {}, true);
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  st = session_initializer.InitializeAndSave(nullptr);
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
  AllocatorPtr allocator =
      execution_providers.Get(onnxruntime::kCpuExecutionProvider)->GetAllocator(0, OrtMemTypeDefault);
  auto* data_type = DataTypeImpl::GetType<float>();
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(data_type, TensorShape({3, 2}), allocator);
  float data[] = {1.f, 1.f, 0.f, 1.f, 1.f, 1.f};
  memcpy(p_tensor->MutableData<float>(), data, sizeof(data));
  MLValue input =
      MLValue{p_tensor.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc()};

  MLValue output;
  st = utils::CopyOneInputAcrossDevices(s, "X", input, output);
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
}
}  // namespace onnxruntime
