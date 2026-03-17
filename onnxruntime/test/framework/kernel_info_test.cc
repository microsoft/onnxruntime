// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel.h"
#include "core/framework/external_data_loader_manager.h"
#include "core/framework/session_state.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/onnxruntime_cxx_api.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

ONNX_OPERATOR_SCHEMA(KernelInfoStringArrayAttrOp)
    .SetDoc("Test op for kernel info string-array attributes.")
    .Attr("strings_attr", "Repeated string attribute for kernel info API tests.",
          AttrType::AttributeProto_AttributeType_STRINGS, std::vector<std::string>{})
    .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

static void VerifyKernelInfoStringArrayAttribute(const std::vector<std::string>& attribute_values) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);

  onnxruntime::Model model("graph_kernel_info_string_attr", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ExecutionProviders execution_providers;
  auto tmp_cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  auto* cpu_execution_provider = tmp_cpu_execution_provider.get();
  ASSERT_STATUS_OK(execution_providers.Add(kCpuExecutionProvider, std::move(tmp_cpu_execution_provider)));

  DataTransferManager dtm;
  ExternalDataLoaderManager edlm;
  profiling::Profiler profiler;

  SessionOptions sess_options;
  sess_options.enable_mem_pattern = true;
  sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  sess_options.use_deterministic_compute = false;
  sess_options.enable_mem_reuse = true;

  SessionState session_state(graph, execution_providers, tp.get(), nullptr, dtm, edlm,
                             DefaultLoggingManager().DefaultLogger(), profiler, sess_options);

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  onnxruntime::NodeArg output_arg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);

  onnxruntime::Node& node = graph.AddNode("node_1", "KernelInfoStringArrayAttrOp", "node 1.", inputs, outputs);
  node.AddAttribute("strings_attr", gsl::make_span(attribute_values));
  ASSERT_STATUS_OK(graph.Resolve());

  auto kernel_def = KernelDefBuilder().SetName("KernelInfoStringArrayAttrOp").Provider(kCpuExecutionProvider).SinceVersion(1, 10).Build();

  OpKernelInfo kernel_info(node, *kernel_def, *cpu_execution_provider, session_state.GetConstantInitializedTensors(),
                           session_state.GetOrtValueNameIdxMap(), session_state.GetDataTransferMgr(), session_state.GetAllocators(),
                           session_state.GetSessionOptions().config_options);

  const OrtApi& ort_api = Ort::GetApi();
  OrtAllocator* allocator = nullptr;
  ASSERT_EQ(nullptr, ort_api.GetAllocatorWithDefaultOptions(&allocator));

  size_t size = 0;
  ASSERT_EQ(nullptr, ort_api.KernelInfoGetAttributeArray_string(reinterpret_cast<const OrtKernelInfo*>(&kernel_info), "strings_attr",
                                                                allocator, nullptr, &size));
  ASSERT_EQ(attribute_values.size(), size);

  char** out = nullptr;
  ASSERT_EQ(nullptr, ort_api.KernelInfoGetAttributeArray_string(reinterpret_cast<const OrtKernelInfo*>(&kernel_info), "strings_attr",
                                                                allocator, &out, &size));
  ASSERT_EQ(attribute_values.size(), size);

  if (attribute_values.empty()) {
    ASSERT_EQ(nullptr, out);
  } else {
    ASSERT_NE(nullptr, out);
    for (size_t i = 0; i < size; ++i) {
      EXPECT_STREQ(attribute_values[i].c_str(), out[i]);
      allocator->Free(allocator, out[i]);
    }
    allocator->Free(allocator, out);
  }

  Ort::ConstKernelInfo ort_kernel_info{reinterpret_cast<const OrtKernelInfo*>(&kernel_info)};
  EXPECT_EQ(attribute_values, ort_kernel_info.GetAttributes<std::string>("strings_attr"));

  OrtStatus* status = ort_api.KernelInfoGetAttributeArray_string(reinterpret_cast<const OrtKernelInfo*>(&kernel_info), "missing_attr",
                                                                 allocator, nullptr, &size);
  ASSERT_NE(nullptr, status);
  ort_api.ReleaseStatus(status);
}

TEST(KernelInfoTests, KernelInfoGetAttributeArrayString) {
  VerifyKernelInfoStringArrayAttribute({"alpha", "beta", "gamma"});
}

TEST(KernelInfoTests, KernelInfoGetAttributeArrayStringEmpty) {
  VerifyKernelInfoStringArrayAttribute({});
}

}  // namespace test
}  // namespace onnxruntime
