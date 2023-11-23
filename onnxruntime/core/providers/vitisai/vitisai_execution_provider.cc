// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/graph_utils.h"
#include "vitisai_execution_provider.h"

#include <cassert>
#include <codecvt>
#include <fstream>
#include <istream>

#include "core/common/common.h"

#include "vaip/capability.h"
#include "vaip/global_api.h"
#include "core/session/custom_ops.h"
#include "core/session/inference_session.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

constexpr const char* VITISAI = "VITISAI";

static vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model(
    const onnxruntime::GraphViewer& graph_viewer, const logging::Logger& logger, const ProviderOptions& options) {
#ifndef _WIN32
  auto model_path = graph_viewer.ModelPath().ToPathString();
#else
  using convert_t = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_t, wchar_t> strconverter;
  auto model_path = strconverter.to_bytes(graph_viewer.ModelPath().ToPathString());
#endif
  return compile_onnx_model_with_options(model_path, graph_viewer.GetGraph(), options);
}

struct MyCustomOpKernel : OpKernel {
  MyCustomOpKernel(const OpKernelInfo& info, const OrtCustomOp& op) : OpKernel(info), op_(op) {
    op_kernel_ =
        op_.CreateKernel(&op_, OrtGetApiBase()->GetApi(op_.version), reinterpret_cast<const OrtKernelInfo*>(&info));
  }

  ~MyCustomOpKernel() override { op_.KernelDestroy(op_kernel_); }

  Status Compute(OpKernelContext* ctx) const override {
    op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MyCustomOpKernel);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

VitisAIExecutionProvider::VitisAIExecutionProvider(const ProviderOptions& info)
    : IExecutionProvider{onnxruntime::kVitisAIExecutionProvider}, info_(info) {
  custom_op_domains_ = initialize_vitisai_ep();
  registry_ = std::make_shared<KernelRegistry>();
  CreateKernelRegistry();
}

void VitisAIExecutionProvider::CreateKernelRegistry() {
  for (const auto& domain : custom_op_domains_) {
    for (const auto* op : domain->custom_ops_) {
      KernelDefBuilder def_builder;
      def_builder.SetName(op->GetName(op));
      def_builder.SetDomain(domain->domain_);
      def_builder.SinceVersion(1);
      if (op->version > 12) {
        auto input_count = op->GetInputTypeCount(op);
        for (auto i = 0u; i < input_count; i++) {
          def_builder.InputMemoryType(op->GetInputMemoryType(op, i), i);
        }
      }
      def_builder.Provider(onnxruntime::kVitisAIExecutionProvider);
      KernelCreateFn kernel_create_fn = [op](FuncManager&, const OpKernelInfo& info,
                                             std::unique_ptr<OpKernel>& out) -> Status {
        out = std::make_unique<MyCustomOpKernel>(info, *op);
        return Status::OK();
      };
      std::ignore = registry_->Register(def_builder, kernel_create_fn);
      vitisai_optypes_.insert(op->GetName(op));
    }
  }
}

std::shared_ptr<KernelRegistry> VitisAIExecutionProvider::GetKernelRegistry() const { return registry_; }

std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph, const IKernelLookup& /*kernel_lookup*/) const {
  if (graph.IsSubgraph()) {
    // VITIS AI EP not support sungraph. Assigned to CPU.
    return {};
  }
  if (execution_providers_) {
    // Only compiling a model once is currently supported
    return {};
  }
  execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(graph, *GetLogger(), info_));
  auto result = vaip::GetComputeCapabilityOps(graph, execution_providers_.get(), vitisai_optypes_);
  size_t index = 0u;
  for (auto& ep : **execution_providers_) {
    result.emplace_back(vaip::XirSubgraphToComputeCapability1(graph, ep.get(), index));
    index = index + 1;
  }
  return result;
}

common::Status VitisAIExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    NodeComputeInfo compute_info;
    const onnx::AttributeProto* attr = graph_utils::GetNodeAttribute(fused_node_graph.fused_node, "index");
    assert(attr != nullptr);
    size_t index = (size_t)attr->i();
    compute_info.create_state_func = [this, index](ComputeContext* context, FunctionState* state) {
      auto* p = (**this->execution_providers_)[index]->compile().release();
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        delete reinterpret_cast<vaip_core::CustomOp*>(state);
      }
    };
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      reinterpret_cast<vaip_core::CustomOp*>(state)->Compute(api, context);
      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime
