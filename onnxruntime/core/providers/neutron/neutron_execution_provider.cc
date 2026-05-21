// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

/* Include `provider_api.h` first to avoid funny inclusion issues */
// #include "core/providers/shared_library/provider_api.h"
/* other headers */
#include "core/providers/neutron/neutron_execution_provider.h"
#include "core/providers/neutron/neutron_allocator.h"
#include "core/framework/kernel_registry.h"

#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_fwd.h"
#include "core/framework/compute_capability.h"

using namespace onnxruntime::common;

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace

namespace onnxruntime {

namespace neutron {

std::shared_ptr<NeutronStackAllocator> neutronAlloc(new NeutronStackAllocator());

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, int8_t,
                                            QLinearMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t,
                                            QLinearMatMul);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, uint8_t, MatMulIntegerToFloat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, int8_t, MatMulIntegerToFloat);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, uint8_t, QGemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, int8_t, QGemm);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t, MatMulInteger);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kMSDomain, 1, float, MatMulNBits);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kNeutronExecutionProvider, kNeutronDomain, 1, int8_t, NeutronGraph);

static Status RegisterNeutronKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kOnnxDomain, 10, int8_t, QLinearMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t, QLinearMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kMSDomain, 1, uint8_t, MatMulIntegerToFloat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kMSDomain, 1, int8_t, MatMulIntegerToFloat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kMSDomain, 1, uint8_t, QGemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kMSDomain, 1, int8_t, QGemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kOnnxDomain, 10, uint8_t, MatMulInteger)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kMSDomain, 1, float, MatMulNBits)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kNeutronExecutionProvider, kNeutronDomain, 1, int8_t, NeutronGraph)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}
}  // namespace neutron

NeutronExecutionProvider::NeutronExecutionProvider(NeutronProviderOptions neutron_options)
    : IExecutionProvider(onnxruntime::kNeutronExecutionProvider),
      neutron_options_(neutron_options) {
  // Initialize the neutron driver library
  NeutronError err = neutronInit();
  if (err != ENONE) {
    neutron_state_ = NEUTRON_STATE::FAILED;
    LOGS_DEFAULT(WARNING) << "Neutron hardware init failed!!! All nodes will be assigned to CPU.";
    return;
  }

  if (!neutron_options_.neutron_op_only) {
    bool success = onnxruntime::neutron::neutronAlloc->Init();
    if (success) {
      neutron_state_ = NEUTRON_STATE::OK;
      return;
    }
  }

  neutron_state_ = NEUTRON_STATE::OP_ONLY;
  LOGS_DEFAULT(WARNING) << "Only NeutronGraph op will be assigned to NPU.";
}

NeutronExecutionProvider::~NeutronExecutionProvider() {}

/* Utils */

KernelRegistryAndStatus GetNeutronKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = ::onnxruntime::neutron::RegisterNeutronKernels(*ret.kernel_registry);
  return ret;
}

std::vector<std::unique_ptr<ComputeCapability>>
NeutronExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                        const IKernelLookup&,
                                        const GraphOptimizerRegistry&,
                                        IResourceAccountant*) const {
  InlinedVector<NodeIndex> candidates;
  const auto& inits = graph.GetAllInitializedTensors();

  if (neutron_state_ == NEUTRON_STATE::FAILED) {
    return std::vector<std::unique_ptr<ComputeCapability>>();
  }

  auto GetBIndex = [](std::string op_type) {
    if ("QLinearMatMul" == op_type ||
        "QGemm" == op_type) {
      return 3;
    }
    return 1;
  };
  auto GetTransB = [](const onnxruntime::Node* node) {
    const auto& attrs = node->GetAttributes();
    auto it = attrs.find("transB");

    if (it != attrs.end()) {
      const auto& attr = it->second;
      if (attr.has_i()) {
        return attr.i() == 1;
      }
    }
    return false;
  };

  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      continue;
    }
    const auto& input_defs = node.InputDefs();

    if ("NeutronGraph" == node.OpType()) {
      candidates.push_back(node.Index());
    }
    if (neutron_state_ == NEUTRON_STATE::OP_ONLY) {
      continue;
    }

    if ("MatMulInteger" == node.OpType() ||
               "MatMulIntegerToFloat" == node.OpType() ||
               "QLinearMatMul" == node.OpType() ||
               "QGemm" == node.OpType()) {
      const auto* b_input = input_defs[GetBIndex(node.OpType())];
      const auto* b_shape = b_input->Shape();
      bool trans_b = GetTransB(&node);
      auto rows_index = trans_b ? b_shape->dim_size() - 2 : b_shape->dim_size() - 1;
      auto b_rows = b_shape->dim(rows_index).dim_value();
      ;
      if (b_rows % 16 || (b_rows * 16 >= 1024 * 1024) || inits.count(b_input->Name()) == 0) {
        LOGS_DEFAULT(INFO) << "NeutronEP: " << node.OpType() << " ("
                           << node.Name() << ") not supported, invalid B rows.";
      } else {
        candidates.push_back(node.Index());
      }
    } else if ("MatMulNBits" == node.OpType()) {
      const auto& attributes = node.GetAttributes();
      int64_t K = SafeInt<int64_t>(attributes.at("K").i());
      int64_t N = SafeInt<int64_t>(attributes.at("N").i());

      if (K % 16 != 0 || N % 128 != 0) {
        LOGS_DEFAULT(INFO) << "NeutronEP: MatMulNBits (" << node.Name() << ") not supported, invalid K or N.";
      } else if (input_defs.size() > 3 && input_defs[3]->Exists()) {
        LOGS_DEFAULT(INFO) << "NeutronEP: MatMulNBits (" << node.Name() << ") with zero-point not supported.";
      } else {
        candidates.push_back(node.Index());
      }
    }
  }

  // Exclude subgraphs that should run on CPU rather than Neutron NPU
  // (typically shape-related operations)
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

std::shared_ptr<KernelRegistry> NeutronExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = GetNeutronKernelRegistry();
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

}  // namespace onnxruntime
