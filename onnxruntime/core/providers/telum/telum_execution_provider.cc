// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_execution_provider.h"
#include "telum_allocator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace telum {

TelumExecutionProvider::TelumExecutionProvider(const TelumExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTelumExecutionProvider},
      info_(info),
      type_(onnxruntime::kTelumExecutionProvider) {

  // Verify zDNN is available
  if (!IsZDNNAvailable()) {
    ORT_THROW("zDNN/NNPA is not available on this system. Telum EP requires IBM z16 or later.");
  }

  // Initialize zDNN library
  zdnn_init();

  // Initialize supported operators list
  // Priority 0 (P0) - Critical for transformers
  supported_ops_.insert("MatMul");
  supported_ops_.insert("Gemm");
  supported_ops_.insert("Add");
  supported_ops_.insert("Relu");
  supported_ops_.insert("Gelu");
  supported_ops_.insert("Softmax");
  supported_ops_.insert("LayerNormalization");

  // Priority 1 (P1) - Important operations
  supported_ops_.insert("Sub");
  supported_ops_.insert("Mul");
  supported_ops_.insert("Tanh");
  supported_ops_.insert("Sigmoid");
  supported_ops_.insert("Exp");
  supported_ops_.insert("Log");
  supported_ops_.insert("Sqrt");
  supported_ops_.insert("Min");
  supported_ops_.insert("Max");
  supported_ops_.insert("Div");

  // Create kernel registry
  kernel_registry_ = std::make_shared<KernelRegistry>();

  // Register graph transformers if fusion is enabled
  if (info_.enable_fusion) {
    RegisterGraphTransformers();
  }
}

TelumExecutionProvider::~TelumExecutionProvider() {
  // Cleanup resources
}

std::shared_ptr<KernelRegistry> TelumExecutionProvider::GetKernelRegistry() const {
  return kernel_registry_;
}

std::unique_ptr<IDataTransfer> TelumExecutionProvider::GetDataTransfer() const {
  // Telum EP operates on CPU memory, no special data transfer needed
  return nullptr;
}

std::vector<AllocatorPtr> TelumExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;

  // Create Telum allocator with 4K alignment
  AllocatorCreationInfo telum_allocator_info{
      [](int) { return std::make_unique<TelumAllocator>(); },
      0,  // device_id
      info_.create_arena
  };

  allocators.push_back(CreateAllocator(telum_allocator_info));

  return allocators;
}

std::vector<std::unique_ptr<ComputeCapability>>
TelumExecutionProvider::GetCapability(const GraphViewer& graph,
                                      const IKernelLookup& kernel_lookup) const {
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;

  // Iterate through all nodes in the graph
  for (const auto& node : graph.Nodes()) {
    // Check if node is supported
    if (!IsNodeSupported(node)) {
      if (info_.log_fallbacks) {
        LOGS_DEFAULT(WARNING) << "Telum EP: Node '" << node.Name()
                             << "' (op: " << node.OpType() << ") not supported. "
                             << "Reason: " << GetRejectionReason(node)
                             << ". Falling back to CPU.";
      }
      continue;
    }

    // Validate static shapes
    if (!ValidateStaticShapes(node)) {
      if (info_.log_fallbacks) {
        LOGS_DEFAULT(WARNING) << "Telum EP: Node '" << node.Name()
                             << "' has dynamic shapes. Falling back to CPU.";
      }
      continue;
    }

    // Validate data types
    if (!ValidateDataTypes(node)) {
      if (info_.log_fallbacks) {
        LOGS_DEFAULT(WARNING) << "Telum EP: Node '" << node.Name()
                             << "' has unsupported data types. Falling back to CPU.";
      }
      continue;
    }

    // Try to find a matching kernel
    const KernelCreateInfo* kernel_info = kernel_lookup.LookUpKernel(node);
    if (kernel_info == nullptr) {
      continue;
    }

    // Create compute capability for this node
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node.Index());

    auto capability = std::make_unique<ComputeCapability>(std::move(sub_graph));
    capabilities.push_back(std::move(capability));
  }

  return capabilities;
}

common::Status TelumExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {

  for (const auto& fused_node_graph : fused_nodes) {
    const Node& fused_node = fused_node_graph.fused_node;

    // Create compute function for fused node
    NodeComputeInfo compute_info;

    // Set up function state creation
    compute_info.create_state_func = [](ComputeContext* context, FunctionState* state) {
      *state = nullptr;
      return 0;
    };

    // Set up compute function
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      // This will be implemented by individual kernel implementations
      return Status::OK();
    };

    // Set up state release function
    compute_info.release_state_func = [](FunctionState state) {
      // Cleanup if needed
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

bool TelumExecutionProvider::IsNodeSupported(const Node& node) const {
  // Check if operator type is in supported list
  return IsOperatorSupported(node.OpType());
}

bool TelumExecutionProvider::ValidateStaticShapes(const Node& node) const {
  // Check all input shapes
  for (const auto* input_def : node.InputDefs()) {
    if (input_def->Shape() == nullptr) {
      return false;
    }

    const auto& shape = *input_def->Shape();
    for (const auto& dim : shape.dim()) {
      if (!dim.has_dim_value()) {
        return false;  // Dynamic dimension
      }
    }
  }

  // Check all output shapes
  for (const auto* output_def : node.OutputDefs()) {
    if (output_def->Shape() == nullptr) {
      return false;
    }

    const auto& shape = *output_def->Shape();
    for (const auto& dim : shape.dim()) {
      if (!dim.has_dim_value()) {
        return false;  // Dynamic dimension
      }
    }
  }

  return true;
}

bool TelumExecutionProvider::ValidateDataTypes(const Node& node) const {
  // Check input data types
  for (const auto* input_def : node.InputDefs()) {
    if (input_def->TypeAsProto() == nullptr) {
      return false;
    }

    const auto& type_proto = *input_def->TypeAsProto();
    if (!type_proto.has_tensor_type()) {
      return false;
    }

    int32_t elem_type = type_proto.tensor_type().elem_type();

    // Check if type is supported
    if (elem_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        elem_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
        elem_type != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      return false;
    }
  }

  return true;
}

std::string TelumExecutionProvider::GetRejectionReason(const Node& node) const {
  if (!IsOperatorSupported(node.OpType())) {
    return "Operator not supported";
  }

  if (!ValidateStaticShapes(node)) {
    return "Dynamic shapes not supported";
  }

  if (!ValidateDataTypes(node)) {
    return "Unsupported data type";
  }

  return "Unknown reason";
}

void TelumExecutionProvider::RegisterGraphTransformers() {
  // Graph transformers will be registered here
  // This will be implemented when we add fusion support
}

bool TelumExecutionProvider::IsOperatorSupported(const std::string& op_type) const {
  return supported_ops_.find(op_type) != supported_ops_.end();
}

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
