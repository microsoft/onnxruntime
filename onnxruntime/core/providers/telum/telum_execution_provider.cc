// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_execution_provider.h"
#include "telum_allocator.h"
#include "telum_kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace telum {

TelumExecutionProvider::TelumExecutionProvider(const TelumExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTelumExecutionProvider},
      info_(info) {

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

  // Register graph transformers if fusion is enabled
  if (info_.enable_fusion) {
    RegisterGraphTransformers();
  }
}

TelumExecutionProvider::~TelumExecutionProvider() {
  // Cleanup resources
}

std::shared_ptr<KernelRegistry> TelumExecutionProvider::GetKernelRegistry() const {
  return GetTelumKernelRegistry();
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
                                      const IKernelLookup& kernel_lookup,
                                      const GraphOptimizerRegistry& /*graph_optimizer_registry*/,
                                      IResourceAccountant* /*resource_accountant*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;

  // Iterate through all nodes in the graph
  for (const auto& node : graph.Nodes()) {
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

    // Check if node is supported (op + shape/attribute constraints)
    if (!IsNodeSupported(node)) {
      if (info_.log_fallbacks) {
        LOGS_DEFAULT(WARNING) << "Telum EP: Node '" << node.Name()
                             << "' (op: " << node.OpType() << ") not supported. "
                             << "Reason: " << GetRejectionReason(node)
                             << ". Falling back to CPU.";
      }
      continue;
    }

    // Try to find a matching kernel
    const KernelCreateInfo* kernel_info = kernel_lookup.LookUpKernel(node);
    if (kernel_info == nullptr) {
      if (info_.log_fallbacks) {
        LOGS_DEFAULT(WARNING) << "Telum EP: Node '" << node.Name()
                             << "' (op: " << node.OpType() << ") has no registered Telum kernel "
                             << "for this build/type/versions. Falling back to CPU.";
      }
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
  if (!IsOperatorSupported(node.OpType())) {
    return false;
  }

  // Per-op constraint checks so we don't claim nodes our kernels cannot handle.
  // NOTE: ValidateStaticShapes() has already been called in GetCapability() prior to this.
  const auto& op_type = node.OpType();

  auto get_shape = [](const NodeArg* arg, std::vector<int64_t>& out) -> bool {
    out.clear();
    if (arg == nullptr || arg->Shape() == nullptr) return false;
    for (const auto& dim : arg->Shape()->dim()) {
      if (!dim.has_dim_value()) return false;
      out.push_back(dim.dim_value());
    }
    return true;
  };

  if (op_type == "Gemm") {
    std::vector<int64_t> a_shape, b_shape;
    if (!get_shape(node.InputDefs()[0], a_shape) || !get_shape(node.InputDefs()[1], b_shape)) {
      return false;
    }
    // Telum Gemm kernel requires A/B to be 2D. (C is optional and handled in-kernel.)
    return a_shape.size() == 2 && b_shape.size() == 2;
  }

  if (op_type == "MatMul") {
    std::vector<int64_t> a_shape, b_shape;
    if (!get_shape(node.InputDefs()[0], a_shape) || !get_shape(node.InputDefs()[1], b_shape)) {
      return false;
    }
    if (a_shape.size() < 2 || b_shape.size() < 2) {
      return false;
    }

    const int64_t K_a = a_shape[a_shape.size() - 1];
    const int64_t K_b = b_shape[b_shape.size() - 2];
    if (K_a != K_b) {
      return false;
    }

    const std::vector<int64_t> a_batch(a_shape.begin(), a_shape.end() - 2);
    const std::vector<int64_t> b_batch(b_shape.begin(), b_shape.end() - 2);

    // Compute broadcasted output batch dims.
    const size_t out_rank = std::max(a_batch.size(), b_batch.size());
    std::vector<int64_t> out_batch(out_rank, 1);
    for (size_t i = 0; i < out_rank; ++i) {
      const int64_t da = (i < a_batch.size()) ? a_batch[a_batch.size() - 1 - i] : 1;
      const int64_t db = (i < b_batch.size()) ? b_batch[b_batch.size() - 1 - i] : 1;
      if (da == db) {
        out_batch[out_rank - 1 - i] = da;
      } else if (da == 1) {
        out_batch[out_rank - 1 - i] = db;
      } else if (db == 1) {
        out_batch[out_rank - 1 - i] = da;
      } else {
        return false;
      }
    }

    auto align_batch = [](const std::vector<int64_t>& batch, size_t target_rank) -> std::vector<int64_t> {
      if (batch.size() >= target_rank) return batch;
      std::vector<int64_t> aligned(target_rank - batch.size(), 1);
      aligned.insert(aligned.end(), batch.begin(), batch.end());
      return aligned;
    };
    auto all_ones = [](const std::vector<int64_t>& dims) -> bool {
      for (int64_t d : dims) if (d != 1) return false;
      return true;
    };

    const auto a_aligned = align_batch(a_batch, out_batch.size());
    const auto b_aligned = align_batch(b_batch, out_batch.size());
    const bool a_matches = (a_aligned == out_batch);
    const bool b_matches = (b_aligned == out_batch);
    const bool a_all_ones = all_ones(a_aligned);
    const bool b_all_ones = all_ones(b_aligned);

    // Same constraints as the MatMul kernel plan:
    // - no partial broadcast (either match output batch dims, or be fully unstacked/all-ones)
    if (a_matches && b_matches) return true;
    if (a_matches && b_all_ones) return true;
    if (a_all_ones && b_matches) return true;
    return false;
  }

  if (op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div" ||
      op_type == "Min" || op_type == "Max") {
    std::vector<int64_t> a_shape, b_shape;
    if (!get_shape(node.InputDefs()[0], a_shape) || !get_shape(node.InputDefs()[1], b_shape)) {
      return false;
    }
    if (a_shape != b_shape) return false;              // no broadcast for now
    if (a_shape.size() > 4) return false;              // TensorConverter only supports up to 4D
    return true;
  }

  if (op_type == "Relu" || op_type == "Gelu" || op_type == "Tanh" || op_type == "Sigmoid" ||
      op_type == "Exp" || op_type == "Log" || op_type == "Sqrt") {
    std::vector<int64_t> x_shape;
    if (!get_shape(node.InputDefs()[0], x_shape)) return false;
    if (x_shape.size() > 4) return false;
    return true;
  }

  if (op_type == "Softmax") {
    std::vector<int64_t> x_shape;
    if (!get_shape(node.InputDefs()[0], x_shape)) return false;
    if (x_shape.empty()) return false;

    int64_t axis_attr = -1;
    const auto& attrs = node.GetAttributes();
    if (auto it = attrs.find("axis"); it != attrs.end() && it->second.has_i()) {
      axis_attr = it->second.i();
    }

    const int64_t rank = static_cast<int64_t>(x_shape.size());
    int64_t axis = axis_attr;
    if (axis < 0) axis += rank;
    if (axis != rank - 1) return false;  // zDNN softmax only supports last-dim softmax via coercion

    int64_t batch = 1;
    for (size_t i = 0; i + 1 < x_shape.size(); ++i) batch *= x_shape[i];
    const int64_t vector_len = x_shape.back();

    const uint32_t max_dim = zdnn_get_nnpa_max_dim_idx_size();
    if (batch > static_cast<int64_t>(max_dim)) return false;
    if (vector_len > static_cast<int64_t>(max_dim)) return false;
    return true;
  }

  if (op_type == "LayerNormalization") {
    std::vector<int64_t> x_shape, scale_shape, bias_shape;
    if (!get_shape(node.InputDefs()[0], x_shape) || !get_shape(node.InputDefs()[1], scale_shape)) return false;
    if (x_shape.empty()) return false;

    const int64_t rank = static_cast<int64_t>(x_shape.size());
    const auto& attrs = node.GetAttributes();
    auto it = attrs.find("axis");
    if (it == attrs.end() || !it->second.has_i()) return false;

    int64_t axis = it->second.i();
    if (axis < 0) axis += rank;
    if (axis != rank - 1) return false;  // initial Telum support: last-dim only

    const int64_t C = x_shape.back();
    if (!(scale_shape.size() == 1 && scale_shape[0] == C)) return false;

    // Bias is optional
    if (node.InputDefs().size() > 2 && node.InputDefs()[2] != nullptr) {
      if (!get_shape(node.InputDefs()[2], bias_shape)) return false;
      if (!(bias_shape.size() == 1 && bias_shape[0] == C)) return false;
    }

    int64_t N = 1;
    for (size_t i = 0; i + 1 < x_shape.size(); ++i) N *= x_shape[i];
    const uint32_t max_dim = zdnn_get_nnpa_max_dim_idx_size();
    if (N > static_cast<int64_t>(max_dim)) return false;
    if (C > static_cast<int64_t>(max_dim)) return false;
    return true;
  }

  // Default: if in supported list, allow. Kernel lookup will still gate.
  return true;
}

bool TelumExecutionProvider::ValidateStaticShapes(const Node& node) const {
  // Check all input shapes
  for (const auto* input_def : node.InputDefs()) {
    if (input_def == nullptr) {
      continue;  // optional input not provided
    }
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
    if (input_def == nullptr) {
      continue;  // optional input not provided
    }
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

  if (!IsNodeSupported(node)) {
    return "Unsupported attributes or shape pattern for this operator";
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
