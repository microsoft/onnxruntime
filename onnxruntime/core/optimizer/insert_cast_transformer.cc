// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/insert_cast_transformer.h"
#include "core/framework/data_types.h"
#include "core/graph/graph_utils.h"
#include "core/mlas/inc/mlas.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
template <typename T>
static bool IsTensorOfType(const NodeArg& node_arg) {
  const auto* type_proto = node_arg.TypeAsProto();
  return node_arg.Exists() &&
         type_proto != nullptr &&
         DataTypeImpl::TypeFromProto(*type_proto) == DataTypeImpl::GetTensorType<T>();
}

template <typename T, typename NodeArgs>
static bool HasTensorArgOfType(const NodeArgs& node_args) {
  return std::any_of(node_args.cbegin(), node_args.cend(),
                     [](const NodeArg* node_arg) {
                       return node_arg != nullptr && IsTensorOfType<T>(*node_arg);
                     });
}

static bool IsMLFloat16Tensor(const NodeArg& node_arg) {
  return IsTensorOfType<MLFloat16>(node_arg);
}

static bool HasCpuFloat32FallbackKernel(
    const onnxruntime::Node& node,
    const InlinedVector<gsl::not_null<const KernelRegistry*>>& cpu_kernel_registries,
    const logging::Logger& logger);

bool InsertCastTransformer::NeedInsertCast(const onnxruntime::Node* node, const onnxruntime::NodeArg* input,
                                           const logging::Logger& logger) const {
  // Returns true when this input is an fp16 input to an unassigned node that is eligible
  // for the cast-to-fp32 fallback path.
  //
  // Nodes with subgraphs are excluded because rewriting explicit and implicit subgraph
  // inputs safely requires additional checks of the subgraph boundaries and contents.
  return node->GetExecutionProviderType().empty() &&
         !node->ContainsSubgraph() &&
         IsMLFloat16Tensor(*input) &&
         HasCpuFloat32FallbackKernel(*node, cpu_kernel_registries_, logger);
}

static bool HasFp16IO(const onnxruntime::Node& node) {
  return HasTensorArgOfType<MLFloat16>(node.InputDefs()) ||
         HasTensorArgOfType<MLFloat16>(node.OutputDefs());
}

static bool HasFp16Input(const onnxruntime::Node& node) {
  return HasTensorArgOfType<MLFloat16>(node.InputDefs());
}

static bool IsCpuFp16OptInPolicyOp(const onnxruntime::Node& node) {
  // Standard-domain ops whose CPU fp16 kernels are governed by session.enable_cpu_fp16
  // and the fp32 fallback heuristic. Existing CPU fp16 kernels are intentionally not
  // included in this opt-in/fallback policy.
  return node.Domain().empty() &&
         (node.OpType() == "MatMul" || node.OpType() == "Gemm");
}

static std::optional<int64_t> DimValue(const ONNX_NAMESPACE::TensorShapeProto* shape, int dim_idx) {
  if (!shape || dim_idx < 0 || dim_idx >= shape->dim_size()) {
    return std::nullopt;
  }

  const auto& dim = shape->dim(dim_idx);
  if (!dim.has_dim_value()) {
    return std::nullopt;
  }

  return dim.dim_value();
}

static std::optional<int64_t> ProductOfDimsExceptLast(const ONNX_NAMESPACE::TensorShapeProto* shape) {
  if (!shape || shape->dim_size() < 2) {
    return std::nullopt;
  }

  int64_t product = 1;
  for (int i = 0; i < shape->dim_size() - 1; ++i) {
    const auto dim = DimValue(shape, i);
    if (!dim || *dim < 0) {
      return std::nullopt;
    }

    if (*dim != 0 && product > std::numeric_limits<int64_t>::max() / *dim) {
      return std::nullopt;
    }
    product *= *dim;
  }

  return product;
}

static std::optional<int64_t> GetAttributeInt(const onnxruntime::Node& node,
                                              const std::string& attr_name,
                                              int64_t default_value) {
  const auto& attrs = node.GetAttributes();
  const auto attr = attrs.find(attr_name);
  if (attr == attrs.end()) {
    return default_value;
  }

  if (attr->second.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
    return std::nullopt;
  }

  return attr->second.i();
}

struct CpuFp16GemmShape {
  int64_t M;
  int64_t N;
  int64_t K;
};

static std::optional<CpuFp16GemmShape> GetMatMulShapeForCpuFp16Heuristic(const onnxruntime::Node& node) {
  const auto& inputs = node.InputDefs();
  if (inputs.size() < 2 || !inputs[0] || !inputs[1]) {
    return std::nullopt;
  }

  const auto* a_shape = inputs[0]->Shape();
  const auto* b_shape = inputs[1]->Shape();
  if (!a_shape || !b_shape || a_shape->dim_size() < 2 || b_shape->dim_size() < 2) {
    return std::nullopt;
  }

  const auto b_inner_dim = DimValue(b_shape, b_shape->dim_size() - 2);
  const auto b_size_to_inner_dim = ProductOfDimsExceptLast(b_shape);
  // Match MatMulComputeHelper: RHS shapes like [1, ..., 1, K, N] also flatten
  // the left input because the leading RHS dims are only padding.
  const bool flattens_left = a_shape->dim_size() >= b_shape->dim_size() &&
                             b_inner_dim && b_size_to_inner_dim &&
                             *b_size_to_inner_dim == *b_inner_dim;

  // Match MatMulComputeHelper: effectively 2D RHS flattens the left input, while
  // genuinely batched RHS uses the per-GEMM row count.
  auto M = flattens_left ? ProductOfDimsExceptLast(a_shape)
                         : DimValue(a_shape, a_shape->dim_size() - 2);
  auto K = DimValue(a_shape, a_shape->dim_size() - 1);
  auto N = DimValue(b_shape, b_shape->dim_size() - 1);
  if (!M || !N || !K) {
    return std::nullopt;
  }

  return CpuFp16GemmShape{*M, *N, *K};
}

struct CpuFp16MatMulRhsShape {
  int64_t N;
  int64_t K;
};

static std::optional<CpuFp16MatMulRhsShape> GetMatMulRhsShapeForCpuFp16Heuristic(const onnxruntime::Node& node) {
  const auto& inputs = node.InputDefs();
  if (inputs.size() < 2 || !inputs[1]) {
    return std::nullopt;
  }

  const auto* b_shape = inputs[1]->Shape();
  if (!b_shape || b_shape->dim_size() != 2) {
    return std::nullopt;
  }

  auto K = DimValue(b_shape, b_shape->dim_size() - 2);
  auto N = DimValue(b_shape, b_shape->dim_size() - 1);
  if (!N || !K || *N < 0 || *K < 0) {
    return std::nullopt;
  }

  if (*K != 0 && *N > std::numeric_limits<int64_t>::max() / *K) {
    return std::nullopt;
  }

  return CpuFp16MatMulRhsShape{*N, *K};
}

static std::optional<CpuFp16GemmShape> GetGemmShapeForCpuFp16Heuristic(const onnxruntime::Node& node) {
  const auto& inputs = node.InputDefs();
  if (inputs.size() < 2 || !inputs[0] || !inputs[1]) {
    return std::nullopt;
  }

  const auto* a_shape = inputs[0]->Shape();
  const auto* b_shape = inputs[1]->Shape();
  if (!a_shape || !b_shape || a_shape->dim_size() != 2 || b_shape->dim_size() != 2) {
    return std::nullopt;
  }

  const auto trans_a = GetAttributeInt(node, "transA", 0);
  const auto trans_b = GetAttributeInt(node, "transB", 0);
  if (!trans_a || !trans_b) {
    return std::nullopt;
  }

  const auto M = DimValue(a_shape, *trans_a ? 1 : 0);
  const auto K = DimValue(a_shape, *trans_a ? 0 : 1);
  const auto N = DimValue(b_shape, *trans_b ? 0 : 1);
  if (!M || !N || !K) {
    return std::nullopt;
  }

  return CpuFp16GemmShape{*M, *N, *K};
}

static bool ShouldKeepNativeCpuFp16ForMatMulOrGemm(
    const Graph& graph,
    const onnxruntime::Node& node,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG& mlas_backend_kernel_selector_config) {
  constexpr int64_t kMaxNativeFp16GemvM = 4;
  constexpr int64_t kMinNativeFp16ConstantMatMulNK = 512 * 1024;
  constexpr int64_t kMinNativeFp16NK = 1024 * 1024;

  std::optional<CpuFp16GemmShape> shape;
  if (node.OpType() == "MatMul") {
    const auto& inputs = node.InputDefs();
    const bool has_constant_rhs =
        inputs.size() > 1 && inputs[1] && graph_utils::IsInitializer(graph, inputs[1]->Name(), true);
    if (has_constant_rhs) {
      const auto rhs_shape = GetMatMulRhsShapeForCpuFp16Heuristic(node);
      if (!rhs_shape) {
        return false;
      }

      const auto nk = rhs_shape->N * rhs_shape->K;
      if (nk < kMinNativeFp16ConstantMatMulNK) {
        return false;
      }

      return MlasHalfGemmNativePackBSize(CblasNoTrans, CblasNoTrans,
                                         static_cast<size_t>(rhs_shape->N),
                                         static_cast<size_t>(rhs_shape->K),
                                         &mlas_backend_kernel_selector_config) != 0;
    }

    shape = GetMatMulShapeForCpuFp16Heuristic(node);
  } else if (node.OpType() == "Gemm") {
    shape = GetGemmShapeForCpuFp16Heuristic(node);
  }

  if (!shape || shape->M < 0 || shape->N < 0 || shape->K < 0) {
    return false;
  }

  if (shape->K != 0 && shape->N > std::numeric_limits<int64_t>::max() / shape->K) {
    return false;
  }

  const auto nk = shape->N * shape->K;

  if (node.OpType() == "MatMul") {
    return shape->M <= kMaxNativeFp16GemvM &&
           nk >= kMinNativeFp16NK &&
           MlasHGemmSupported(CblasNoTrans, CblasNoTrans);
  }

  if (shape->M > kMaxNativeFp16GemvM) {
    return false;
  }

  return nk >= kMinNativeFp16NK;
}

static const IKernelTypeStrResolver& GetInsertCastKernelTypeStrResolver() {
#if !defined(ORT_MINIMAL_BUILD)
  static const OpSchemaKernelTypeStrResolver resolver;
#else
  static const KernelTypeStrResolver resolver;
#endif
  return resolver;
}

static bool BuildTypeConstraintMapForNode(const onnxruntime::Node& node,
                                          bool replace_fp16_with_float,
                                          InlinedHashMap<std::string, MLDataType>& type_constraint_map) {
  // Build the type-constraint map that kernel lookup uses for this node.
  //
  // ONNX kernel lookup is based on the operator schema's type variables (e.g. T, T1, T2)
  // rather than directly on individual NodeArg names. For example, a schema may say that
  // both inputs and the output are of type "T". To ask "does CPU have a kernel for this
  // node as currently typed?" or "does CPU have a float32 fallback for this fp16 node?",
  // we first need to resolve those schema type variables to concrete MLDataType values.
  //
  // When replace_fp16_with_float is false we record the node's current types as-is.
  // When it is true we rewrite any float16 tensors to float in the constructed map so
  // we can ask whether a valid float32 fallback kernel exists for the same operator.
  const auto* schema = node.Op();
  if (!schema) {
    return false;
  }

  const TypeConstraintMap& type_schema = schema->typeConstraintMap();
  type_constraint_map.reserve(type_schema.size());

  const auto SetTypeConstraint = [&](const std::string& type_str, const NodeArg* def) {
    if (!def || !def->Exists()) {
      return;
    }

    TypeConstraintMap::const_iterator it = type_schema.find(type_str);
    if (it == type_schema.end()) {
      return;
    }

    auto type = DataTypeImpl::TypeFromProto(*(def->TypeAsProto()));
    if (replace_fp16_with_float && type == DataTypeImpl::GetTensorType<MLFloat16>()) {
      type = DataTypeImpl::GetTensorType<float>();
    }

    type_constraint_map[type_str] = type;
  };

  const auto& input_arg_counts = node.InputArgCount();
  const auto& input_defs = node.InputDefs();
  const auto& formal_inputs = schema->inputs();
  const size_t num_inputs = std::min(formal_inputs.size(), input_arg_counts.size());
  int input_idx_start = 0;
  for (size_t formal_idx = 0;
       formal_idx < num_inputs;
       input_idx_start += input_arg_counts[formal_idx], formal_idx++) {
    const auto& type_str = formal_inputs[formal_idx].GetTypeStr();
    // Variadic formal parameters can map to multiple actual inputs. For current CPU fp16
    // preservation/fallback decisions we only need one concrete binding for the schema type
    // variable, so we take the first existing actual input for that formal parameter.
    for (int input_idx = 0; input_idx < input_arg_counts[formal_idx]; input_idx++) {
      const size_t idx = static_cast<size_t>(input_idx_start) + static_cast<size_t>(input_idx);
      ORT_ENFORCE(idx < input_defs.size());
      const NodeArg* input_def = input_defs[idx];
      if (!input_def || !input_def->Exists()) {
        continue;
      }

      SetTypeConstraint(type_str, input_def);
      break;
    }
  }

  const auto& output_defs = node.OutputDefs();
  const auto& formal_outputs = schema->outputs();
  const size_t num_outputs = std::min(formal_outputs.size(), output_defs.size());
  for (size_t idx = 0; idx < num_outputs; idx++) {
    const auto& type_str = formal_outputs[idx].GetTypeStr();
    SetTypeConstraint(type_str, output_defs[idx]);
  }

  return true;
}

static bool HasCpuKernelForCurrentTypes(
    const onnxruntime::Node& node,
    const InlinedVector<gsl::not_null<const KernelRegistry*>>& cpu_kernel_registries,
    const logging::Logger& logger) {
  const auto& resolver = GetInsertCastKernelTypeStrResolver();
  for (const KernelRegistry* cpu_kernel_registry : cpu_kernel_registries) {
    if (KernelRegistry::HasImplementationOf(*cpu_kernel_registry, node, kCpuExecutionProvider, resolver, logger)) {
      return true;
    }
  }

  return false;
}

static bool HasCpuFloat32FallbackKernel(
    const onnxruntime::Node& node,
    const InlinedVector<gsl::not_null<const KernelRegistry*>>& cpu_kernel_registries,
    const logging::Logger& logger) {
  InlinedHashMap<std::string, MLDataType> type_constraint_map;
  if (!BuildTypeConstraintMapForNode(node, true, type_constraint_map)) {
    return false;
  }

  for (const KernelRegistry* cpu_kernel_registry : cpu_kernel_registries) {
    const KernelCreateInfo* kernel_create_info{};
    const auto lookup_status = cpu_kernel_registry->TryFindKernel(
        kCpuExecutionProvider, node.OpType(), node.Domain(),
        node.SinceVersion(), type_constraint_map, logger, &kernel_create_info);
    if (lookup_status.IsOK() && kernel_create_info != nullptr) {
      return true;
    }
  }

  return false;
}

onnxruntime::NodeArg* AddCastNode(onnxruntime::Graph& graph,
                                  onnxruntime::NodeArg* old_arg,
                                  TypeProto* new_type,
                                  bool new_on_input,
                                  int64_t to_type,
                                  onnxruntime::ProviderType providerType) {
  // insert cast op to cast input
  std::string node_name = graph.GenerateNodeName("InsertedPrecisionFreeCast_" + old_arg->Name());

  auto* new_arg = &graph.GetOrCreateNodeArg(node_name, new_type);

  std::vector<onnxruntime::NodeArg*> input_defs = {new_on_input ? new_arg : old_arg};
  std::vector<onnxruntime::NodeArg*> output_defs = {new_on_input ? old_arg : new_arg};

  auto& cast_node = graph.AddNode(node_name, "Cast", "cast node to cast from float16 to float32 on cpu",
                                  input_defs, output_defs);
  cast_node.AddAttribute("to", to_type);
  cast_node.SetExecutionProviderType(providerType);
  return new_arg;
}

// check if the node has an fp16 input but was not able to be assigned an execution provider.
// we will need to add casts to/from fp32 around the node for it to be executed using the CPU EP.
static bool NodeNeedsInputCastToFp32(const onnxruntime::Node& node,
                                     const InlinedVector<gsl::not_null<const KernelRegistry*>>& cpu_kernel_registries,
                                     const logging::Logger& logger) {
  bool not_assigned = node.GetExecutionProviderType().empty();

  if (not_assigned) {
    return HasFp16Input(node) && HasCpuFloat32FallbackKernel(node, cpu_kernel_registries, logger);
  }

  return false;
}

static bool CpuAssignedFp16NodeNeedsFallbackCast(
    const onnxruntime::Node& node,
    const InlinedVector<gsl::not_null<const KernelRegistry*>>& cpu_kernel_registries,
    const logging::Logger& logger) {
  return node.GetExecutionProviderType() == kCpuExecutionProvider &&
         !node.ContainsSubgraph() &&
         HasFp16IO(node) &&
         !HasCpuKernelForCurrentTypes(node, cpu_kernel_registries, logger) &&
         HasCpuFloat32FallbackKernel(node, cpu_kernel_registries, logger);
}

// Detect an isolated node that is able to process fp16 data but is between other nodes that have fp16 inputs
// but will need a Cast inserted to enable them to run.
//
// Say we have 3 nodes in the middle of a graph that all have fp16 inputs.
//
// -> NodeA -> NodeB -> NodeC ->
//
// NodeA and NodeC have no kernel that can handle fp16 data (no execution provider assigned).
//   e.g. 'Add' does not have an fp16 kernel
// NodeB has a kernel that can process fp16 data (assigned to CPU EP).
//
// By default, we would insert Cast to/from fp32 around NodeA and NodeC as all operators have an fp32 kernel.
//
// i.e. -> CastToFp32 -> NodeA -> CastToFp16 -> NodeB -> CastToFp32 -> NodeC -> CastToFp16
//
// We can avoid the casts around NodeB if we also force that to run using fp32 data.
//
// Detect this scenario by checking the input and output edges of the node for fp16 values to that are coming from or
// going to a node that will need a Cast.
//
// Return true if all the fp16 inputs and outputs are connected to nodes that will be cast to fp32.
static bool IsIsolatedFp16NodeOnCpu(const onnxruntime::Node& node, onnxruntime::Graph& graph,
                                    const InlinedVector<gsl::not_null<const KernelRegistry*>>& cpu_kernel_registries,
                                    const logging::Logger& logger) {
  // we can check if it's an isolated fp16 node
  // if node has input coming from other nodes (only consuming graph inputs or initializers if it doesn't),
  //    does not have a subgraph (would have to alter subgraph inputs if we cast the input to this node),
  //    does not produce a graph output (node must produce fp16 output for the graph output),
  //    and is assigned to the CPU EP (we have fp32 implementations of all kernels so forcing to fp32 is safe)
  if (node.GetInputEdgesCount() > 0 &&
      !node.ContainsSubgraph() &&
      !graph.NodeProducesGraphOutput(node) &&
      node.GetExecutionProviderType() == kCpuExecutionProvider) {
    //
    // Three tasks here:
    // 1. make sure all tensor(float16) inputs and first output coming from or
    //    going to nodes that will be cast to fp32
    // 2. check the current node is float16 node.
    // 3. check the current node has a float32 implementation
    // Only return true when all three are satisfied
    //
    const auto* schema = node.Op();
    if (!schema) {
      // no way to know whether it is safe to convert this to fp32, give up
      return false;
    }

    const TypeConstraintMap& type_schema = schema->typeConstraintMap();
    InlinedHashMap<std::string, MLDataType> type_constraint_map;
    type_constraint_map.reserve(type_schema.size());

    // For each formal parameters, there might be 0-n
    // actual inputs, this makes it very tricky to find out which
    // actual input should map to which formal parameter

    const auto& input_arg_counts = node.InputArgCount();
    const auto& input_defs = node.InputDefs();
    const auto& formal_inputs = schema->inputs();
    const size_t num_inputs = std::min(formal_inputs.size(), input_arg_counts.size());

    InlinedHashSet<int> fp16_args;
    int input_idx_start = 0;
    for (size_t formal_idx = 0;
         formal_idx < num_inputs;
         input_idx_start += input_arg_counts[formal_idx], formal_idx++) {
      const auto& type_str = formal_inputs[formal_idx].GetTypeStr();
      TypeConstraintMap::const_iterator it = type_schema.find(type_str);
      if (it == type_schema.end()) {
        // Don't care about parameter that does not have a type constraint.
        continue;
      }

      // type_str is like T, T1 or T2 ...
      for (int input_idx = 0; input_idx < input_arg_counts[formal_idx]; input_idx++) {
        const size_t idx = static_cast<size_t>(input_idx_start) + static_cast<size_t>(input_idx);
        ORT_ENFORCE(idx < input_defs.size());
        const NodeArg* input_def = input_defs[idx];
        if (!input_def || !input_def->Exists()) {
          continue;
        }
        if (IsMLFloat16Tensor(*input_def)) {
          fp16_args.emplace(static_cast<int>(idx));
          type_constraint_map[type_str] = DataTypeImpl::GetTensorType<float>();
          break;  // we don't have multiple tensors feeding into one input
        }
        type_constraint_map[type_str] = DataTypeImpl::TypeFromProto(*(input_def->TypeAsProto()));
        break;  // we don't have multiple tensors feeding into one input
      }
    }

    if (fp16_args.empty()) {
      return false;
    }

    // check if all nodes providing our fp16 input need to be cast to fp32
    for (auto input_edge = node.InputEdgesBegin(), end = node.InputEdgesEnd(); input_edge != end; ++input_edge) {
      const int arg_idx = input_edge->GetDstArgIndex();
      if (fp16_args.find(arg_idx) != fp16_args.end()) {
        // if the node producing our fp16 input does not need its input cast to fp32 we should run in fp16
        if (!NodeNeedsInputCastToFp32(input_edge->GetNode(), cpu_kernel_registries, logger)) {
          return false;
        }
      }
    }

    // if we got here all nodes providing our fp16 input/s will be cast to fp32.
    // check if the same applies to the nodes consuming our fp16 output.
    fp16_args.clear();
    const auto& output_defs = node.OutputDefs();
    const auto& formal_outputs = schema->outputs();
    const size_t num_outputs = std::min(formal_outputs.size(), output_defs.size());
    for (size_t idx = 0; idx < num_outputs; idx++) {
      const auto& type_str = formal_outputs[idx].GetTypeStr();
      TypeConstraintMap::const_iterator it = type_schema.find(type_str);
      if (it == type_schema.end()) {
        // Don't care about parameter that does not have a type constraint.
        continue;
      }

      const NodeArg* output_def = output_defs[idx];
      if (!output_def || !output_def->Exists()) {
        continue;
      }
      if (IsMLFloat16Tensor(*output_def)) {
        fp16_args.emplace((int)idx);
        type_constraint_map[type_str] = DataTypeImpl::GetTensorType<float>();
      } else {
        type_constraint_map[type_str] = DataTypeImpl::TypeFromProto(*(output_def->TypeAsProto()));
      }
    }

    if (fp16_args.empty()) {
      return false;  // no fp16 output
    }

    for (auto output_edge = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); output_edge != end; ++output_edge) {
      const int arg_idx = output_edge->GetSrcArgIndex();
      if (fp16_args.find(arg_idx) != fp16_args.end()) {
        // if the node producing our fp16 input does not need its input cast to fp32 we should run in fp16
        if (!NodeNeedsInputCastToFp32(output_edge->GetNode(), cpu_kernel_registries, logger)) {
          return false;
        }
      }
    }

    // now all fp16 inputs and outputs would have a cast
    // make sure fp32 version of the kernel is available.
    for (const KernelRegistry* cpu_kernel_registry : cpu_kernel_registries) {
      const KernelCreateInfo* kernel_create_info{};
      const auto lookup_status = cpu_kernel_registry->TryFindKernel(
          kCpuExecutionProvider, node.OpType(), node.Domain(),
          node.SinceVersion(), type_constraint_map, logger, &kernel_create_info);
      if (lookup_status.IsOK() && kernel_create_info != nullptr) {
        return true;
      }
    }
  }

  return false;
}

static Status ForceSingleNodeCPUFloat16ToFloat32(
    onnxruntime::Graph& graph,
    const InlinedVector<gsl::not_null<const KernelRegistry*>>& cpu_kernel_registries,
    const logging::Logger& logger,
    InlinedHashSet<NodeIndex>& forced_fp32_nodes) {
  for (auto& node : graph.Nodes()) {
    if (IsIsolatedFp16NodeOnCpu(node, graph, cpu_kernel_registries, logger)) {
      // Unassign the node so that NeedInsertCast will return true for it, forcing it to fp32.
      // Track the node index as well so the later broad fp16-preservation logic does not
      // immediately assign it back to CPU and undo the heuristic.
      node.SetExecutionProviderType("");
      forced_fp32_nodes.insert(node.Index());
    }
  }

  return Status::OK();
}

enum TypeGroup {
  Unknown = -1,
  Bool = 0,
  Integer = 1,
  Unsigned = 2,
  Float = 3,
};

TypeGroup GetTypeGroup(DataType type) {
  if (*type == "tensor(bool)") {
    return Bool;
  }

  if (*type == "tensor(int16)" || *type == "tensor(int32)" || *type == "tensor(int64)" || *type == "tensor(int8)") {
    return Integer;
  }

  if (*type == "tensor(uint16)" || *type == "tensor(uint32)" || *type == "tensor(uint64)" || *type == "tensor(uint8)") {
    return Unsigned;
  }

  if (*type == "tensor(bfloat16)" || *type == "tensor(double)" || *type == "tensor(float)" || *type == "tensor(float16)") {
    return Float;
  }

  return Unknown;
}

int BitLength(DataType type) {
  if (*type == "tensor(bool)") {
    return 1;
  } else if (*type == "tensor(uint8)" || *type == "tensor(int8)") {
    return 8;
  } else if (*type == "tensor(int16)" || *type == "tensor(uint16)" || *type == "tensor(bfloat16)" || *type == "tensor(float16)") {
    return 16;
  } else if (*type == "tensor(int32)" || *type == "tensor(uint32)" || *type == "tensor(float)") {
    return 32;
  } else if (*type == "tensor(int64)" || *type == "tensor(uint64)" || *type == "tensor(double)") {
    return 64;
  } else {
    return -1;
  }
}

/** Transformer to remove duplicate Cast nodes. */
class RemoveDuplicateCastTransformer : public GraphTransformer {
 public:
  RemoveDuplicateCastTransformer() : GraphTransformer("RemoveDuplicateCastTransformer") {
  }

 private:
  static bool UnsafeCast(DataType src_type, DataType dst_type, const Node& node) {
    // This is not a complete cast optimisation pass, and is more conservative than it could be.
    // For instance, certain integral -> floating point casts could be optimized but
    // this is left to an explicit cast optimisation pass.

    // The comparison with "InsertedPrecisionFreeCast_" reflects cast nodes that are inserted by InsertCastTransformer.
    // Such casts should not be considered as loss of precision - the inserted upcasts (f16 -> f32) and
    // downcasts (f32 -> f16) are inserted to support kernels when on a CPU EP without F16 support.
    auto src_type_group = GetTypeGroup(src_type);
    auto dst_type_group = GetTypeGroup(dst_type);
    if (Unknown == src_type_group || Unknown == dst_type_group) {
      return true;
    }

    // Do not remove any signed -> unsigned cast.
    if ((src_type_group != Bool && src_type_group != Unsigned) && Unsigned == dst_type_group) {
      return true;
    }

    // Do not remove any floating point -> non floating point cast.
    if (Float == src_type_group && Float != dst_type_group) {
      return true;
    }

    auto src_bit_length = BitLength(src_type);
    auto dst_bit_length = BitLength(dst_type);

    // unsigned integer -> integer cast may overflow if the destination integer is smaller or equal to the source integer.
    if (Unsigned == src_type_group && Integer == dst_type_group) {
      return dst_bit_length <= src_bit_length;
    }

    // integral -> floating cast may overflow if integer cannot be encoded in the mantissa. This check could be more precise.
    if ((Integer == src_type_group || Unsigned == src_type_group) && Float == dst_type_group) {
      return dst_bit_length <= src_bit_length;
    }

    if ((*src_type == "tensor(float16)" && *dst_type == "tensor(bfloat16)") ||
        (*src_type == "tensor(bfloat16)" && *dst_type == "tensor(float16)")) {
      return true;
    }

    return src_bit_length > dst_bit_length && (node.Name().compare(0, 26, "InsertedPrecisionFreeCast_"));
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override {
    auto output_args = graph.GetOutputs();
    InlinedHashSet<const onnxruntime::NodeArg*> graph_outputs;
    graph_outputs.reserve(output_args.size());
    graph_outputs.insert(output_args.begin(), output_args.end());
    const auto graph_outputs_end = graph_outputs.end();

    for (auto& node : graph.Nodes()) {
      bool removed = false;
      if (node.OpType() == "Cast") {
        InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
        InlinedVector<std::reference_wrapper<Node>> cast_nodes_to_keep;

        // if cast's next node is also cast:
        //     - if the next cast's output type is equal to cast's input type, remove these two casts.
        //     - otherwise, remove the first cast.
        // Below are some exception cases for this optimization:
        //     - it's for non-numeric type casting.
        //     - if the casts are for (high precision -> low precision -> high precision),
        //       since there is actual loss of precision.
        //     - if the first cast loses precision and a downstream cast targets bool,
        //       since removing it changes zero/non-zero semantics (e.g., float->int truncation
        //       before a bool cast). See https://github.com/microsoft/onnxruntime/issues/28089
        // Other cases are OK for this optimization, including below two cases,
        // which are not actual loss of precision:
        //     - (low precision -> high precision -> low precision)
        //     - (high precision -> low precision -> lower precision) when not targeting bool
        // It's possible that there are more than one casts following the first cast,
        // the first cast can be removed only when:
        //     - not providing graph output, and
        //     - all consumer nodes are cast nodes, and
        //     - for each consumer cast node, it meets above condition for this optimization.
        auto src_type = node.InputDefs()[0]->Type();
        auto dst_type = node.OutputDefs()[0]->Type();

        bool loss_precision_cast = UnsafeCast(src_type, dst_type, node);
        size_t num_children = node.GetOutputEdgesCount();

        bool inconsistent_casts = false;
        for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
          const Node& output_node(*it);
          if (output_node.OpType() == "Cast") {
            auto src_type1 = output_node.InputDefs()[0]->Type();
            auto dst_type1 = output_node.OutputDefs()[0]->Type();
            if (loss_precision_cast && UnsafeCast(dst_type1, src_type1, output_node)) {
              inconsistent_casts = true;
              break;
            }

            // Cannot remove node if it's output is also an output of the graph
            if (graph_outputs.find(output_node.OutputDefs()[0]) == graph_outputs_end &&
                src_type == dst_type1 && src_type1 == dst_type) {
              // get a mutable reference to the output node and save it
              nodes_to_remove.push_back(*graph.GetNode(output_node.Index()));
            } else {
              cast_nodes_to_keep.push_back(*graph.GetNode(output_node.Index()));
            }
          }
        }

        if (inconsistent_casts) {
          continue;
        }

        if (!nodes_to_remove.empty()) {
          if (node.GetInputEdgesCount() == 0) {
            // replacing with initializer or graph input so we just need the NodeArg for the input
            auto& input = *node.MutableInputDefs()[0];

            for (Node& node_to_remove : nodes_to_remove) {
              NodeIndex node_idx = node_to_remove.Index();

              // copy the edges so we can remove as we iterate them
              std::vector<Node::EdgeEnd> edges(node_to_remove.OutputEdgesBegin(), node_to_remove.OutputEdgesEnd());

              for (auto edge = edges.cbegin(), end = edges.cend(); edge != end; ++edge) {
                int dst_idx = edge->GetDstArgIndex();
                graph.RemoveEdge(node_idx, edge->GetNode().Index(), edge->GetSrcArgIndex(), dst_idx);

                // replace the input of the downstream nodes with the initializer
                Node& mutable_target = *graph.GetNode(edge->GetNode().Index());
                graph_utils::ReplaceNodeInput(mutable_target, dst_idx, input);
              }

              graph.RemoveNode(node_idx);
            }
          } else {
            // replace the output from the second Cast node with the input to 'node'
            const Node::EdgeEnd& input_edge = *node.InputEdgesBegin();
            Node& mutable_src_node = *graph.GetNode(input_edge.GetNode().Index());
            int replacement_idx = input_edge.GetSrcArgIndex();

            for (auto& n : nodes_to_remove) {
              Node& node_to_remove = n;
              // replace output index 0 (Cast only produces one output)
              graph_utils::ReplaceDownstreamNodeInput(graph, node_to_remove, 0, mutable_src_node, replacement_idx);

              graph.RemoveNode(node_to_remove.Index());
            }
          }

          modified = true;
        }

        // If all the child nodes are either removed or another Cast node and we're not providing graph output,
        // we can remove this node. Connect those remaining child Cast nodes to current Cast node's input.
        //
        // However, we must NOT do this if the first cast loses precision AND any kept child casts to bool.
        // Bool conversion (non-zero → true, zero → false) interacts badly with lossy intermediate casts
        // that can map non-zero values to zero, changing the semantics.
        // For example, Cast(float->int32) -> Cast(int32->bool) must not become Cast(float->bool)
        // because float->int32 truncates (e.g. -0.1 -> 0 -> false), whereas float->bool would give true.
        //
        // We also must NOT do this if any kept Cast child is on a different EP than the current node.
        // Fusing across EP boundaries can produce a node whose input type is not supported by its EP.
        // For example, Cast(int64->float, CPU) -> Cast(float->float16, WebGPU) would become
        // Cast(int64->float16, WebGPU), but WebGPU doesn't support int64 inputs.
        // See: https://github.com/microsoft/onnxruntime/issues/27291
        if (num_children > 0 && nodes_to_remove.size() + cast_nodes_to_keep.size() == num_children &&
            graph_outputs.find(node.OutputDefs()[0]) == graph_outputs_end) {
          // Check that all kept Cast children are on the same EP as the current node.
          // An empty EP means the node has not been assigned yet (e.g. pre-partitioning or in tests),
          // so we only flag a cross-EP conflict when both EPs are explicitly assigned and different.
          bool cross_ep = false;
          const auto& current_ep = node.GetExecutionProviderType();
          for (const auto& n : cast_nodes_to_keep) {
            const Node& kept_node = n;
            const auto& kept_ep = kept_node.GetExecutionProviderType();
            if (!current_ep.empty() && !kept_ep.empty() && kept_ep != current_ep) {
              cross_ep = true;
              break;
            }
          }

          if (!cross_ep) {
            // Check if any kept child Cast targets bool when the first cast is lossy.
            // Bool conversion tests for zero/non-zero, so any lossy intermediate cast
            // that maps non-zero values to zero (e.g. float truncation) changes the result.
            bool is_loss_cast_and_child_cast_to_bool = false;
            if (loss_precision_cast) {
              for (const auto& n : cast_nodes_to_keep) {
                const Node& kept_node = n;
                auto kept_dst_type = kept_node.OutputDefs()[0]->Type();
                if (kept_dst_type != nullptr && GetTypeGroup(kept_dst_type) == Bool) {
                  is_loss_cast_and_child_cast_to_bool = true;
                  break;
                }
              }
            }

            if (!is_loss_cast_and_child_cast_to_bool) {
              for (auto& n : cast_nodes_to_keep) {
                Node& cast_node_to_keep = n;
                graph.SetNodeArgType(*cast_node_to_keep.MutableInputDefs()[0], *node.InputDefs()[0]->TypeAsProto());
              }

              removed = graph_utils::RemoveNode(graph, node);
              modified = true;
            }
          }
        }
      }

      if (!removed) {
        ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
      }
    }

    return Status::OK();
  }
};

Status InsertCastTransformer::ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger) const {
  InlinedHashSet<NodeIndex> forced_fp32_nodes;
  if (force_cpu_fp32_ && !cpu_kernel_registries_.empty()) {
    ORT_RETURN_IF_ERROR(
        ForceSingleNodeCPUFloat16ToFloat32(graph, cpu_kernel_registries_, logger, forced_fp32_nodes));
  }

  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();
  TypeProto float_16_tensor_proto;
  TypeProto float_tensor_proto;
  float_16_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  float_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  std::map<onnxruntime::NodeArg*, onnxruntime::NodeArg*> input_def_updates;

  for (onnxruntime::NodeIndex i : order) {
    auto node = graph.GetNode(i);
    if (!node)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT);

    if (CpuAssignedFp16NodeNeedsFallbackCast(*node, cpu_kernel_registries_, logger)) {
      node->SetExecutionProviderType("");
    }

    if (!enable_cpu_fp16_ &&
        node->GetExecutionProviderType() == kCpuExecutionProvider &&
        IsCpuFp16OptInPolicyOp(*node) &&
        HasFp16Input(*node) &&
        !node->ContainsSubgraph()) {
      node->SetExecutionProviderType("");
    }

    if (enable_cpu_fp16_ &&
        force_cpu_fp32_ &&
        IsCpuFp16OptInPolicyOp(*node) &&
        HasFp16Input(*node) &&
        !node->ContainsSubgraph() &&
        (node->GetExecutionProviderType().empty() ||
         node->GetExecutionProviderType() == kCpuExecutionProvider) &&
        !ShouldKeepNativeCpuFp16ForMatMulOrGemm(graph, *node, mlas_backend_kernel_selector_config_) &&
        HasCpuFloat32FallbackKernel(*node, cpu_kernel_registries_, logger)) {
      // Current Arm fp16 paths are profitable for constant-RHS MatMul once native
      // packed-B is available, and for large GEMV-like shapes. Keep Gemm conservative
      // until MLAS native fp16 is consistently faster across its common shapes.
      node->SetExecutionProviderType("");
      forced_fp32_nodes.insert(node->Index());
    }

    const bool has_fp16_io = !node->ContainsSubgraph() && HasFp16IO(*node);
    const bool has_cpu_fp16_kernel =
        has_fp16_io && HasCpuKernelForCurrentTypes(*node, cpu_kernel_registries_, logger);

    if (enable_cpu_fp16_ &&
        node->GetExecutionProviderType().empty() &&
        has_cpu_fp16_kernel &&
        forced_fp32_nodes.find(node->Index()) == forced_fp32_nodes.end()) {
      // When CPU fp16 is enabled, assign any currently-unassigned fp16-capable node to CPU
      // so it is preserved in fp16 instead of being routed through the fp32 cast fallback.
      node->SetExecutionProviderType(kCpuExecutionProvider);
    }

    auto& inputs = node->MutableInputDefs();
    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;
    bool casted = false;
    for (auto input : inputs) {
      if (NeedInsertCast(node, input, logger)) {
        auto src_arg = input;
        if (input_def_updates.count(src_arg)) {
          replacement_defs[src_arg] = input_def_updates[src_arg];
        } else {
          // insert cast op to cast input
          auto dst_arg = AddCastNode(graph,
                                     src_arg,
                                     &float_tensor_proto,
                                     false,
                                     static_cast<int64_t>(TensorProto_DataType_FLOAT),
                                     // right now we only cast for cpu cases.
                                     onnxruntime::kCpuExecutionProvider);
          replacement_defs[src_arg] = dst_arg;
          input_def_updates[src_arg] = dst_arg;
        }
        casted = true;
      }
    }

    if (casted) {
      // Set current node to run on the CPU execution provider
      // Keep in mind that the EP will be empty because NeedInsertCast() already ensures that
      node->SetExecutionProviderType(kCpuExecutionProvider);

      // Some ONNX operators have an attribute `dtype` which define the output type for these operators
      // (mostly Generator ops like RandomNormal, RandomNormalLike, EyeLike, etc.).
      // Update that so that `dtype` is now Float. Otherwise there could be a mis-match between the actual
      // type of the NodeArg and the ONNX inferred type of the NodeArg and Graph Resolve() will complain.
      auto& attributes = node->GetMutableAttributes();
      auto dtype_attribute = attributes.find("dtype");

      if (dtype_attribute != attributes.end()) {
        // Simple sanity check
        ORT_ENFORCE(dtype_attribute->second.has_i(),
                    "InsertCastTransformer works on the assumption that `dtype` attribute holds an integer.");

        // Modify the dtype attribute (which defines the output type) to FLOAT if it is FLOAT16.
        if (dtype_attribute->second.i() == TensorProto_DataType_FLOAT16) {
          dtype_attribute->second.set_i(TensorProto_DataType_FLOAT);
        }
      }

      auto& outputs = node->MutableOutputDefs();
      for (auto output : outputs) {
        // TODO 1: Check if the kernel available
        // TODO 2: There is an inherent assumption that if we cast a cpu op's input from float16 to float
        // then this cpu op's output will be float (if it was inferred to be float16 previously).
        // Not sure if this is always true. Handle any corner case if it does exist.

        if (IsMLFloat16Tensor(*output)) {
          // insert cast op to cast output back to float16
          auto dst_arg = output;
          auto src_arg = AddCastNode(graph,
                                     dst_arg,
                                     &float_tensor_proto,
                                     true,
                                     static_cast<int64_t>(TensorProto_DataType_FLOAT16),
                                     onnxruntime::kCpuExecutionProvider);
          replacement_defs[dst_arg] = src_arg;
        }
      }

      node->ReplaceDefs(replacement_defs);
      modified = modified || casted;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
  }

  auto status = Status::OK();

  // if this is the main graph we've recursed into all the subgraphs and added Cast nodes.
  // run the duplicate remover now, which will call Graph::Resolve from Apply(...) and handle the main and subgraphs.
  if (graph_level == 0) {
    if (modified) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
    }

    // if we had multiple nodes in a row that were converted to fp32 we will have casts around every node.
    // Casts in between converted nodes cancel each other out and can be removed.
    // e.g.
    //      -> NodeA(fp16) -> NodeB(fp16) ->
    // After converting both to fp32
    //      -> CastToFp32 -> NodeA(fp32) -> CastToFp16 -> CastToFp32 -> NodeB(fp32) -> CastToFp16
    // After running duplicate cast removal
    //      -> CastToFp32 -> NodeA(fp32) -> NodeB(fp32) -> CastToFp16
    //
    RemoveDuplicateCastTransformer remover;
    status = remover.Apply(graph, modified, logger);
  }

  return status;
}
}  // namespace onnxruntime
