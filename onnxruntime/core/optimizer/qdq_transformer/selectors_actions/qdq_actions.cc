// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <utility>

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_actions.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/initializer.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/mlas/inc/mlas_q4.h"

namespace onnxruntime {
namespace QDQ {

namespace {
// Derive MatMulNBits 'bits' attribute from the DQ weight element type.
int64_t DQWeightBits(int32_t dt_weight) {
  using TensorProto = ONNX_NAMESPACE::TensorProto;
  switch (dt_weight) {
    case TensorProto::INT2:
    case TensorProto::UINT2:
      return 2;
    case TensorProto::INT4:
    case TensorProto::UINT4:
      return 4;
    case TensorProto::INT8:
    case TensorProto::UINT8:
      return 8;
    default:
      ORT_THROW("Unsupported DQ weight type for MatMulNBits fusion: ", dt_weight);
  }
}

// Whether the DQ weight type is signed (requires zero-point offset conversion).
bool IsDQWeightSigned(int32_t dt_weight) {
  using TensorProto = ONNX_NAMESPACE::TensorProto;
  return dt_weight == TensorProto::INT2 ||
         dt_weight == TensorProto::INT4 ||
         dt_weight == TensorProto::INT8;
}

// Holds transposed weight/scale/zp tensors and their TensorProtos for MatMulNBits.
// Used by both DQMatMulToMatMulNBitsAction and DQCastMatMulToMatMulNBitsAction.
struct TransposedQuantizedTensors {
  Tensor weight;
  Tensor scale;
  std::optional<Tensor> zero_point;

  ONNX_NAMESPACE::TensorProto weight_proto;
  ONNX_NAMESPACE::TensorProto scale_proto;
  std::optional<ONNX_NAMESPACE::TensorProto> zero_point_proto;
};

// Transpose DQ weight/scale/zp tensors from column-wise layout to MatMulNBits layout via MLAS.
// default_zp_name_prefix: prefix for auto-generated zero-point name when unsigned type has no explicit zp.
Status TransposeDQWeightsForMatMulNBits(
    Graph& graph,
    const Node& dq_node,
    const std::string& default_zp_name_prefix,
    concurrency::ThreadPool* intra_op_thread_pool,
    TransposedQuantizedTensors& result) {
  const auto* weight_arg = dq_node.InputDefs()[0];
  const auto* scale_arg = dq_node.InputDefs()[1];
  const auto* zp_arg = dq_node.InputDefs().size() > 2 ? dq_node.InputDefs()[2] : nullptr;
  const auto& attrs = dq_node.GetAttributes();

  const ONNX_NAMESPACE::TensorProto* weight_tensor_proto = nullptr;
  ORT_RETURN_IF_NOT(graph.GetInitializedTensor(weight_arg->Name(), weight_tensor_proto),
                    "Missing required weight: ", weight_arg->Name(), " for node: ", dq_node.Name());
  const ONNX_NAMESPACE::TensorProto* scale_tensor_proto = nullptr;
  ORT_RETURN_IF_NOT(graph.GetInitializedTensor(scale_arg->Name(), scale_tensor_proto),
                    "Missing required scale: ", scale_arg->Name(), " for node: ", dq_node.Name());
  const ONNX_NAMESPACE::TensorProto* zp_tensor_proto = nullptr;
  if (zp_arg) {
    graph.GetInitializedTensor(zp_arg->Name(), zp_tensor_proto);
  }

  auto K = weight_arg->Shape()->dim(0).dim_value();
  auto N = weight_arg->Shape()->dim(1).dim_value();
  auto block_size = attrs.at("block_size").i();
  int32_t dt_weight = weight_arg->TypeAsProto()->tensor_type().elem_type();
  auto bits = DQWeightBits(dt_weight);
  auto quant_num = (K + block_size - 1) / block_size;
  auto blob_bytes = (block_size * bits + 7) / 8;

  Initializer weight_src(graph, *weight_tensor_proto, graph.ModelPath());
  Initializer scale_src(graph, *scale_tensor_proto, graph.ModelPath());
  auto uint8_type = DataTypeImpl::TensorTypeFromONNXEnum(ONNX_NAMESPACE::TensorProto_DataType_UINT8)->GetElementType();
  auto scale_type = DataTypeImpl::TensorTypeFromONNXEnum(scale_src.data_type())->GetElementType();

  std::optional<Initializer> zp_src;
  auto cpu_allocator = CPUAllocator::DefaultInstance();

  auto weight_dst_name = graph.GenerateNodeArgName(weight_arg->Name() + "_T");
  result.weight = Tensor(uint8_type, TensorShape{N, quant_num, blob_bytes}, cpu_allocator);

  auto scale_dst_name = graph.GenerateNodeArgName(scale_arg->Name() + "_T");
  auto scale_size = (TensorShape{N, quant_num}).Size();
  result.scale = Tensor(scale_type, TensorShape{scale_size}, cpu_allocator);

  std::string zp_dst_name;
  auto zp_size = (TensorShape{N, (quant_num * bits + 7) / 8}).Size();

  if (zp_tensor_proto) {
    zp_src.emplace(graph, *zp_tensor_proto, graph.ModelPath());
    zp_dst_name = graph.GenerateNodeArgName(zp_arg->Name() + "_T");
    result.zero_point = Tensor(uint8_type, TensorShape{zp_size}, cpu_allocator);
  } else if (!IsDQWeightSigned(dt_weight)) {
    zp_dst_name = graph.GenerateNodeArgName(default_zp_name_prefix + "_zero_point_T");
    result.zero_point = Tensor(uint8_type, TensorShape{zp_size}, cpu_allocator);
    memset(result.zero_point->MutableDataRaw(), 0, result.zero_point->SizeInBytes());
  }

  // Dispatch MLAS transpose based on scale type, bits, and signedness.
  auto transpose = [&](auto* scale_data, auto* scale_dst_data) {
    using ScaleType = std::remove_pointer_t<decltype(scale_data)>;
    bool is_signed = IsDQWeightSigned(dt_weight);
    const uint8_t* src_w = weight_src.DataAsByteSpan().data();
    const uint8_t* src_zp = zp_src ? zp_src->DataAsByteSpan().data() : nullptr;
    uint8_t* dst_w = result.weight.MutableData<uint8_t>();
    uint8_t* dst_zp = result.zero_point ? result.zero_point->MutableData<uint8_t>() : nullptr;
    int K_int = static_cast<int>(K);
    int N_int = static_cast<int>(N);
    int bs_int = static_cast<int>(block_size);

    if (bits == 2) {
      if (is_signed) {
        MlasQDQTransposeBlockwiseQuantized<ScaleType, 2, true>(src_w, scale_data, src_zp, dst_w, scale_dst_data, dst_zp, true, K_int, N_int, bs_int, intra_op_thread_pool);
      } else {
        MlasQDQTransposeBlockwiseQuantized<ScaleType, 2, false>(src_w, scale_data, src_zp, dst_w, scale_dst_data, dst_zp, true, K_int, N_int, bs_int, intra_op_thread_pool);
      }
    } else if (bits == 4) {
      if (is_signed) {
        MlasQDQTransposeBlockwiseQuantized<ScaleType, 4, true>(src_w, scale_data, src_zp, dst_w, scale_dst_data, dst_zp, true, K_int, N_int, bs_int, intra_op_thread_pool);
      } else {
        MlasQDQTransposeBlockwiseQuantized<ScaleType, 4, false>(src_w, scale_data, src_zp, dst_w, scale_dst_data, dst_zp, true, K_int, N_int, bs_int, intra_op_thread_pool);
      }
    } else {
      if (is_signed) {
        MlasQDQTransposeBlockwiseQuantized<ScaleType, 8, true>(src_w, scale_data, src_zp, dst_w, scale_dst_data, dst_zp, true, K_int, N_int, bs_int, intra_op_thread_pool);
      } else {
        MlasQDQTransposeBlockwiseQuantized<ScaleType, 8, false>(src_w, scale_data, src_zp, dst_w, scale_dst_data, dst_zp, true, K_int, N_int, bs_int, intra_op_thread_pool);
      }
    }
  };

  if (scale_src.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    transpose(scale_src.data<float>(), result.scale.MutableData<float>());
  } else {
    transpose(scale_src.data<MLFloat16>(), result.scale.MutableData<MLFloat16>());
  }

  result.weight_proto = utils::TensorToTensorProto(result.weight, weight_dst_name, true);
  result.scale_proto = utils::TensorToTensorProto(result.scale, scale_dst_name, true);
  if (result.zero_point) {
    result.zero_point_proto.emplace(utils::TensorToTensorProto(*result.zero_point, zp_dst_name, true));
  }

  return Status::OK();
}
}  // namespace

namespace {
using NTO = NodesToOptimize;

// moves for replacing a node with a single DQ input with the qlinear version
std::vector<NodeAndMoveInfo> UnaryMoves() {
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq, ArgType::kInput),                           // append all inputs from dq to new node
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),  // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),  // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};

  return moves;
}

// moves for replacing a node with two DQ inputs with the qlinear version
std::vector<NodeAndMoveInfo> BinaryMoves() {
  NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq2{NTO::NodeType::kInput, 1};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq1, ArgType::kInput),                          // append all inputs from dq1 to new node
      MoveAll(dq2, ArgType::kInput),                          // append all inputs from dq2
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),  // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),  // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};                          // and use the outputs from q

  return moves;
}

// moves for replacing a node with a single variadic DQ input with the qlinear version
std::vector<NodeAndMoveInfo> VariadicMoves() {
  NTO::NodeLocation variadic_dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),  // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),  // append zp (input 2) from q
      MoveAll(variadic_dq, ArgType::kInput),                  // append all inputs from all dq nodes
      MoveAll(q, ArgType::kOutput)};                          // and use the outputs from q

  return moves;
}

// moves for replacing a node with a Conv node with DQ inputs with the qlinear version
std::vector<NodeAndMoveInfo> ConvMoves() {
  NTO::NodeLocation dq_x{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_w{NTO::NodeType::kInput, 1};
  NTO::NodeLocation dq_bias{NTO::NodeType::kInput, 2};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq_x, ArgType::kInput),                                     // append all inputs from x
      MoveAll(dq_w, ArgType::kInput),                                     // append all inputs from w
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),              // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),              // append zp (input 2) from q
      MoveAndAppend(dq_bias, ArgType::kInput, 0, ArgType::kInput, true),  // (optional) append bias
      MoveAll(q, ArgType::kOutput)};                                      // and use the outputs from q

  return moves;
}
std::vector<NodeAndMoveInfo> WhereMoves() {
  NTO::NodeLocation dq_x{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_y{NTO::NodeType::kInput, 1};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAndAppend(target, ArgType::kInput, 0, ArgType::kInput),  // move the condition to the new node
      MoveAll(dq_x, ArgType::kInput),                              // append all inputs from x
      MoveAll(dq_y, ArgType::kInput),                              // append all inputs from x
      MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),       // append scale (input 1) from q
      MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput),       // append zp (input 2) from q
      MoveAll(q, ArgType::kOutput)};
  return moves;
}
QDQReplaceWithNew SplitReplacer(bool has_split_as_input) {
  NTO::NodeLocation dq{NTO::NodeType::kInput, 0};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};
  std::vector<NodeAndMoveInfo> moves{MoveAndAppend(dq, ArgType::kInput, 0, ArgType::kInput)};

  if (has_split_as_input) {
    // Move the optional split input to the new node.
    moves.push_back(MoveAndAppend(target, ArgType::kInput, 1, ArgType::kInput, true));
  }

  moves.push_back(MoveAll(q, ArgType::kOutput));

  return QDQReplaceWithNew(kOnnxDomain, "Split", std::move(moves));
}

QDQReplaceWithNew MatMulIntToFloatReplacer() {
  NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq2{NTO::NodeType::kInput, 1};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAndAppend(dq1, ArgType::kInput, 0, ArgType::kInput),
      MoveAndAppend(dq2, ArgType::kInput, 0, ArgType::kInput),
      MoveAndAppend(dq1, ArgType::kInput, 1, ArgType::kInput),
      MoveAndAppend(dq2, ArgType::kInput, 1, ArgType::kInput),
      MoveAndAppend(dq1, ArgType::kInput, 2, ArgType::kInput),
      MoveAndAppend(dq2, ArgType::kInput, 2, ArgType::kInput),
      MoveAll(target, ArgType::kOutput)};

  return QDQReplaceWithNew(kMSDomain, "MatMulIntegerToFloat", std::move(moves));
}

struct SetOptionalZeroPoint {
  static void UpdateNodes(Graph&, const NodesToOptimize& selected_nodes);

 private:
  // We assume this function won't fail
  static const ONNX_NAMESPACE::TensorProto init_optional_zero_point_int8() {
    // guid as arbitrary name to provide a unique value
    const char* const name = "init_optional_zero_point_int8_b33fd0fa-cd7b-4b10-ae5a-df64cabfe1f8";
    std::array<uint8_t, 1> a{0};
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
    onnxruntime::utils::SetRawDataInTensorProto(tensor_proto, a.data(), sizeof(int8_t));

    return tensor_proto;
  };

  // We assume this function won't fail
  static const ONNX_NAMESPACE::TensorProto init_optional_zero_point_uint8() {
    // guid as arbitrary name to provide a unique value
    const char* const name = "init_optional_zero_point_uint8_b33f88f7-c464-43e3-8692-97ac832bb14a";
    std::array<uint8_t, 1> a{0};
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    onnxruntime::utils::SetRawDataInTensorProto(tensor_proto, a.data(), sizeof(uint8_t));
    return tensor_proto;
  };
  static ONNX_NAMESPACE::TensorProto GetOptionalZeroPointInt8() {
    static ONNX_NAMESPACE::TensorProto proto = init_optional_zero_point_int8();
    return proto;
  }
  static ONNX_NAMESPACE::TensorProto GetOptionalZeroPointUint8() {
    static ONNX_NAMESPACE::TensorProto proto = init_optional_zero_point_uint8();
    return proto;
  }
};

void SetOptionalZeroPoint::UpdateNodes(Graph& graph, const NodesToOptimize& selected_nodes) {
  const auto nodes = selected_nodes.AllNodes();
  for (Node* node_ptr : nodes) {
    if (node_ptr == nullptr) {
      continue;
    }

    Node& node = *node_ptr;

    bool is_dq = node.OpType() == DQOpName;
    bool is_q = node.OpType() == QOpName;
    if (!is_dq && !is_q) {
      continue;
    }

    std::vector<NodeArg*>& input_defs = node.MutableInputDefs();
    bool has_zp_input = input_defs.size() == 3;
    if (has_zp_input && input_defs[InputIndex::ZERO_POINT_ID]->Exists()) {
      continue;  // zero point was set. No need to fill.
    }

    bool is_default_zp_signed = false;
    if (is_dq) {
      auto input_type = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
      is_default_zp_signed = ONNX_NAMESPACE::TensorProto_DataType_INT8 == input_type;
    }

    const ONNX_NAMESPACE::TensorProto& zp_tensor_proto = is_default_zp_signed
                                                             ? GetOptionalZeroPointInt8()
                                                             : GetOptionalZeroPointUint8();

    const ONNX_NAMESPACE::TensorProto* dummy_zp_tensor_proto;
    if (!graph.GetInitializedTensor(zp_tensor_proto.name(), dummy_zp_tensor_proto)) {
      // Zero points are small, no need for external data
      graph_utils::AddInitializer(graph, zp_tensor_proto);
    }

    auto& node_arg = graph.GetOrCreateNodeArg(zp_tensor_proto.name(), nullptr);
    if (!has_zp_input) {
      input_defs.push_back(&node_arg);
    } else {
      input_defs[InputIndex::ZERO_POINT_ID] = &node_arg;
    }
  }
}

}  // namespace

Status QDQReplaceWithNew::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  SetOptionalZeroPoint::UpdateNodes(graph, selected_nodes);
  return ReplaceWithNew::Run(graph, selected_nodes);
}

#if !defined(ORT_MINIMAL_BUILD)
Status QDQReplaceWithNew::RunForSave(Graph& graph, const NodesToOptimize& selected_nodes,
                                     const SatRuntimeOptimizationSaveContext& save_context,
                                     SavedState& saved_state, bool& graph_modified) const {
  SetOptionalZeroPoint::UpdateNodes(graph, selected_nodes);
  graph_modified = true;
  return ReplaceWithNew::RunForSave(graph, selected_nodes, save_context, saved_state, graph_modified);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

UnaryReplaceWithQLinear::UnaryReplaceWithQLinear(std::string domain)
    : ReplaceWithQLinear(std::move(domain), UnaryMoves()) {
}

NodeAttributes UnaryReplaceWithQLinear::ExtraAttributes(const RuntimeState& state) const {
  const auto& target = state.selected_nodes.Target();
  NodeAttributes attr;
  if (target.OpType() == "Softmax") {
    attr["opset"] = utils::MakeAttribute(std::string("opset"), int64_t(target.SinceVersion()));
  }
  return attr;
}

BinaryReplaceWithQLinear::BinaryReplaceWithQLinear(std::string domain)
    : ReplaceWithQLinear(std::move(domain), BinaryMoves()) {
}

VariadicReplaceWithQLinear::VariadicReplaceWithQLinear(std::string domain)
    : ReplaceWithQLinear(std::move(domain), VariadicMoves()) {
}

ConvReplaceWithQLinear::ConvReplaceWithQLinear()
    : ReplaceWithQLinear(kOnnxDomain, ConvMoves()) {
}
WhereReplaceWithQLinear::WhereReplaceWithQLinear()
    : ReplaceWithQLinear(kMSDomain, WhereMoves()) {
}
MatMulReplaceWithQLinear::MatMulReplaceWithQLinear()
    : matmul_int_to_float_replacer_{MatMulIntToFloatReplacer()},
      qlinear_matmul_replacer_{kOnnxDomain} {
}

Status SplitReplaceWithQuant::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  const auto& target_node = selected_nodes.Target();
  const auto& input_defs = target_node.InputDefs();

  // The 'split' attribute became an optional input at opset 13.
  bool has_split_as_input = target_node.SinceVersion() >= 13 && input_defs.size() == 2;
  return SplitReplacer(has_split_as_input).Run(graph, selected_nodes);
}

Status MatMulReplaceWithQLinear::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  // if the output is empty there were no Q nodes selected, so replace with MatMulIntegerToFloat
  // otherwise replace with QLinearMatMul
  bool matmul_integer_to_float = selected_nodes.num_outputs == 0;
  if (matmul_integer_to_float) {
    return matmul_int_to_float_replacer_.Run(graph, selected_nodes);
  } else {
    return qlinear_matmul_replacer_.Run(graph, selected_nodes);
  }
}

DQMatMulToMatMulNBitsAction::DQMatMulToMatMulNBitsAction(
    int64_t accuracy_level,
    concurrency::ThreadPool* intra_op_thread_pool)
    : accuracy_level_{accuracy_level},
      domain_{kMSDomain},
      op_type_{"MatMulNBits"},
      value_moves_{[]() {
        NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
        return std::vector<NodeAndMoveInfo>{
            MoveAndAppend(target, ArgType::kInput, 0, ArgType::kInput),
            MoveAll(target, ArgType::kOutput)};
      }()},
      intra_op_thread_pool_{intra_op_thread_pool} {
  ORT_ENFORCE(accuracy_level_ >= 0 && accuracy_level_ <= 4, "MatMulNBits accuracy level must be between 0 and 4");
}

NodeAttributes
DQMatMulToMatMulNBitsAction::ExtraAttributes(const RuntimeState& runtime_state) const {
  NodeAttributes extra_attributes;

  const auto* dq_node = runtime_state.selected_nodes.Input(0);
  auto& attrs = dq_node->GetAttributes();
  const auto* weight_shape = dq_node->InputDefs()[0]->Shape();

  utils::SetNodeAttribute(utils::MakeAttribute("K", weight_shape->dim(0).dim_value()), extra_attributes);
  utils::SetNodeAttribute(utils::MakeAttribute("N", weight_shape->dim(1).dim_value()), extra_attributes);
  utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level_), extra_attributes);
  int32_t dt_weight = dq_node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  utils::SetNodeAttribute(utils::MakeAttribute("bits", DQWeightBits(dt_weight)), extra_attributes);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", attrs.at("block_size").i()), extra_attributes);

  return extra_attributes;
}

Status DQMatMulToMatMulNBitsAction::ProcessNewNode(Graph& graph,
                                                   const NodesToOptimize& selected_nodes,
                                                   Node& replacement_node) const {
  const auto* dq_node = selected_nodes.Input(0);

  TransposedQuantizedTensors transposed;
  ORT_RETURN_IF_ERROR(TransposeDQWeightsForMatMulNBits(
      graph, *dq_node, "fused_DQ_MatMul", intra_op_thread_pool_, transposed));

  auto& input_defs = replacement_node.MutableInputDefs();
  input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, transposed.weight_proto, std::move(transposed.weight)));
  replacement_node.MutableInputArgsCount().push_back(1);

  input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, transposed.scale_proto, std::move(transposed.scale)));
  replacement_node.MutableInputArgsCount().push_back(1);

  if (transposed.zero_point_proto) {
    input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, *transposed.zero_point_proto, std::move(*transposed.zero_point)));
    replacement_node.MutableInputArgsCount().push_back(1);
  }

  return Status::OK();
}

DQCastMatMulToMatMulNBitsAction::DQCastMatMulToMatMulNBitsAction(
    int64_t accuracy_level,
    concurrency::ThreadPool* intra_op_thread_pool)
    : accuracy_level_{accuracy_level},
      intra_op_thread_pool_{intra_op_thread_pool} {
  ORT_ENFORCE(accuracy_level_ >= 0 && accuracy_level_ <= 4, "MatMulNBits accuracy level must be between 0 and 4");
}

Status DQCastMatMulToMatMulNBitsAction::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  // Selected nodes layout (from DQCastMatMulToMatMulNBitsSelector):
  //   Input(0) = DQ node
  //   Input(1) = Cast on input B (between DQ and MatMul)
  //   Target() = MatMul node
  auto* dq_node = selected_nodes.Input(0);
  auto* cast_b_node = selected_nodes.Input(1);
  auto& matmul_node = selected_nodes.Target();

  // --- Transpose DQ weights/scales/zp via shared helper ---
  TransposedQuantizedTensors transposed;
  ORT_RETURN_IF_ERROR(TransposeDQWeightsForMatMulNBits(
      graph, *dq_node, "fused_DQ_Cast_MatMul", intra_op_thread_pool_, transposed));

  // MatMulNBits operates in the DQ scale dtype.
  // Always insert Cast on input A (to DQ dtype) and Cast on output (DQ dtype to MatMul output dtype).
  // ORT's redundant cast elimination optimizer will clean up unnecessary casts later.

  // Determine DQ output element type (e.g., fp16)
  int32_t dq_output_dtype = cast_b_node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  // Determine MatMul output element type (e.g., fp32)
  int32_t matmul_output_dtype = matmul_node.OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  const auto& dq_attrs = dq_node->GetAttributes();
  const auto* weight_arg = dq_node->InputDefs()[0];
  auto K = weight_arg->Shape()->dim(0).dim_value();
  auto N = weight_arg->Shape()->dim(1).dim_value();
  auto block_size = dq_attrs.at("block_size").i();
  int32_t dt_weight = weight_arg->TypeAsProto()->tensor_type().elem_type();
  auto bits = DQWeightBits(dt_weight);

  // --- Create fp16 NodeArg for MatMulNBits input A ---
  NodeArg* matmul_input_a = matmul_node.MutableInputDefs()[0];
  ONNX_NAMESPACE::TypeProto input_a_fp16_type;
  input_a_fp16_type.mutable_tensor_type()->set_elem_type(dq_output_dtype);
  if (matmul_input_a->Shape()) {
    *input_a_fp16_type.mutable_tensor_type()->mutable_shape() =
        matmul_input_a->TypeAsProto()->tensor_type().shape();
  }
  auto cast_a_out_name = graph.GenerateNodeArgName(matmul_node.Name() + "_input_a_cast");
  NodeArg* input_a_arg = &graph.GetOrCreateNodeArg(cast_a_out_name, &input_a_fp16_type);

  // --- Create fp16 NodeArg for MatMulNBits output ---
  ONNX_NAMESPACE::TypeProto output_fp16_type;
  output_fp16_type.mutable_tensor_type()->set_elem_type(dq_output_dtype);
  if (matmul_node.OutputDefs()[0]->Shape()) {
    *output_fp16_type.mutable_tensor_type()->mutable_shape() =
        matmul_node.OutputDefs()[0]->TypeAsProto()->tensor_type().shape();
  }
  auto mnb_out_name = graph.GenerateNodeArgName(matmul_node.Name() + "_matmulnbits_out");
  NodeArg* mnb_output_arg = &graph.GetOrCreateNodeArg(mnb_out_name, &output_fp16_type);

  // --- Create MatMulNBits node ---
  NodeAttributes attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("K", K), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("N", N), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("bits", bits), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level_), attrs);

  auto& new_node = graph.AddNode(
      graph.GenerateNodeName(matmul_node.Name() + "_MatMulNBits"),
      "MatMulNBits",
      "Fused DQ+Cast+MatMul to MatMulNBits",
      {input_a_arg},
      {mnb_output_arg},
      &attrs,
      kMSDomain);

  const auto& target_provider = matmul_node.GetExecutionProviderType();
  new_node.SetExecutionProviderType(target_provider.empty() ? kCpuExecutionProvider : target_provider);

  // Add transposed weight, scale, zp to inputs
  auto& input_defs = new_node.MutableInputDefs();
  input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, transposed.weight_proto, std::move(transposed.weight)));
  new_node.MutableInputArgsCount().push_back(1);

  input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, transposed.scale_proto, std::move(transposed.scale)));
  new_node.MutableInputArgsCount().push_back(1);

  if (transposed.zero_point_proto) {
    input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, *transposed.zero_point_proto, std::move(*transposed.zero_point)));
    new_node.MutableInputArgsCount().push_back(1);
  }

  // --- Insert Cast on input A: matmul_input_dtype -> dq_output_dtype ---
  {
    NodeAttributes cast_attrs;
    utils::SetNodeAttribute(
        utils::MakeAttribute("to", static_cast<int64_t>(dq_output_dtype)),
        cast_attrs);
    auto& cast_node = graph.AddNode(
        graph.GenerateNodeName(matmul_node.Name() + "_Cast_input_a"),
        "Cast", "",
        {matmul_input_a},
        {input_a_arg},
        &cast_attrs,
        kOnnxDomain);
    cast_node.SetExecutionProviderType(new_node.GetExecutionProviderType());
  }

  // --- Insert Cast on output: dq_output_dtype -> matmul_output_dtype ---
  {
    NodeAttributes cast_attrs;
    utils::SetNodeAttribute(
        utils::MakeAttribute("to", static_cast<int64_t>(matmul_output_dtype)),
        cast_attrs);
    auto& cast_node = graph.AddNode(
        graph.GenerateNodeName(matmul_node.Name() + "_Cast_output"),
        "Cast", "",
        {mnb_output_arg},
        {const_cast<NodeArg*>(matmul_node.OutputDefs()[0])},
        &cast_attrs,
        kOnnxDomain);
    cast_node.SetExecutionProviderType(new_node.GetExecutionProviderType());
  }

  // --- Remove original nodes ---
  auto remove_node = [&graph](Node* node) {
    if (node) {
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    }
  };

  remove_node(&matmul_node);
  remove_node(cast_b_node);
  remove_node(dq_node);

  return Status::OK();
}

static std::vector<NodeAndMoveInfo> GetGemmMoveInfo(bool does_q_node_exist) {
  NTO::NodeLocation dq_A{NTO::NodeType::kInput, 0};
  NTO::NodeLocation dq_B{NTO::NodeType::kInput, 1};
  NTO::NodeLocation dq_bias{NTO::NodeType::kInput, 2};
  NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
  NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

  std::vector<NodeAndMoveInfo> moves{
      MoveAll(dq_A, ArgType::kInput),                                            // append all inputs from DQ of A
      MoveAll(dq_B, ArgType::kInput),                                            // append all inputs from DQ of B
      MoveAndAppend(dq_bias, ArgType::kInput, 0, ArgType::kInput, true, true)};  // (optional) append bias

  if (does_q_node_exist) {
    moves.push_back(MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput));  // append scale (input 1) from Q
    moves.push_back(MoveAndAppend(q, ArgType::kInput, 2, ArgType::kInput));  // append zp (input 2) from Q
    moves.push_back(MoveAll(q, ArgType::kOutput));                           // and use the outputs from Q
  } else {
    moves.push_back(MoveAll(target, ArgType::kOutput));
  }

  return moves;
}

GemmReplaceWithQuant::GemmReplaceWithQuant()
    : qgemm_with_float_as_output_replacer_(kMSDomain, "QGemm", GetGemmMoveInfo(false)),
      qgemm_with_8bits_as_output_replacer_(kMSDomain, "QGemm", GetGemmMoveInfo(true)) {
}

Status GemmReplaceWithQuant::Run(Graph& graph, const NodesToOptimize& selected_nodes) const {
  RemoveAttrBeta(selected_nodes);
  bool is_output_float = selected_nodes.num_outputs == 0;
  if (is_output_float) {
    return qgemm_with_float_as_output_replacer_.Run(graph, selected_nodes);
  }

  return qgemm_with_8bits_as_output_replacer_.Run(graph, selected_nodes);
}

#if !defined(ORT_MINIMAL_BUILD)
Status GemmReplaceWithQuant::RunForSave(Graph& graph,
                                        const NodesToOptimize& selected_nodes,
                                        const SatRuntimeOptimizationSaveContext& save_context,
                                        SavedState& saved_state,
                                        bool& graph_modified) const {
  RemoveAttrBeta(selected_nodes);
  bool is_output_float = selected_nodes.num_outputs == 0;
  if (is_output_float) {
    return qgemm_with_float_as_output_replacer_.RunForSave(graph,
                                                           selected_nodes,
                                                           save_context,
                                                           saved_state,
                                                           graph_modified);
  }

  return qgemm_with_8bits_as_output_replacer_.RunForSave(graph,
                                                         selected_nodes,
                                                         save_context,
                                                         saved_state,
                                                         graph_modified);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace QDQ
}  // namespace onnxruntime
