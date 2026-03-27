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

// Compute the effective block_size for per-tensor/per-channel DQ nodes that lack a block_size attribute.
// session_block_size: 0 = default (32), positive = explicit, -1 = min-padding heuristic.
int64_t ComputeEffectiveBlockSize(int64_t session_block_size, int64_t K) {
  // MatMulNBits CPU kernel currently only supports block_size in [16, 256] correctly.
  constexpr int64_t kMinBlockSize = 16;
  constexpr int64_t kMaxBlockSize = 256;

  if (session_block_size > 0) {
    // Explicit block_size — must be power-of-2 and within [kMinBlockSize, kMaxBlockSize].
    ORT_ENFORCE(session_block_size >= kMinBlockSize &&
                    ((session_block_size & (session_block_size - 1)) == 0),
                "Explicit qdq_matmulnbits_block_size must be a power-of-2 and >= ",
                kMinBlockSize, ", got: ", session_block_size);
    ORT_ENFORCE(session_block_size <= kMaxBlockSize,
                "Explicit qdq_matmulnbits_block_size must be <= ",
                kMaxBlockSize, ", got: ", session_block_size);
    return session_block_size;
  }

  if (session_block_size == -1) {
    // Heuristic: largest power-of-2 <= min(K, kMaxBlockSize) that minimizes padding.
    // Capped at kMaxBlockSize because CPU EP only supports block_size up to kMaxBlockSize correctly.
    // We want ceil(K / B) * B - K to be minimized (least wasted padding).
    int64_t best_bs = kMinBlockSize;
    int64_t best_padding = (((K + (kMinBlockSize - 1)) / kMinBlockSize) * kMinBlockSize) - K;
    for (int64_t bs = kMinBlockSize * 2; bs <= std::min(K, kMaxBlockSize); bs *= 2) {
      int64_t padding = (((K + bs - 1) / bs) * bs) - K;
      if (padding <= best_padding) {
        best_padding = padding;
        best_bs = bs;
      }
    }
    return best_bs;
  }

  // Default (session_block_size == 0): use 32
  return 32;
}

// Get the DQ block_size: from the attribute if blockwise, or computed for per-tensor/per-channel.
int64_t GetEffectiveBlockSize(const Node& dq_node, int64_t block_size_for_non_blockwise) {
  const auto& dq_attrs = dq_node.GetAttributes();
  const auto bs_iter = dq_attrs.find("block_size");
  if (bs_iter != dq_attrs.end() && bs_iter->second.i() > 0) {
    return bs_iter->second.i();
  }

  // Derive K from the weight input shape if available. Shape information may be missing even
  // when the weight is a constant initializer, so guard against nullptrs / unknown dims.
  int64_t K = 32;  // reasonable default consistent with ComputeEffectiveBlockSize default
  const auto* weight_arg = dq_node.InputDefs()[0];
  if (weight_arg != nullptr) {
    const auto* shape = weight_arg->Shape();
    if (shape != nullptr && shape->dim_size() > 0 && shape->dim(0).has_dim_value()) {
      K = static_cast<int64_t>(shape->dim(0).dim_value());
    }
  }

  return ComputeEffectiveBlockSize(block_size_for_non_blockwise, K);
}

// Holds transposed weight/scale/zp tensors and their TensorProtos for MatMulNBits.
// Used by DQMatMulToMatMulNBitsAction.
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
// effective_block_size: the block_size to use for MatMulNBits (may differ from DQ's block_size for per-tensor/per-channel).
Status TransposeDQWeightsForMatMulNBits(
    Graph& graph,
    const Node& dq_node,
    const std::string& default_zp_name_prefix,
    concurrency::ThreadPool* intra_op_thread_pool,
    int64_t effective_block_size,
    TransposedQuantizedTensors& result) {
  const auto* weight_arg = dq_node.InputDefs()[0];
  const auto* scale_arg = dq_node.InputDefs()[1];
  const auto* zp_arg = dq_node.InputDefs().size() > 2 ? dq_node.InputDefs()[2] : nullptr;

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

  ORT_RETURN_IF_NOT(weight_tensor_proto->dims_size() >= 2,
                    "Weight tensor for node ", dq_node.Name(), " must be at least 2D.");
  auto K = weight_tensor_proto->dims(0);
  auto N = weight_tensor_proto->dims(1);
  auto block_size = effective_block_size;
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

  // Determine if scale/zp need expansion from per-tensor/per-channel to blockwise [quant_num, N].
  const bool is_blockwise = (scale_tensor_proto->dims_size() == 2);
  std::optional<Tensor> expanded_scale;
  std::optional<Tensor> expanded_zp;

  if (!is_blockwise) {
    // Expand scale to [quant_num, N]
    expanded_scale.emplace(scale_type, TensorShape{quant_num, N}, cpu_allocator);
    bool is_per_tensor = (scale_tensor_proto->dims_size() == 0);

    auto expand_scale = [&](auto* src_data, auto* dst_data) {
      if (is_per_tensor) {
        auto val = src_data[0];
        for (int64_t i = 0; i < quant_num * N; ++i) {
          dst_data[i] = val;
        }
      } else {
        // Per-channel: scale shape [N], replicate across quant_num blocks
        for (int64_t b = 0; b < quant_num; ++b) {
          for (int64_t n = 0; n < N; ++n) {
            dst_data[b * N + n] = src_data[n];
          }
        }
      }
    };

    if (scale_src.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      expand_scale(scale_src.data<float>(), expanded_scale->MutableData<float>());
    } else {
      expand_scale(scale_src.data<MLFloat16>(), expanded_scale->MutableData<MLFloat16>());
    }

    // Expand zp if present
    if (zp_tensor_proto) {
      zp_src.emplace(graph, *zp_tensor_proto, graph.ModelPath());
      // Allocate as uint8 with enough bytes to hold quant_num*N packed sub-byte elements.
      int64_t expanded_zp_bytes = (quant_num * N * bits + 7) / 8;
      expanded_zp.emplace(uint8_type, TensorShape{expanded_zp_bytes}, cpu_allocator);

      // For sub-byte types, the zp is packed in bytes. We need to expand element-wise.
      // For 8-bit, each byte is one element. For 4-bit, 2 elements per byte. For 2-bit, 4 elements per byte.
      const uint8_t* zp_bytes = zp_src->DataAsByteSpan().data();
      uint8_t* dst_zp_bytes = expanded_zp->MutableData<uint8_t>();

      auto get_element = [bits](const uint8_t* data, int64_t idx) -> uint8_t {
        if (bits == 8) return data[idx];
        if (bits == 4) {
          uint8_t byte = data[idx / 2];
          return (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        }
        // bits == 2
        uint8_t byte = data[idx / 4];
        int shift = static_cast<int>((idx % 4) * 2);
        return (byte >> shift) & 0x03;
      };

      auto set_element = [bits](uint8_t* data, int64_t idx, uint8_t val) {
        if (bits == 8) {
          data[idx] = val;
          return;
        }
        if (bits == 4) {
          int64_t byte_idx = idx / 2;
          if (idx % 2 == 0) {
            data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F);
          } else {
            data[byte_idx] = (data[byte_idx] & 0x0F) | ((val & 0x0F) << 4);
          }
          return;
        }
        // bits == 2
        int64_t byte_idx = idx / 4;
        int shift = static_cast<int>((idx % 4) * 2);
        data[byte_idx] = (data[byte_idx] & ~(0x03 << shift)) | ((val & 0x03) << shift);
      };

      // Initialize expanded zp to 0
      memset(dst_zp_bytes, 0, expanded_zp->SizeInBytes());

      for (int64_t b = 0; b < quant_num; ++b) {
        for (int64_t n = 0; n < N; ++n) {
          int64_t src_idx = is_per_tensor ? 0 : n;
          uint8_t val = get_element(zp_bytes, src_idx);
          set_element(dst_zp_bytes, b * N + n, val);
        }
      }
    }
  }

  auto weight_dst_name = graph.GenerateNodeArgName(weight_arg->Name() + "_T");
  result.weight = Tensor(uint8_type, TensorShape{N, quant_num, blob_bytes}, cpu_allocator);
  // Zero-initialize: MLAS 4-bit transpose does not zero-pad when K < block_size,
  // leaving uninitialized bytes in the last block's padding region.
  memset(result.weight.MutableDataRaw(), 0, result.weight.SizeInBytes());

  auto scale_dst_name = graph.GenerateNodeArgName(scale_arg->Name() + "_T");
  auto scale_size = (TensorShape{N, quant_num}).Size();
  result.scale = Tensor(scale_type, TensorShape{scale_size}, cpu_allocator);

  std::string zp_dst_name;
  auto zp_size = (TensorShape{N, (quant_num * bits + 7) / 8}).Size();

  if (!is_blockwise && expanded_zp.has_value()) {
    // Per-tensor/per-channel path with expanded zero-point
    zp_dst_name = graph.GenerateNodeArgName(
        (zp_arg ? zp_arg->Name() : default_zp_name_prefix + "_zero_point") + "_T");
    result.zero_point = Tensor(uint8_type, TensorShape{zp_size}, cpu_allocator);
  } else if (zp_tensor_proto) {
    // Blockwise path with explicit zero-point
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
    using ScaleType = std::remove_const_t<std::remove_pointer_t<decltype(scale_data)>>;
    bool is_signed = IsDQWeightSigned(dt_weight);
    const uint8_t* src_w = weight_src.DataAsByteSpan().data();
    const uint8_t* src_zp = nullptr;
    if (expanded_zp.has_value()) {
      src_zp = expanded_zp->Data<uint8_t>();
    } else if (zp_src.has_value()) {
      src_zp = zp_src->DataAsByteSpan().data();
    }
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
    const float* s_data = expanded_scale.has_value() ? expanded_scale->Data<float>() : scale_src.data<float>();
    transpose(s_data, result.scale.MutableData<float>());
  } else {
    const MLFloat16* s_data = expanded_scale.has_value() ? expanded_scale->Data<MLFloat16>() : scale_src.data<MLFloat16>();
    transpose(s_data, result.scale.MutableData<MLFloat16>());
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
    concurrency::ThreadPool* intra_op_thread_pool,
    int64_t block_size_for_non_blockwise)
    : accuracy_level_{accuracy_level},
      domain_{kMSDomain},
      op_type_{"MatMulNBits"},
      value_moves_{[]() {
        NTO::NodeLocation target{NTO::NodeType::kTarget, 0};
        return std::vector<NodeAndMoveInfo>{
            MoveAndAppend(target, ArgType::kInput, 0, ArgType::kInput),
            MoveAll(target, ArgType::kOutput)};
      }()},
      intra_op_thread_pool_{intra_op_thread_pool},
      block_size_for_non_blockwise_{block_size_for_non_blockwise} {
  ORT_ENFORCE(accuracy_level_ >= 0 && accuracy_level_ <= 4, "MatMulNBits accuracy level must be between 0 and 4");
}

NodeAttributes
DQMatMulToMatMulNBitsAction::ExtraAttributes(const RuntimeState& runtime_state) const {
  NodeAttributes extra_attributes;

  const auto* dq_node = runtime_state.selected_nodes.Input(0);
  const auto* weight_shape = dq_node->InputDefs()[0]->Shape();
  ORT_ENFORCE(weight_shape != nullptr && weight_shape->dim_size() >= 2,
              "Weight shape unavailable for DQ node ", dq_node->Name());

  utils::SetNodeAttribute(utils::MakeAttribute("K", weight_shape->dim(0).dim_value()), extra_attributes);
  utils::SetNodeAttribute(utils::MakeAttribute("N", weight_shape->dim(1).dim_value()), extra_attributes);
  utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level_), extra_attributes);
  int32_t dt_weight = dq_node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  utils::SetNodeAttribute(utils::MakeAttribute("bits", DQWeightBits(dt_weight)), extra_attributes);
  int64_t effective_bs = GetEffectiveBlockSize(*dq_node, block_size_for_non_blockwise_);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", effective_bs), extra_attributes);

  return extra_attributes;
}

Status DQMatMulToMatMulNBitsAction::ProcessNewNode(Graph& graph,
                                                   const NodesToOptimize& selected_nodes,
                                                   Node& replacement_node) const {
  const auto* dq_node = selected_nodes.Input(0);

  int64_t effective_bs = GetEffectiveBlockSize(*dq_node, block_size_for_non_blockwise_);

  TransposedQuantizedTensors transposed;
  ORT_RETURN_IF_ERROR(TransposeDQWeightsForMatMulNBits(
      graph, *dq_node, "fused_DQ_MatMul", intra_op_thread_pool_, effective_bs, transposed));

  auto& input_defs = replacement_node.MutableInputDefs();
  input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, transposed.weight_proto, std::move(transposed.weight)));
  replacement_node.MutableInputArgsCount().push_back(1);

  input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, transposed.scale_proto, std::move(transposed.scale)));
  replacement_node.MutableInputArgsCount().push_back(1);

  if (transposed.zero_point_proto) {
    input_defs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, *transposed.zero_point_proto, std::move(*transposed.zero_point)));
    replacement_node.MutableInputArgsCount().push_back(1);
  }

  // If the target was Gemm, strip Gemm-specific attributes from the replacement MatMulNBits node
  // and wire the bias (if present) to MatMulNBits input 5.
  const auto& target = selected_nodes.Target();
  if (target.OpType() == "Gemm") {
    replacement_node.ClearAttribute("alpha");
    replacement_node.ClearAttribute("beta");
    replacement_node.ClearAttribute("transA");
    replacement_node.ClearAttribute("transB");

    // Wire Gemm bias to MatMulNBits input 5 (bias slot).
    // The bias can be a direct float tensor or the output of a DQ node.
    const auto& target_inputs = target.InputDefs();
    if (target_inputs.size() > 2 && target_inputs[2] && target_inputs[2]->Exists()) {
      // MatMulNBits input layout: 0:A, 1:B, 2:scales, 3:zp(opt), 4:g_idx(opt), 5:bias(opt)
      // Pad with empty NodeArgs up to position 5.
      NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);
      while (input_defs.size() < 5) {
        input_defs.push_back(&empty_arg);
        replacement_node.MutableInputArgsCount().push_back(1);
      }
      input_defs.push_back(const_cast<NodeArg*>(target_inputs[2]));
      replacement_node.MutableInputArgsCount().push_back(1);
    }
  }

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
