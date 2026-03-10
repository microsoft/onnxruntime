// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/dq_matmulnbits_fusion.h"

#if !defined(ORT_MINIMAL_BUILD)

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <unordered_set>

namespace onnxruntime {

namespace {

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

struct QuantTypeInfo {
  int64_t bits;
  bool is_signed;
};

// Map ONNX data types to quantization bit-width info.
std::optional<QuantTypeInfo> GetQuantTypeInfo(int32_t data_type) {
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UINT2:
      return QuantTypeInfo{2, false};
    case ONNX_NAMESPACE::TensorProto_DataType_INT2:
      return QuantTypeInfo{2, true};
    case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      return QuantTypeInfo{4, false};
    case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      return QuantTypeInfo{4, true};
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return QuantTypeInfo{8, false};
    default:
      return std::nullopt;
  }
}

// Extract a single N-bit element from packed data.
// For sub-byte types, elements are packed with even indices in the low bits.
uint8_t GetPackedElement(const uint8_t* packed, size_t index, size_t num_elements, int64_t bits) {
  ORT_ENFORCE(index < num_elements, "GetPackedElement: index ", index,
              " out of bounds (num_elements=", num_elements, ")");
  if (bits == 8) {
    return packed[index];
  }
  const int elems_per_byte = 8 / static_cast<int>(bits);
  const size_t byte_index = index / elems_per_byte;
  const int bit_offset = static_cast<int>((index % elems_per_byte) * bits);
  const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
  return static_cast<uint8_t>((packed[byte_index] >> bit_offset) & mask);
}

// Set a single N-bit element in packed data.
void SetPackedElement(uint8_t* packed, size_t index, uint8_t value, int64_t bits) {
  if (bits == 8) {
    packed[index] = value;
    return;
  }
  const int elems_per_byte = 8 / static_cast<int>(bits);
  const size_t byte_index = index / elems_per_byte;
  const int bit_offset = static_cast<int>((index % elems_per_byte) * bits);
  const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
  packed[byte_index] = static_cast<uint8_t>(
      (packed[byte_index] & ~(mask << bit_offset)) | ((value & mask) << bit_offset));
}

bool IsUniformPackedValue(const Initializer& init, uint8_t expected_value, int64_t bits) {
  const auto qtype = GetQuantTypeInfo(init.data_type());
  if (!qtype || qtype->bits != bits) {
    return false;
  }

  const size_t values_count = static_cast<size_t>(init.size());
  if (values_count == 0) {
    return false;
  }

  const auto packed = init.DataAsByteSpan();
  const uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
  const uint8_t expected = static_cast<uint8_t>(expected_value & mask);
  for (size_t i = 0; i < values_count; ++i) {
    if (GetPackedElement(packed.data(), i, values_count, bits) != expected) {
      return false;
    }
  }

  return true;
}

bool HasRank2Shape(const ONNX_NAMESPACE::TensorProto& tp, int64_t dim0, int64_t dim1) {
  return tp.dims_size() == 2 && tp.dims(0) == dim0 && tp.dims(1) == dim1;
}

// Compute the number of bytes needed to store 'count' N-bit elements.
int64_t PackedByteSize(int64_t count, int64_t bits) {
  return (count * bits + 7) / 8;
}

// Pack N-bit elements row-by-row from DQ layout to MatMulNBits layout.
void PackRows(const Initializer& src, int64_t rows, int64_t cols, int64_t bits, uint8_t* dst) {
  const int64_t row_bytes = PackedByteSize(cols, bits);
  const size_t dst_bytes = SafeInt<size_t>(rows) * row_bytes;
  const size_t total_elements = SafeInt<size_t>(rows) * cols;
  memset(dst, 0, dst_bytes);

  const auto src_packed = src.DataAsByteSpan();
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const size_t src_index = SafeInt<size_t>(r) * cols + c;
      const uint8_t value = GetPackedElement(src_packed.data(), src_index, total_elements, bits);

      const size_t dst_index = SafeInt<size_t>(r) * row_bytes * (8 / bits) + c;
      SetPackedElement(dst, dst_index, value, bits);
    }
  }
}

// Transpose and pack N-bit weights from DQ axis=0 layout [K, N] to MatMulNBits layout
// [N, k_blocks, blob_size]. blob_size = block_size * bits / 8.
void TransposePackWeightsAxis0(
    const uint8_t* src_packed, int64_t K, int64_t N, int64_t block_size, int64_t bits,
    uint8_t* dst) {
  const int64_t k_blocks = (K + block_size - 1) / block_size;
  const int64_t blob_size = block_size * bits / 8;
  const size_t dst_bytes = SafeInt<size_t>(N) * k_blocks * blob_size;
  const size_t total_elements = SafeInt<size_t>(K) * N;
  memset(dst, 0, dst_bytes);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const size_t src_index = SafeInt<size_t>(k) * N + n;
      const uint8_t val = GetPackedElement(src_packed, src_index, total_elements, bits);

      const int64_t kb = k / block_size;
      const int64_t off = k % block_size;
      // Destination: element index within the block's blob
      const size_t dst_elem = SafeInt<size_t>(n) * k_blocks * block_size + kb * block_size + off;
      SetPackedElement(dst, dst_elem, val, bits);
    }
  }
}

// Transpose and pack N-bit zero points from DQ axis=0 layout [k_blocks, N] to
// MatMulNBits layout UINT8 [N, packed_zp_bytes_per_n].
void TransposePackZPAxis0(
    const uint8_t* src_packed, int64_t k_blocks, int64_t N, int64_t bits,
    uint8_t* dst) {
  const int64_t zp_bytes_per_n = PackedByteSize(k_blocks, bits);
  const size_t dst_bytes = SafeInt<size_t>(N) * zp_bytes_per_n;
  const size_t total_elements = SafeInt<size_t>(k_blocks) * N;
  memset(dst, 0, dst_bytes);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t kb = 0; kb < k_blocks; ++kb) {
      const size_t src_index = SafeInt<size_t>(kb) * N + n;
      const uint8_t val = GetPackedElement(src_packed, src_index, total_elements, bits);

      const size_t dst_elem = SafeInt<size_t>(n) * k_blocks + kb;
      SetPackedElement(dst, dst_elem, val, bits);
    }
  }
}

// Returns the Cast node's target element type (the "to" attribute), or nullopt if invalid.
std::optional<int32_t> GetCastToType(const Node& cast_node) {
  const auto* to_attr = graph_utils::GetNodeAttribute(cast_node, "to");
  if (!to_attr) return std::nullopt;
  return static_cast<int32_t>(to_attr->i());
}

bool IsFloatOrFloat16(int32_t dt) {
  return dt == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
         dt == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
}

// ---------------------------------------------------------------------------
// Match structs
// ---------------------------------------------------------------------------

struct FusionMatch {
  NodeIndex matmul_idx;
  std::optional<NodeIndex> weight_cast_idx;  // Cast on weight path (between Transpose and MatMul)
  NodeIndex transpose_idx;
  NodeIndex reshape_idx;
  NodeIndex dq_idx;
  int64_t bits;
  std::optional<NodeIndex> input_a_cast_idx;   // Cast on input A
  std::optional<NodeIndex> output_cast_idx;    // Cast on MatMul output
  int32_t effective_dt_a;  // T1 for MatMulNBits (scale type)
};

struct DirectDQMatch {
  NodeIndex matmul_idx;
  NodeIndex dq_idx;
  int64_t bits;
  std::optional<NodeIndex> weight_cast_idx;    // Cast on weight path (between DQ and MatMul)
  std::optional<NodeIndex> input_a_cast_idx;   // Cast on input A
  std::optional<NodeIndex> output_cast_idx;    // Cast on MatMul output
  int32_t effective_dt_a;  // T1 for MatMulNBits (scale type)
};

// ---------------------------------------------------------------------------
// Shared Gemm validation (alpha=1, beta=1, transA=0, transB=0, bias 1-D [N])
// ---------------------------------------------------------------------------

bool ValidateGemmForFusion(const Node& gemm_node, int64_t N) {
  if (const auto* alpha_attr = graph_utils::GetNodeAttribute(gemm_node, "alpha");
      alpha_attr && std::abs(alpha_attr->f() - 1.0f) > 1e-6f)
    return false;
  if (const auto* beta_attr = graph_utils::GetNodeAttribute(gemm_node, "beta");
      beta_attr && std::abs(beta_attr->f() - 1.0f) > 1e-6f)
    return false;
  if (const auto* trans_a = graph_utils::GetNodeAttribute(gemm_node, "transA");
      trans_a && trans_a->i() != 0)
    return false;
  if (const auto* trans_b = graph_utils::GetNodeAttribute(gemm_node, "transB");
      trans_b && trans_b->i() != 0)
    return false;

  const auto& inputs = gemm_node.InputDefs();
  if (inputs.size() > 2 && inputs[2] && inputs[2]->Exists()) {
    const auto* bias_shape = inputs[2]->Shape();
    if (!bias_shape || bias_shape->dim_size() != 1 ||
        !utils::HasDimValue(bias_shape->dim(0)) ||
        bias_shape->dim(0).dim_value() != N)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Pattern 1 matching: DQ -> Reshape -> Transpose -> [Cast] -> MatMul/Gemm
// With optional Cast on input A and/or MatMul output for FP16 models.
// ---------------------------------------------------------------------------

std::vector<FusionMatch> CollectReshapeTransposeMatches(
    Graph& graph,
    const std::vector<NodeIndex>& node_topology_list,
    const logging::Logger& logger) {
  std::vector<FusionMatch> matches;

  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    if (!node) continue;

    if (node->OpType() != "MatMul" && node->OpType() != "Gemm") continue;

    const auto& mm_inputs = node->InputDefs();
    if (mm_inputs.size() < 2 || !mm_inputs[1] || !mm_inputs[1]->Exists()) continue;

    // Trace weight path: MatMul input B <- [Cast] <- Transpose <- Reshape <- DQ
    const Node* weight_cast_node = nullptr;
    const Node* transpose_node = graph.GetProducerNode(mm_inputs[1]->Name());
    if (transpose_node && transpose_node->OpType() == "Cast") {
      weight_cast_node = transpose_node;
      if (weight_cast_node->GetOutputEdgesCount() != 1) continue;
      const auto& cast_inputs = weight_cast_node->InputDefs();
      if (cast_inputs.empty() || !cast_inputs[0] || !cast_inputs[0]->Exists()) continue;
      transpose_node = graph.GetProducerNode(cast_inputs[0]->Name());
    }

    if (!transpose_node || transpose_node->OpType() != "Transpose") continue;
    if (transpose_node->GetOutputEdgesCount() != 1) continue;

    const auto& tp_inputs = transpose_node->InputDefs();
    if (tp_inputs.empty() || !tp_inputs[0] || !tp_inputs[0]->Exists()) continue;
    const Node* reshape_node = graph.GetProducerNode(tp_inputs[0]->Name());
    if (!reshape_node || reshape_node->OpType() != "Reshape") continue;
    if (reshape_node->GetOutputEdgesCount() != 1) continue;

    const auto& reshape_inputs = reshape_node->InputDefs();
    if (reshape_inputs.empty() || !reshape_inputs[0] || !reshape_inputs[0]->Exists()) continue;
    const Node* dq_node = graph.GetProducerNode(reshape_inputs[0]->Name());
    if (!dq_node || dq_node->OpType() != "DequantizeLinear") continue;
    if (dq_node->GetOutputEdgesCount() != 1) continue;

    const auto& dq_attrs = dq_node->GetAttributes();
    {
      auto it = dq_attrs.find("axis");
      if (it == dq_attrs.end() || it->second.i() != 2) continue;
    }
    int64_t block_size = 0;
    {
      auto it = dq_attrs.find("block_size");
      if (it == dq_attrs.end()) continue;
      block_size = it->second.i();
      if (block_size < 16 || ((block_size - 1) & block_size)) continue;
    }

    // Validate weight type: must be a supported quantized type
    const auto* weight_arg = dq_node->InputDefs()[0];
    if (!weight_arg || !weight_arg->Exists()) continue;
    const auto* weight_const_tp = graph.GetConstantInitializer(weight_arg->Name(), true);
    if (!weight_const_tp) continue;
    const auto weight_qtype = GetQuantTypeInfo(weight_const_tp->data_type());
    if (!weight_qtype) continue;
    const int64_t bits = weight_qtype->bits;
    if (weight_const_tp->dims_size() != 3) continue;
    const int64_t N = weight_const_tp->dims(0);
    const int64_t blocks = weight_const_tp->dims(1);
    const int64_t bs_dim = weight_const_tp->dims(2);
    if (N <= 0 || blocks <= 0 || bs_dim <= 0) continue;
    if (bs_dim != block_size) continue;
    const int64_t K = SafeInt<int64_t>(blocks) * bs_dim;

    // Scale type determines the effective T1 for MatMulNBits
    const auto* scale_arg = dq_node->InputDefs()[1];
    if (!scale_arg || !scale_arg->Exists()) continue;
    const auto* scale_const_tp = graph.GetConstantInitializer(scale_arg->Name(), true);
    if (!scale_const_tp) continue;
    int32_t dt_scale = scale_const_tp->data_type();
    if (!IsFloatOrFloat16(dt_scale)) continue;

    // Check input A type, looking through optional Cast
    const auto* a_arg = mm_inputs[0];
    if (!a_arg || !a_arg->TypeAsProto()) continue;
    int32_t dt_a = a_arg->TypeAsProto()->tensor_type().elem_type();

    const Node* input_a_cast_node = nullptr;
    int32_t effective_dt_a = dt_a;

    if (dt_a != dt_scale) {
      // Check if input A is produced by a Cast from dt_scale
      const Node* a_producer = graph.GetProducerNode(a_arg->Name());
      if (a_producer && a_producer->OpType() == "Cast") {
        const auto cast_to = GetCastToType(*a_producer);
        if (cast_to && *cast_to == dt_a) {
          const auto* cast_in = a_producer->InputDefs().empty() ? nullptr : a_producer->InputDefs()[0];
          if (cast_in && cast_in->TypeAsProto()) {
            int32_t dt_cast_in = cast_in->TypeAsProto()->tensor_type().elem_type();
            if (dt_cast_in == dt_scale && a_producer->GetOutputEdgesCount() == 1) {
              input_a_cast_node = a_producer;
              effective_dt_a = dt_scale;
            }
          }
        }
      }
      if (effective_dt_a != dt_scale) continue;
    }

    // Validate weight-path Cast: must cast to dt_a (the MatMul compute type)
    if (weight_cast_node) {
      const auto cast_to = GetCastToType(*weight_cast_node);
      if (!cast_to || *cast_to != dt_a) continue;
    }

    // Check for Cast on MatMul output
    const Node* output_cast_node = nullptr;
    if (node->GetOutputEdgesCount() == 1) {
      const auto edge = node->OutputEdgesBegin();
      const Node& consumer = edge->GetNode();
      if (consumer.OpType() == "Cast") {
        const auto cast_to = GetCastToType(consumer);
        if (cast_to && *cast_to == dt_scale && consumer.GetOutputEdgesCount() >= 1) {
          output_cast_node = &consumer;
        }
      }
    }

    const auto* reshape_shape_arg =
        reshape_node->InputDefs().size() > 1 ? reshape_node->InputDefs()[1] : nullptr;
    if (!reshape_shape_arg || !reshape_shape_arg->Exists()) continue;
    const auto* reshape_shape_tp = graph.GetConstantInitializer(reshape_shape_arg->Name(), true);
    if (!reshape_shape_tp) continue;

    Initializer reshape_shape_init(graph, *reshape_shape_tp, graph.ModelPath());
    if (reshape_shape_init.size() != 2) continue;

    int64_t reshape_dim0 = 0;
    int64_t reshape_dim1 = 0;
    if (reshape_shape_init.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      const auto* shape_data = reshape_shape_init.data<int64_t>();
      reshape_dim0 = shape_data[0];
      reshape_dim1 = shape_data[1];
    } else if (reshape_shape_init.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      const auto* shape_data = reshape_shape_init.data<int32_t>();
      reshape_dim0 = shape_data[0];
      reshape_dim1 = shape_data[1];
    } else {
      continue;
    }

    auto resolve_reshape_dim = [](int64_t dim, int64_t expected) -> std::optional<int64_t> {
      if (dim == expected || dim == 0 || dim == -1) {
        return expected;
      }
      return std::nullopt;
    };
    const auto resolved_reshape_dim0 = resolve_reshape_dim(reshape_dim0, N);
    const auto resolved_reshape_dim1 = resolve_reshape_dim(reshape_dim1, K);
    if (!resolved_reshape_dim0 || !resolved_reshape_dim1 ||
        *resolved_reshape_dim0 != N || *resolved_reshape_dim1 != K) {
      continue;
    }

    if (const auto* perm_attr = graph_utils::GetNodeAttribute(*transpose_node, "perm")) {
      if (perm_attr->ints_size() != 2 || perm_attr->ints(0) != 1 || perm_attr->ints(1) != 0) {
        continue;
      }
    }

    if (const auto* b_shape = mm_inputs[1]->Shape(); b_shape && b_shape->dim_size() == 2 &&
                                                     utils::HasDimValue(b_shape->dim(0)) && utils::HasDimValue(b_shape->dim(1)) &&
                                                     (b_shape->dim(0).dim_value() != K || b_shape->dim(1).dim_value() != N)) {
      continue;
    }

    if (const auto* a_shape = mm_inputs[0] ? mm_inputs[0]->Shape() : nullptr;
        a_shape && a_shape->dim_size() >= 1) {
      const int last_a_dim_idx = a_shape->dim_size() - 1;
      if (utils::HasDimValue(a_shape->dim(last_a_dim_idx)) &&
          a_shape->dim(last_a_dim_idx).dim_value() != K) {
        continue;
      }
    }

    const auto* y_shape = node->OutputDefs().empty() ? nullptr : node->OutputDefs()[0]->Shape();
    if (y_shape && y_shape->dim_size() >= 1) {
      const int last_y_dim_idx = y_shape->dim_size() - 1;
      if (utils::HasDimValue(y_shape->dim(last_y_dim_idx)) &&
          y_shape->dim(last_y_dim_idx).dim_value() != N) {
        continue;
      }
    }

    if (node->OpType() == "Gemm" && !ValidateGemmForFusion(*node, N)) continue;

    // Validate zero-point type matches weight type
    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();
    if (has_zp) {
      const auto* zp_const_tp = graph.GetConstantInitializer(zp_arg->Name(), true);
      if (!zp_const_tp) continue;
      const auto zp_qtype = GetQuantTypeInfo(zp_const_tp->data_type());
      if (!zp_qtype || zp_qtype->bits != bits) continue;
    }

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: matched pattern at MatMul node '"
                       << node->Name() << "' (bits=" << bits << ")";

    matches.push_back({node->Index(),
                       weight_cast_node ? std::optional<NodeIndex>(weight_cast_node->Index()) : std::nullopt,
                       transpose_node->Index(),
                       reshape_node->Index(), dq_node->Index(),
                       bits,
                       input_a_cast_node ? std::optional<NodeIndex>(input_a_cast_node->Index()) : std::nullopt,
                       output_cast_node ? std::optional<NodeIndex>(output_cast_node->Index()) : std::nullopt,
                       dt_scale});
  }

  return matches;
}

// ---------------------------------------------------------------------------
// Pattern 2 matching: direct DQ(axis=0, 2D) -> [Cast] -> MatMul/Gemm
// With optional Cast on input A and/or MatMul output for FP16 models.
// ---------------------------------------------------------------------------

std::vector<DirectDQMatch> CollectDirectDQMatches(
    Graph& graph,
    const std::vector<NodeIndex>& node_topology_list,
    const std::unordered_set<NodeIndex>& skip_indices,
    const logging::Logger& logger) {
  std::vector<DirectDQMatch> direct_matches;

  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    if (!node) continue;

    if (node->OpType() != "MatMul" && node->OpType() != "Gemm") continue;
    if (skip_indices.count(node->Index())) continue;

    const auto& mm_inputs = node->InputDefs();
    if (mm_inputs.size() < 2 || !mm_inputs[1] || !mm_inputs[1]->Exists()) continue;

    // Trace weight path: MatMul input B <- [Cast] <- DQ
    const Node* weight_cast_node = nullptr;
    const Node* dq_node = graph.GetProducerNode(mm_inputs[1]->Name());
    if (dq_node && dq_node->OpType() == "Cast") {
      weight_cast_node = dq_node;
      if (weight_cast_node->GetOutputEdgesCount() != 1) continue;
      const auto& cast_inputs = weight_cast_node->InputDefs();
      if (cast_inputs.empty() || !cast_inputs[0] || !cast_inputs[0]->Exists()) continue;
      dq_node = graph.GetProducerNode(cast_inputs[0]->Name());
    }

    if (!dq_node || dq_node->OpType() != "DequantizeLinear") continue;
    if (dq_node->GetOutputEdgesCount() != 1) continue;

    const auto& dq_attrs = dq_node->GetAttributes();
    {
      auto it = dq_attrs.find("axis");
      if (it == dq_attrs.end() || it->second.i() != 0) continue;
    }
    int64_t block_size = 0;
    {
      auto it = dq_attrs.find("block_size");
      if (it == dq_attrs.end()) continue;
      block_size = it->second.i();
      if (block_size < 16 || ((block_size - 1) & block_size)) continue;
    }

    const auto* weight_arg = dq_node->InputDefs()[0];
    if (!weight_arg || !weight_arg->Exists()) continue;
    const auto* weight_const_tp = graph.GetConstantInitializer(weight_arg->Name(), true);
    if (!weight_const_tp) continue;
    const auto weight_qtype = GetQuantTypeInfo(weight_const_tp->data_type());
    if (!weight_qtype) continue;
    const int64_t bits = weight_qtype->bits;
    if (weight_const_tp->dims_size() != 2) continue;
    const int64_t K = weight_const_tp->dims(0);
    const int64_t N = weight_const_tp->dims(1);
    if (K <= 0 || N <= 0 || K % block_size != 0) continue;
    const int64_t k_blocks = K / block_size;

    const auto* scale_arg = dq_node->InputDefs()[1];
    if (!scale_arg || !scale_arg->Exists()) continue;
    const auto* scale_const_tp = graph.GetConstantInitializer(scale_arg->Name(), true);
    if (!scale_const_tp) continue;
    int32_t dt_scale = scale_const_tp->data_type();
    if (!IsFloatOrFloat16(dt_scale)) continue;
    if (!HasRank2Shape(*scale_const_tp, k_blocks, N)) continue;

    // Check input A type, looking through optional Cast
    const auto* a_arg = mm_inputs[0];
    if (!a_arg || !a_arg->TypeAsProto()) continue;
    int32_t dt_a = a_arg->TypeAsProto()->tensor_type().elem_type();

    const Node* input_a_cast_node = nullptr;
    int32_t effective_dt_a = dt_a;

    if (dt_a != dt_scale) {
      const Node* a_producer = graph.GetProducerNode(a_arg->Name());
      if (a_producer && a_producer->OpType() == "Cast") {
        const auto cast_to = GetCastToType(*a_producer);
        if (cast_to && *cast_to == dt_a) {
          const auto* cast_in = a_producer->InputDefs().empty() ? nullptr : a_producer->InputDefs()[0];
          if (cast_in && cast_in->TypeAsProto()) {
            int32_t dt_cast_in = cast_in->TypeAsProto()->tensor_type().elem_type();
            if (dt_cast_in == dt_scale && a_producer->GetOutputEdgesCount() == 1) {
              input_a_cast_node = a_producer;
              effective_dt_a = dt_scale;
            }
          }
        }
      }
      if (effective_dt_a != dt_scale) continue;
    }

    // Validate weight-path Cast
    if (weight_cast_node) {
      const auto cast_to = GetCastToType(*weight_cast_node);
      if (!cast_to || *cast_to != dt_a) continue;
    }

    // Check for Cast on MatMul output
    const Node* output_cast_node = nullptr;
    if (node->GetOutputEdgesCount() == 1) {
      const auto edge = node->OutputEdgesBegin();
      const Node& consumer = edge->GetNode();
      if (consumer.OpType() == "Cast") {
        const auto cast_to = GetCastToType(consumer);
        if (cast_to && *cast_to == dt_scale && consumer.GetOutputEdgesCount() >= 1) {
          output_cast_node = &consumer;
        }
      }
    }

    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();
    if (has_zp) {
      const auto* zp_const_tp = graph.GetConstantInitializer(zp_arg->Name(), true);
      if (!zp_const_tp) continue;
      const auto zp_qtype = GetQuantTypeInfo(zp_const_tp->data_type());
      if (!zp_qtype || zp_qtype->bits != bits) continue;
      if (!HasRank2Shape(*zp_const_tp, k_blocks, N)) continue;
    }

    if (node->OpType() == "Gemm" && !ValidateGemmForFusion(*node, N)) continue;

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: matched direct DQ->MatMul pattern at node '"
                       << node->Name() << "' (K=" << K << ", N=" << N
                       << ", block_size=" << block_size << ", bits=" << bits << ")";
    direct_matches.push_back({node->Index(), dq_node->Index(), bits,
                              weight_cast_node ? std::optional<NodeIndex>(weight_cast_node->Index()) : std::nullopt,
                              input_a_cast_node ? std::optional<NodeIndex>(input_a_cast_node->Index()) : std::nullopt,
                              output_cast_node ? std::optional<NodeIndex>(output_cast_node->Index()) : std::nullopt,
                              dt_scale});
  }

  return direct_matches;
}

// ---------------------------------------------------------------------------
// Pattern 1 rewriting: DQ+Reshape+Transpose+[Cast]+MatMul/Gemm -> MatMulNBits
// ---------------------------------------------------------------------------

void ApplyReshapeTransposeFusions(
    Graph& graph,
    const std::vector<FusionMatch>& matches,
    int64_t accuracy_level,
    bool& modified,
    const logging::Logger& logger) {
  for (const auto& match : matches) {
    const Node* mm_node = graph.GetNode(match.matmul_idx);
    const Node* weight_cast_node = match.weight_cast_idx ? graph.GetNode(*match.weight_cast_idx) : nullptr;
    const Node* tp_node = graph.GetNode(match.transpose_idx);
    const Node* dq_node = graph.GetNode(match.dq_idx);
    const Node* reshape_node = graph.GetNode(match.reshape_idx);
    const Node* input_a_cast_node = match.input_a_cast_idx ? graph.GetNode(*match.input_a_cast_idx) : nullptr;
    const Node* output_cast_node = match.output_cast_idx ? graph.GetNode(*match.output_cast_idx) : nullptr;
    if (!mm_node || !tp_node || !dq_node || !reshape_node ||
        (match.weight_cast_idx && !weight_cast_node) ||
        (match.input_a_cast_idx && !input_a_cast_node) ||
        (match.output_cast_idx && !output_cast_node)) {
      continue;
    }

    const int64_t bits = match.bits;
    const int32_t effective_dt_a = match.effective_dt_a;

    const auto* weight_arg = dq_node->InputDefs()[0];
    const auto* scale_arg = dq_node->InputDefs()[1];
    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();

    const auto& dq_attrs = dq_node->GetAttributes();
    const int64_t block_size = dq_attrs.at("block_size").i();

    const ONNX_NAMESPACE::TensorProto* weight_tp = nullptr;
    if (!graph.GetInitializedTensor(weight_arg->Name(), weight_tp) || !weight_tp) continue;
    const ONNX_NAMESPACE::TensorProto* scale_tp = nullptr;
    if (!graph.GetInitializedTensor(scale_arg->Name(), scale_tp) || !scale_tp) continue;
    const ONNX_NAMESPACE::TensorProto* zp_tp = nullptr;
    if (has_zp) {
      if (!graph.GetInitializedTensor(zp_arg->Name(), zp_tp) || !zp_tp) continue;
    }

    const auto weight_qtype = GetQuantTypeInfo(weight_tp->data_type());
    if (!weight_qtype || weight_qtype->bits != bits || weight_tp->dims_size() != 3) {
      continue;
    }

    const int64_t N = weight_tp->dims(0);
    const int64_t quant_num = weight_tp->dims(1);
    const int64_t bs_dim = weight_tp->dims(2);
    if (N <= 0 || quant_num <= 0 || bs_dim <= 0 || bs_dim != block_size) continue;
    const int64_t K = SafeInt<int64_t>(quant_num) * bs_dim;
    const int64_t blob_bytes = PackedByteSize(block_size, bits);

    Initializer weight_src(graph, *weight_tp, graph.ModelPath());
    Initializer scale_src(graph, *scale_tp, graph.ModelPath());
    if (!IsFloatOrFloat16(scale_src.data_type())) {
      continue;
    }

    auto uint8_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          ONNX_NAMESPACE::TensorProto_DataType_UINT8)
                          ->GetElementType();
    auto scale_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          scale_src.data_type())
                          ->GetElementType();

    auto cpu_allocator = CPUAllocator::DefaultInstance();

    auto weight_dst_name = graph.GenerateNodeArgName(weight_arg->Name() + "_mnb");
    auto weight_dst = Tensor(uint8_type, TensorShape{N, quant_num, blob_bytes}, cpu_allocator);

    auto scale_dst_name = graph.GenerateNodeArgName(scale_arg->Name() + "_mnb");
    const int64_t scale_size = (TensorShape{N, quant_num}).Size();
    if (scale_src.size() != static_cast<size_t>(scale_size)) continue;
    auto scale_dst = Tensor(scale_type, TensorShape{scale_size}, cpu_allocator);

    std::string zp_dst_name;
    std::optional<Tensor> zp_dst;
    const int64_t zp_packed_size = SafeInt<int64_t>(N) * PackedByteSize(quant_num, bits);

    bool elide_default_zp = false;
    std::optional<Initializer> zp_src;
    const uint8_t mnb_default_zp = static_cast<uint8_t>(1 << (bits - 1));  // 2^(bits-1)

    const auto weight_bytes = weight_src.DataAsByteSpan();
    if (weight_bytes.size() != static_cast<size_t>(weight_dst.SizeInBytes())) continue;
    memcpy(weight_dst.MutableDataRaw(), weight_bytes.data(), weight_bytes.size());

    if (scale_src.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      memcpy(scale_dst.MutableData<float>(), scale_src.data<float>(),
             static_cast<size_t>(scale_size) * sizeof(float));
    } else {
      memcpy(scale_dst.MutableData<MLFloat16>(), scale_src.data<MLFloat16>(),
             static_cast<size_t>(scale_size) * sizeof(MLFloat16));
    }

    if (zp_tp) {
      zp_src.emplace(graph, *zp_tp, graph.ModelPath());
      const auto zp_qtype = GetQuantTypeInfo(zp_src->data_type());
      if (!zp_qtype || zp_qtype->bits != bits) continue;
      if (zp_src->size() != static_cast<size_t>(N * quant_num)) continue;

      if (IsUniformPackedValue(*zp_src, mnb_default_zp, bits)) {
        elide_default_zp = true;
      } else {
        zp_dst_name = graph.GenerateNodeArgName(zp_arg->Name() + "_mnb");
        zp_dst = Tensor(uint8_type, TensorShape{zp_packed_size}, cpu_allocator);
        PackRows(*zp_src, N, quant_num, bits, zp_dst->MutableData<uint8_t>());
      }
    } else {
      // DequantizeLinear default zero-point is 0, while MatMulNBits
      // default is 2^(bits-1). Emit explicit zeros to preserve semantics.
      zp_dst_name = graph.GenerateNodeArgName("fused_DQ_zp_mnb");
      zp_dst = Tensor(uint8_type, TensorShape{zp_packed_size}, cpu_allocator);
      memset(zp_dst->MutableDataRaw(), 0, zp_dst->SizeInBytes());
    }

    auto weight_mnb_tp = utils::TensorToTensorProto(weight_dst, weight_dst_name, true);
    auto scale_mnb_tp = utils::TensorToTensorProto(scale_dst, scale_dst_name, true);
    std::optional<ONNX_NAMESPACE::TensorProto> zp_mnb_tp;
    if (zp_dst && !elide_default_zp) {
      zp_mnb_tp.emplace(utils::TensorToTensorProto(*zp_dst, zp_dst_name, true));
    }

    NodeAttributes mnb_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("K", K), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("N", N), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("bits", bits), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), mnb_attrs);

    // Determine input A: use pre-Cast value if input A Cast is being removed
    NodeArg* mnb_input_a = input_a_cast_node
                               ? const_cast<NodeArg*>(input_a_cast_node->InputDefs()[0])
                               : const_cast<NodeArg*>(mm_node->InputDefs()[0]);

    std::vector<NodeArg*> mnb_inputs;
    mnb_inputs.push_back(mnb_input_a);
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, weight_mnb_tp, std::move(weight_dst)));
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, scale_mnb_tp, std::move(scale_dst)));
    if (zp_mnb_tp) {
      mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, zp_mnb_tp.value(), std::move(*zp_dst)));
    }

    // MatMulNBits input layout: 0:A, 1:B, 2:scales, 3:zero_points(opt), 4:g_idx(opt), 5:bias(opt)
    bool fused_with_bias = false;
    if (mm_node->OpType() == "Gemm" &&
        mm_node->InputDefs().size() > 2 &&
        mm_node->InputDefs()[2] &&
        mm_node->InputDefs()[2]->Exists()) {
      NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);
      while (mnb_inputs.size() < 5) {
        mnb_inputs.push_back(&empty_arg);
      }
      mnb_inputs.push_back(const_cast<NodeArg*>(mm_node->InputDefs()[2]));
      fused_with_bias = true;
    }

    // Determine output: if output Cast exists, take over its output; otherwise MatMul's output
    std::vector<NodeArg*> mnb_outputs;
    if (output_cast_node) {
      mnb_outputs.push_back(const_cast<NodeArg*>(output_cast_node->OutputDefs()[0]));
    } else if (effective_dt_a != mm_node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      // MatMulNBits outputs T1 (=effective_dt_a) but consumers expect the original type.
      // Create a new intermediate output for MatMulNBits and insert a Cast after it.
      auto& mnb_out_arg = graph.GetOrCreateNodeArg(
          graph.GenerateNodeArgName("mnb_out"),
          nullptr);
      mnb_outputs.push_back(&mnb_out_arg);
    } else {
      mnb_outputs.push_back(const_cast<NodeArg*>(mm_node->OutputDefs()[0]));
    }

    auto& mnb_node = graph.AddNode(
        graph.GenerateNodeName("DQFusedMatMulNBits"),
        "MatMulNBits",
        "Fused from DQ+Reshape+Transpose+MatMul",
        mnb_inputs, mnb_outputs, &mnb_attrs, kMSDomain);
    mnb_node.SetExecutionProviderType(mm_node->GetExecutionProviderType());

    // If we need a Cast after MatMulNBits (no output_cast_node to absorb the type difference)
    if (!output_cast_node &&
        effective_dt_a != mm_node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      int32_t original_dt = mm_node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      NodeAttributes cast_attrs;
      utils::SetNodeAttribute(utils::MakeAttribute("to", static_cast<int64_t>(original_dt)), cast_attrs);
      graph.AddNode(
          graph.GenerateNodeName("DQFusedMatMulNBits_Cast"),
          "Cast", "Cast MNB output to original type",
          {mnb_outputs[0]},
          {const_cast<NodeArg*>(mm_node->OutputDefs()[0])},
          &cast_attrs);
    }

    // Remove nodes in reverse dependency order
    if (output_cast_node) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.output_cast_idx.value()));
      graph.RemoveNode(match.output_cast_idx.value());
    }

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.matmul_idx));
    graph.RemoveNode(match.matmul_idx);

    if (input_a_cast_node) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.input_a_cast_idx.value()));
      graph.RemoveNode(match.input_a_cast_idx.value());
    }

    if (match.weight_cast_idx && graph.GetNode(*match.weight_cast_idx)) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(*match.weight_cast_idx));
      graph.RemoveNode(*match.weight_cast_idx);
    }

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.transpose_idx));
    graph.RemoveNode(match.transpose_idx);

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.reshape_idx));
    graph.RemoveNode(match.reshape_idx);

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.dq_idx));
    graph.RemoveNode(match.dq_idx);

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: fused DQ+Reshape+Transpose"
                       << (match.weight_cast_idx ? "+Cast" : "")
                       << "+MatMul/Gemm -> MatMulNBits"
                       << " (bits=" << bits << ")"
                       << (fused_with_bias ? " (bias preserved)" : "")
                       << (elide_default_zp ? " (default zp elided)" : "")
                       << (input_a_cast_node ? " (input Cast removed)" : "")
                       << (output_cast_node ? " (output Cast removed)" : "");
    modified = true;
  }
}

// ---------------------------------------------------------------------------
// Pattern 2 rewriting: direct DQ(axis=0) + [Cast] + MatMul/Gemm -> MatMulNBits
// ---------------------------------------------------------------------------

void ApplyDirectDQFusions(
    Graph& graph,
    const std::vector<DirectDQMatch>& matches,
    int64_t accuracy_level,
    bool& modified,
    const logging::Logger& logger) {
  for (const auto& match : matches) {
    const Node* mm_node = graph.GetNode(match.matmul_idx);
    const Node* dq_node = graph.GetNode(match.dq_idx);
    const Node* weight_cast_node = match.weight_cast_idx ? graph.GetNode(*match.weight_cast_idx) : nullptr;
    const Node* input_a_cast_node = match.input_a_cast_idx ? graph.GetNode(*match.input_a_cast_idx) : nullptr;
    const Node* output_cast_node = match.output_cast_idx ? graph.GetNode(*match.output_cast_idx) : nullptr;
    if (!mm_node || !dq_node ||
        (match.weight_cast_idx && !weight_cast_node) ||
        (match.input_a_cast_idx && !input_a_cast_node) ||
        (match.output_cast_idx && !output_cast_node)) {
      continue;
    }

    const int64_t bits = match.bits;
    const int32_t effective_dt_a = match.effective_dt_a;

    const auto* weight_arg = dq_node->InputDefs()[0];
    const auto* scale_arg = dq_node->InputDefs()[1];
    const auto* zp_arg = dq_node->InputDefs().size() > 2 ? dq_node->InputDefs()[2] : nullptr;
    bool has_zp = zp_arg && zp_arg->Exists();

    const auto& dq_attrs = dq_node->GetAttributes();
    const int64_t block_size = dq_attrs.at("block_size").i();

    const ONNX_NAMESPACE::TensorProto* weight_tp = nullptr;
    if (!graph.GetInitializedTensor(weight_arg->Name(), weight_tp) || !weight_tp) continue;
    const ONNX_NAMESPACE::TensorProto* scale_tp = nullptr;
    if (!graph.GetInitializedTensor(scale_arg->Name(), scale_tp) || !scale_tp) continue;
    const ONNX_NAMESPACE::TensorProto* zp_tp = nullptr;
    if (has_zp) {
      if (!graph.GetInitializedTensor(zp_arg->Name(), zp_tp) || !zp_tp) continue;
    }

    const auto weight_qtype = GetQuantTypeInfo(weight_tp->data_type());
    if (!weight_qtype || weight_qtype->bits != bits || weight_tp->dims_size() != 2) continue;

    const int64_t K = weight_tp->dims(0);
    const int64_t N = weight_tp->dims(1);
    if (K <= 0 || N <= 0 || block_size <= 0 || K % block_size != 0) continue;
    const int64_t k_blocks = K / block_size;
    const int64_t blob_bytes = block_size * bits / 8;
    if (!HasRank2Shape(*scale_tp, k_blocks, N)) continue;
    if (zp_tp && !HasRank2Shape(*zp_tp, k_blocks, N)) continue;

    Initializer weight_src(graph, *weight_tp, graph.ModelPath());
    const size_t required_weight_bytes = SafeInt<size_t>(N) * k_blocks * blob_bytes;
    if (weight_src.DataAsByteSpan().size() < required_weight_bytes) continue;
    Initializer scale_src(graph, *scale_tp, graph.ModelPath());
    if (!IsFloatOrFloat16(scale_src.data_type())) continue;

    auto uint8_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          ONNX_NAMESPACE::TensorProto_DataType_UINT8)
                          ->GetElementType();
    auto scale_type = DataTypeImpl::TensorTypeFromONNXEnum(
                          scale_src.data_type())
                          ->GetElementType();
    auto cpu_allocator = CPUAllocator::DefaultInstance();

    auto weight_dst_name = graph.GenerateNodeArgName(weight_arg->Name() + "_mnb");
    auto weight_dst = Tensor(uint8_type, TensorShape{N, k_blocks, blob_bytes}, cpu_allocator);
    TransposePackWeightsAxis0(weight_src.DataAsByteSpan().data(), K, N, block_size, bits,
                              weight_dst.MutableData<uint8_t>());

    auto scale_dst_name = graph.GenerateNodeArgName(scale_arg->Name() + "_mnb");
    const int64_t scale_count = SafeInt<int64_t>(N) * k_blocks;
    if (scale_src.size() != static_cast<size_t>(scale_count)) continue;
    auto scale_dst = Tensor(scale_type, TensorShape{scale_count}, cpu_allocator);

    if (scale_src.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      const float* src = scale_src.data<float>();
      float* dst = scale_dst.MutableData<float>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t kb = 0; kb < k_blocks; ++kb)
          dst[n * k_blocks + kb] = src[kb * N + n];
    } else {
      const MLFloat16* src = scale_src.data<MLFloat16>();
      MLFloat16* dst = scale_dst.MutableData<MLFloat16>();
      for (int64_t n = 0; n < N; ++n)
        for (int64_t kb = 0; kb < k_blocks; ++kb)
          dst[n * k_blocks + kb] = src[kb * N + n];
    }

    std::string zp_dst_name;
    std::optional<Tensor> zp_dst;
    const int64_t zp_bytes_total = SafeInt<int64_t>(N) * PackedByteSize(k_blocks, bits);

    bool elide_zp = false;
    const uint8_t mnb_default_zp = static_cast<uint8_t>(1 << (bits - 1));

    if (zp_tp) {
      Initializer zp_src(graph, *zp_tp, graph.ModelPath());
      const auto zp_qtype = GetQuantTypeInfo(zp_src.data_type());
      if (!zp_qtype || zp_qtype->bits != bits) continue;
      if (zp_src.size() != static_cast<size_t>(k_blocks * N)) continue;

      if (IsUniformPackedValue(zp_src, mnb_default_zp, bits)) {
        elide_zp = true;
      } else {
        zp_dst_name = graph.GenerateNodeArgName(zp_arg->Name() + "_mnb");
        zp_dst = Tensor(uint8_type, TensorShape{zp_bytes_total}, cpu_allocator);
        TransposePackZPAxis0(zp_src.DataAsByteSpan().data(), k_blocks, N, bits,
                             zp_dst->MutableData<uint8_t>());
      }
    } else {
      // DQ default ZP is 0, MatMulNBits default is 2^(bits-1). Emit explicit zeros.
      zp_dst_name = graph.GenerateNodeArgName("direct_DQ_zp_mnb");
      zp_dst = Tensor(uint8_type, TensorShape{zp_bytes_total}, cpu_allocator);
      memset(zp_dst->MutableDataRaw(), 0, zp_dst->SizeInBytes());
    }

    auto weight_mnb_tp = utils::TensorToTensorProto(weight_dst, weight_dst_name, true);
    auto scale_mnb_tp = utils::TensorToTensorProto(scale_dst, scale_dst_name, true);
    std::optional<ONNX_NAMESPACE::TensorProto> zp_mnb_tp;
    if (zp_dst && !elide_zp) {
      zp_mnb_tp.emplace(utils::TensorToTensorProto(*zp_dst, zp_dst_name, true));
    }

    NodeAttributes mnb_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("K", K), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("N", N), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("accuracy_level", accuracy_level), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("bits", bits), mnb_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), mnb_attrs);

    // Determine input A: use pre-Cast value if input A Cast is being removed
    NodeArg* mnb_input_a = input_a_cast_node
                               ? const_cast<NodeArg*>(input_a_cast_node->InputDefs()[0])
                               : const_cast<NodeArg*>(mm_node->InputDefs()[0]);

    std::vector<NodeArg*> mnb_inputs;
    mnb_inputs.push_back(mnb_input_a);
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, weight_mnb_tp, std::move(weight_dst)));
    mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, scale_mnb_tp, std::move(scale_dst)));
    if (zp_mnb_tp) {
      mnb_inputs.push_back(&graph_utils::AddInitializerWithOrtValue(graph, zp_mnb_tp.value(), std::move(*zp_dst)));
    }

    bool fused_with_bias = false;
    if (mm_node->OpType() == "Gemm" &&
        mm_node->InputDefs().size() > 2 &&
        mm_node->InputDefs()[2] &&
        mm_node->InputDefs()[2]->Exists()) {
      NodeArg& empty_arg = graph.GetOrCreateNodeArg("", nullptr);
      while (mnb_inputs.size() < 5) {
        mnb_inputs.push_back(&empty_arg);
      }
      mnb_inputs.push_back(const_cast<NodeArg*>(mm_node->InputDefs()[2]));
      fused_with_bias = true;
    }

    // Determine output: if output Cast exists, take over its output; otherwise MatMul's output
    std::vector<NodeArg*> mnb_outputs;
    if (output_cast_node) {
      mnb_outputs.push_back(const_cast<NodeArg*>(output_cast_node->OutputDefs()[0]));
    } else if (effective_dt_a != mm_node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      auto& mnb_out_arg = graph.GetOrCreateNodeArg(
          graph.GenerateNodeArgName("mnb_out"),
          nullptr);
      mnb_outputs.push_back(&mnb_out_arg);
    } else {
      mnb_outputs.push_back(const_cast<NodeArg*>(mm_node->OutputDefs()[0]));
    }

    auto& mnb_node = graph.AddNode(
        graph.GenerateNodeName("DirectDQFusedMatMulNBits"),
        "MatMulNBits",
        "Fused from direct DQ(axis=0)+MatMul",
        mnb_inputs, mnb_outputs, &mnb_attrs, kMSDomain);
    mnb_node.SetExecutionProviderType(mm_node->GetExecutionProviderType());

    // If we need a Cast after MatMulNBits
    if (!output_cast_node &&
        effective_dt_a != mm_node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      int32_t original_dt = mm_node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      NodeAttributes cast_attrs;
      utils::SetNodeAttribute(utils::MakeAttribute("to", static_cast<int64_t>(original_dt)), cast_attrs);
      graph.AddNode(
          graph.GenerateNodeName("DirectDQFusedMatMulNBits_Cast"),
          "Cast", "Cast MNB output to original type",
          {mnb_outputs[0]},
          {const_cast<NodeArg*>(mm_node->OutputDefs()[0])},
          &cast_attrs);
    }

    // Remove nodes in reverse dependency order
    if (output_cast_node) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.output_cast_idx.value()));
      graph.RemoveNode(match.output_cast_idx.value());
    }

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.matmul_idx));
    graph.RemoveNode(match.matmul_idx);

    if (input_a_cast_node) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.input_a_cast_idx.value()));
      graph.RemoveNode(match.input_a_cast_idx.value());
    }

    if (weight_cast_node) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.weight_cast_idx.value()));
      graph.RemoveNode(match.weight_cast_idx.value());
    }

    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(match.dq_idx));
    graph.RemoveNode(match.dq_idx);

    LOGS(logger, INFO) << "DQMatMulNBitsFusion: fused direct DQ(axis=0)+MatMul/Gemm -> MatMulNBits"
                       << " (K=" << K << ", N=" << N << ", block_size=" << block_size
                       << ", bits=" << bits << ")"
                       << (fused_with_bias ? " (bias preserved)" : "")
                       << (elide_zp ? " (default zp elided)" : "")
                       << (input_a_cast_node ? " (input Cast removed)" : "")
                       << (output_cast_node ? " (output Cast removed)" : "");
    modified = true;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// DQMatMulNBitsFusion public interface
// ---------------------------------------------------------------------------

DQMatMulNBitsFusion::DQMatMulNBitsFusion(
    int64_t accuracy_level,
    const InlinedHashSet<std::string_view>& compatible_eps)
    : GraphTransformer("DQMatMulNBitsFusion", compatible_eps),
      accuracy_level_(accuracy_level) {
  ORT_ENFORCE(accuracy_level_ >= 0 && accuracy_level_ <= 4,
              "MatMulNBits accuracy level must be between 0 and 4");
}

Status DQMatMulNBitsFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                      const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node = graph.GetNode(node_index);
    if (!node) continue;
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
  }

  auto matches = CollectReshapeTransposeMatches(graph, node_topology_list, logger);

  std::unordered_set<NodeIndex> matched_matmul_indices;
  for (const auto& m : matches) {
    matched_matmul_indices.insert(m.matmul_idx);
  }

  auto direct_matches = CollectDirectDQMatches(graph, node_topology_list,
                                               matched_matmul_indices, logger);

  ApplyReshapeTransposeFusions(graph, matches, accuracy_level_, modified, logger);
  ApplyDirectDQFusions(graph, direct_matches, accuracy_level_, modified, logger);

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
