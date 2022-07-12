// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if (defined(_M_AMD64) && !defined(_M_ARM64EC)) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__) || !defined(DISABLE_CONTRIB_OPS)

#include "core/optimizer/avx2_weight_s8_to_u8.h"

#include <algorithm>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

/**
 * @brief Convert the source int8_t TensorProto to a uint8_t one if the tensor
 *        contains values outside of [-64, 64]
 * @param src           The source tensor, must be type int8_t
 * @param dst           An empty tensor, will contain the converted tensor data
 * @param graph         Graph for generating tensor name or provide external
 *                      data path
 * @param force         Perform conversion even when tensor values within [-64, 64]
 * @return              Whether the conversion happened.
*/
static inline bool Int8TensorProto2Uint8(
    const ONNX_NAMESPACE::TensorProto* src,
    ONNX_NAMESPACE::TensorProto& dst,
    Graph& graph, bool force = false) {
  dst.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);

  if (nullptr == src) {
    uint8_t zero_val = 128;
    dst.set_name(graph.GenerateNodeArgName("weight_zp_s8_2_u8"));
    dst.set_raw_data(&zero_val, sizeof(uint8_t));
    return true;
  }

  dst.set_name(src->name() + "_s8_2_u8");
  dst.mutable_dims()->CopyFrom(src->dims());

  // TODO(fuchen): too many copies!
  //
  // Here we do two memory copies: Proto -> Initializer -> Proto.
  // Ideally we only do 1 copy, just iterate the source data, and write directly
  // to the dst raw buffer.
  // Unfortunately iterating the source data is complicated, the data maybe in
  // external file, a raw buffer, or a repeated field depending on the data
  // type.  UnpackTensor() already contains some of these logic and is closest
  // to what we need. But it does not handle external data. Write our own code
  // here means copy the logic of TensorProtoToTensor(), a violation of DRY
  // principle. A better solution is to provide an efficient const iterator for
  // TensorProto. This require coordination with onnx side.

  Initializer temp(*src, graph.ModelPath());
  int8_t* p = temp.data<int8_t>();
  bool should_convert = false;
  for (int i = 0; i < temp.size(); i++) {
    if (*p < -64 || *p > 64) {
      should_convert = true;
    }
    *p ^= 0x80;
    p++;
  }
  if (force || should_convert) {
    dst.set_raw_data(temp.data<int8_t>(), temp.size());
    return true;
  }
  return false;
}

/**
 * @brief If the op_node has an uint8_t const weight tensor, convert it to int8_t
 * @param graph 
 * @param op_node 
 * @param weights_idx
 * @param weight_zp_idx
 * @return true when conversion happened.
*/
static bool ConvertS8WeightToU8(Graph& graph, Node& op_node,
                                size_t weights_idx, size_t weight_zp_idx) {
  auto& input_defs = op_node.MutableInputDefs();
  if (input_defs.size() < weights_idx + 1) {
    return false;
  }

  // Weight tensor must be const int8_t
  const ONNX_NAMESPACE::TensorProto* weight_tensor_proto = nullptr;
  const auto* w_def = input_defs[weights_idx];
  if (!graph_utils::NodeArgIsConstant(graph, *w_def) ||
      !graph.GetInitializedTensor(w_def->Name(), weight_tensor_proto) ||
      weight_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    return false;
  }
  ORT_ENFORCE(nullptr != weight_tensor_proto,
              "Internal Error: weight tensor must be const int8 for Avx2WeightS8ToU8Transformer.");

  // Weight zero point must be either const int8_t or null tensor
  const ONNX_NAMESPACE::TensorProto* weight_zp_tensor_proto = nullptr;
  const auto* zp_def = input_defs.size() <= weight_zp_idx ? nullptr : input_defs[weight_zp_idx];
  if (nullptr != zp_def) {
    if (!graph_utils::NodeArgIsConstant(graph, *zp_def) ||
        !graph.GetInitializedTensor(zp_def->Name(), weight_zp_tensor_proto) ||
        weight_zp_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      return false;
    }
    ORT_ENFORCE(nullptr != weight_zp_tensor_proto,
                "Internal Error: weight zero point must be const int8 for Avx2WeightS8ToU8Transformer.");
  }

  // Convert weight tensor to uint8
  ONNX_NAMESPACE::TensorProto weights_proto_u8;
  bool converted = Int8TensorProto2Uint8(weight_tensor_proto, weights_proto_u8, graph);
  if (!converted) {
    // The weights fits into S7, overflow is not a problem, no need to convert to U8
    return false;
  }
  input_defs[weights_idx] = &graph_utils::AddInitializer(graph, weights_proto_u8);

  // Convert weight zero point to uint8
  ONNX_NAMESPACE::TensorProto weight_zp_proto_u8;
  Int8TensorProto2Uint8(weight_zp_tensor_proto, weight_zp_proto_u8, graph, true);
  input_defs[weight_zp_idx] = &graph_utils::AddInitializer(graph, weight_zp_proto_u8);

  return true;
}


struct OperatorWeightInfo {
  std::vector<ONNX_NAMESPACE::OperatorSetVersion> versions;
  const char* domain;
  const size_t weights_idx;
  const size_t weight_zp_idx;
};

static const std::unordered_map<std::string, struct OperatorWeightInfo> s8_overflow_ops = {
    {"QAttention", {{1}, kMSDomain, 1, 7}},
    {"MatMulIntegerToFloat", {{1}, kMSDomain, 1, 5}},
    {"DynamicQuantizeMatMul", {{1}, kMSDomain, 1, 3}},
    {"QGemm", {{1}, kMSDomain, 3, 5}},
    {"MatMulInteger", {{10}, kOnnxDomain, 1, 3}},
    {"QLinearMatMul", {{10}, kOnnxDomain, 3, 5}},
    {"QLinearConv", {{10}, kOnnxDomain, 3, 5}},
    /* {"ConvInteger", {10}, kOnnxDomain, 1, 3},  // ConvInteger does not support int8_t weight at all */
};

static inline bool MatchesOpSinceVersion(
    const Node& node, const std::vector<ONNX_NAMESPACE::OperatorSetVersion>& versions) {
  return std::find(versions.begin(), versions.end(), node.SinceVersion()) != versions.end();
}


static bool TryConvertDynamicQuantizeLSTM(Node& op_node, Graph& graph) {
  constexpr size_t w_idx = 1;
  constexpr size_t w_zp_idx = 9;
  constexpr size_t r_idx = 2;
  constexpr size_t r_zp_idx = 11;

  auto& input_defs = op_node.MutableInputDefs();
  if (input_defs.size() < 3) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* weight_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *input_defs[w_idx]) ||
      !graph.GetInitializedTensor(input_defs[w_idx]->Name(), weight_tensor_proto) ||
      weight_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    return false;
  }
  ORT_ENFORCE(nullptr != weight_tensor_proto,
              "Internal Error: weight tensor must be const int8 for Avx2WeightS8ToU8Transformer.");

  const ONNX_NAMESPACE::TensorProto* r_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *input_defs[r_idx]) ||
      !graph.GetInitializedTensor(input_defs[r_idx]->Name(), r_tensor_proto) ||
      r_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    LOGS_DEFAULT(WARNING) << "Unable transforming DynamicQuantizeLSTM operator,"
                          << " cannot locate recurrence tensor of const int8 type,"
                          << " int8 overflow might impact precision !";
    return false;
  }
  ORT_ENFORCE(nullptr != r_tensor_proto,
              "Internal Error: recurrence tensor must be const int8 for Avx2WeightS8ToU8Transformer.");

  const ONNX_NAMESPACE::TensorProto* weight_zp_tensor_proto = nullptr;
  const auto* zp_def = input_defs.size() <= w_zp_idx ? nullptr : input_defs[w_zp_idx];
  if (nullptr != zp_def) {
    if (!graph_utils::NodeArgIsConstant(graph, *zp_def) ||
        !graph.GetInitializedTensor(zp_def->Name(), weight_zp_tensor_proto) ||
        weight_zp_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        return false;
    }
    ORT_ENFORCE(nullptr != weight_zp_tensor_proto,
                "Internal Error: weight zero point must be const int8 for Avx2WeightS8ToU8Transformer.");
  }

  const ONNX_NAMESPACE::TensorProto* r_zp_tensor_proto = nullptr;
  const auto* rzp_def = input_defs.size() <= r_zp_idx ? nullptr : input_defs[r_zp_idx];
  if (nullptr != rzp_def) {
    if (!graph_utils::NodeArgIsConstant(graph, *input_defs[r_zp_idx]) ||
        !graph.GetInitializedTensor(input_defs[r_zp_idx]->Name(), r_zp_tensor_proto) ||
        r_zp_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      LOGS_DEFAULT(WARNING) << "Unable transforming DynamicQuantizeLSTM operator,"
                            << " unable to locate recurrence tensor or its zero point value,"
                            << " int8 overflow might impact precision !";
      return false;
    }
    ORT_ENFORCE(nullptr != r_zp_tensor_proto,
                "Internal Error: recurrence zero point must be const int8 for Avx2WeightS8ToU8Transformer.");
  }

  bool should_convert = false;
  Initializer w_temp(*weight_tensor_proto, graph.ModelPath());
  {
    int8_t* p = w_temp.data<int8_t>();
    for (int i = 0; i < w_temp.size(); i++) {
      if (*p < -64 || *p > 64) {
        should_convert = true;
      }
      *p ^= 0x80;
      p++;
    }
  }

  Initializer r_temp(*r_tensor_proto, graph.ModelPath());
  {
    int8_t* p = r_temp.data<int8_t>();
    for (int i = 0; i < r_temp.size(); i++) {
      if (*p < -64 || *p > 64) {
        should_convert = true;
      }
      *p ^= 0x80;
      p++;
    }
  }

  if (!should_convert) {
    // The weights fit into S7, overflow is not a problem, no need to convert to U8
    return false;
  }

  ONNX_NAMESPACE::TensorProto weights_proto_u8;
  weights_proto_u8.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  weights_proto_u8.set_name(weight_tensor_proto->name() + "_s8_2_u8");
  weights_proto_u8.mutable_dims()->CopyFrom(weight_tensor_proto->dims());
  weights_proto_u8.set_raw_data(w_temp.data<int8_t>(), w_temp.size());
  input_defs[w_idx] = &graph_utils::AddInitializer(graph, weights_proto_u8);

  ONNX_NAMESPACE::TensorProto weight_zp_proto_u8;
  Int8TensorProto2Uint8(weight_zp_tensor_proto, weight_zp_proto_u8, graph, true);
  input_defs[w_zp_idx] = &graph_utils::AddInitializer(graph, weight_zp_proto_u8);

  ONNX_NAMESPACE::TensorProto r_proto_u8;
  r_proto_u8.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  r_proto_u8.set_name(r_tensor_proto->name() + "_s8_2_u8");
  r_proto_u8.mutable_dims()->CopyFrom(r_tensor_proto->dims());
  r_proto_u8.set_raw_data(r_temp.data<int8_t>(), r_temp.size());
  input_defs[r_idx] = &graph_utils::AddInitializer(graph, r_proto_u8);

  ONNX_NAMESPACE::TensorProto r_zp_proto_u8;
  Int8TensorProto2Uint8(r_zp_tensor_proto, r_zp_proto_u8, graph, true);
  input_defs[r_zp_idx] = &graph_utils::AddInitializer(graph, r_zp_proto_u8);

  return true;
}


// For QAttention operator, if the weight is const int8, convert it to const uint8
Status Avx2WeightS8ToU8Transformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                       const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr)
      continue;  // node removed as part of an earlier fusion

    Node& op_node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(op_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedProvider(op_node, GetCompatibleExecutionProviders())) {
      continue;  // only care about CPU operators
    }

    if (graph_utils::IsSupportedOptypeVersionAndDomain(
        op_node, "DynamicQuantizeLSTM", {1}, kMSDomain)) {
      // This one has two set of quantized arguments
      modified |= TryConvertDynamicQuantizeLSTM(op_node, graph);
      continue;  // go on to next operator node
    }

    const auto it = s8_overflow_ops.find(op_node.OpType());
    if (it == s8_overflow_ops.end()) {
      continue;  // unknown operator, next
    }

    if (
#if !defined(ORT_MINIMAL_BUILD)
        !op_node.Op()->Deprecated() &&
#endif
        MatchesOpSinceVersion(op_node, it->second.versions) &&
        graph_utils::MatchesOpSetDomain(op_node, it->second.domain)) {
      modified |= ConvertS8WeightToU8(graph, op_node,
                                      it->second.weights_idx,
                                      it->second.weight_zp_idx);
      continue;  // finished with this op, next
    }

  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // x86 or x64
