// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"

#include <vector>

#include "core/common/common.h"
#include "core/common/span_utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime::QDQ {

bool IsQDQPairSupported(
    const Node& q_node, const Node& dq_node,
    const GetConstantInitializerFn& get_const_initializer,
    const std::filesystem::path& model_path,
    bool check_op_type) {
  if (check_op_type) {
    if (!MatchQNode(q_node) || !MatchDQNode(dq_node)) {
      return false;
    }
  }

  ConstPointerContainer<std::vector<NodeArg*>> dq_input_defs = dq_node.InputDefs();
  ConstPointerContainer<std::vector<NodeArg*>> q_input_defs = q_node.InputDefs();

  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != InputIndex::TOTAL_COUNT ||
      q_input_defs.size() != InputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::ZERO_POINT_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == q_zp_tensor_proto ||
      nullptr == dq_zp_tensor_proto ||
      nullptr == q_scale_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;
  }

  // check Q/DQ have same scale and zero point
  Initializer q_zp(*q_zp_tensor_proto, model_path);
  Initializer q_scale(*q_scale_tensor_proto, model_path);
  Initializer dq_zp(*dq_zp_tensor_proto, model_path);
  Initializer dq_scale(*dq_scale_tensor_proto, model_path);

  if (q_zp.data_type() != dq_zp.data_type() ||
      q_scale.data_type() != dq_scale.data_type() ||
      !SpanEq(q_zp.DataAsByteSpan(), dq_zp.DataAsByteSpan())) {
    return false;
  }

  switch (q_scale.data_type()) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      return *q_scale.data<float>() == *dq_scale.data<float>();

    case ONNX_NAMESPACE::TensorProto::FLOAT16:
      return *q_scale.data<MLFloat16>() == *dq_scale.data<MLFloat16>();

    case ONNX_NAMESPACE::TensorProto::BFLOAT16:
      return *q_scale.data<BFloat16>() == *dq_scale.data<BFloat16>();

    default:
      assert(false);
      return false;
  }
}

bool IsDQQConversion(
    const Node& dq_node, const Node& q_node,
    const GetConstantInitializerFn& get_const_initializer,
    const std::filesystem::path& model_path) {
  ConstPointerContainer<std::vector<NodeArg*>> dq_input_defs = dq_node.InputDefs();
  ConstPointerContainer<std::vector<NodeArg*>> q_input_defs = q_node.InputDefs();

  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != InputIndex::TOTAL_COUNT ||
      q_input_defs.size() != InputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::ZERO_POINT_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == q_zp_tensor_proto ||
      nullptr == dq_zp_tensor_proto ||
      nullptr == q_scale_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;
  }

  // check Q/DQ have same scale type and different zero point type
  Initializer q_zp(*q_zp_tensor_proto, model_path);
  Initializer q_scale(*q_scale_tensor_proto, model_path);
  Initializer dq_zp(*dq_zp_tensor_proto, model_path);
  Initializer dq_scale(*dq_scale_tensor_proto, model_path);

  return (dq_zp.data_type() != q_zp.data_type()) && (dq_scale.data_type() == q_scale.data_type());
}

bool IsDQSupported(const Node& dq_node, const GetConstantInitializerFn& get_const_initializer) {
  bool zero_point_exists = false;
  if (!QOrDQNodeHasConstantScalarScaleAndZeroPoint(dq_node, get_const_initializer, zero_point_exists)) {
    return false;
  }

  if (!zero_point_exists) {
    return false;
  }

  return true;
}

bool QOrDQNodeHasConstantScalarScaleAndZeroPoint(
    const Node& q_or_dq_node,
    const GetConstantInitializerFn& get_const_initializer,
    bool& zero_point_exists) {
  auto q_or_dq_input_defs = q_or_dq_node.InputDefs();

  ORT_ENFORCE(q_or_dq_input_defs.size() >= 2);

  zero_point_exists = q_or_dq_input_defs.size() > 2 &&
                      q_or_dq_input_defs[InputIndex::ZERO_POINT_ID]->Exists();

  auto is_constant_scalar = [&](const NodeArg& node_arg) {
    return optimizer_utils::IsScalar(node_arg) && get_const_initializer(node_arg.Name()) != nullptr;
  };

  if (!is_constant_scalar(*q_or_dq_input_defs[InputIndex::SCALE_ID])) {
    return false;
  }

  if (zero_point_exists &&
      !is_constant_scalar(*q_or_dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  return true;
}

bool IsQOrDQScalePositiveConstantScalar(
    const Node& q_or_dq_node, const GetConstantInitializerFn& get_const_initializer,
    const std::filesystem::path& model_path) {
  auto q_or_dq_input_defs = q_or_dq_node.InputDefs();

  ORT_ENFORCE(q_or_dq_input_defs.size() >= 2);

  if (!optimizer_utils::IsScalar(*q_or_dq_input_defs[InputIndex::SCALE_ID])) {
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* q_or_dq_scale_tensor_proto =
      get_const_initializer(q_or_dq_input_defs[InputIndex::SCALE_ID]->Name());
  if (nullptr == q_or_dq_scale_tensor_proto) {
    return false;
  }

  Initializer q_or_dq_scale(*q_or_dq_scale_tensor_proto, model_path);

  switch (q_or_dq_scale.data_type()) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      return q_or_dq_scale.data<float>()[0] > 0;

    case ONNX_NAMESPACE::TensorProto::FLOAT16:
      return q_or_dq_scale.data<MLFloat16>()[0] > 0;

    case ONNX_NAMESPACE::TensorProto::BFLOAT16:
      return q_or_dq_scale.data<BFloat16>()[0] > 0;

    default:
      assert(false);
      return false;
  }
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

bool MatchQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, QOpName, {10, 13, 19, 21}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, QOpName, {1}, kMSDomain);
}

bool MatchDQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, DQOpName, {10, 13, 19, 21}) ||
         graph_utils::IsSupportedOptypeVersionAndDomain(node, DQOpName, {1}, kMSDomain);
}

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

namespace {

bool GetDataTypeMinMax(int32_t data_type, int32_t& min, int32_t& max) {
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::INT8:
      min = static_cast<int32_t>(std::numeric_limits<int8_t>::min());
      max = static_cast<int32_t>(std::numeric_limits<int8_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto::UINT8:
      min = static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
      max = static_cast<int32_t>(std::numeric_limits<uint8_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto::INT16:
      min = static_cast<int32_t>(std::numeric_limits<int16_t>::min());
      max = static_cast<int32_t>(std::numeric_limits<int16_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto::UINT16:
      min = static_cast<int32_t>(std::numeric_limits<uint16_t>::min());
      max = static_cast<int32_t>(std::numeric_limits<uint16_t>::max());
      break;
    default:
      return false;
  }
  return true;
}
bool GetQScalarScaleZp(const Graph& graph, const Node& q_node, float& scale, int32_t& zp, int32_t& data_type) {
  assert(q_node.OpType() == QOpName);
  const auto& q_input_defs = q_node.InputDefs();

  const ONNX_NAMESPACE::TensorProto* scale_tensor_proto = graph.GetConstantInitializer(q_input_defs[1]->Name(), true);
  if (!scale_tensor_proto) {
    return false;
  }

  // Support scalar float scale only for now. Need to extend to other float types if needed.
  Initializer scale_initializer(*scale_tensor_proto, graph.ModelPath());
  if (scale_initializer.dims().size() != 0 || scale_initializer.data_type() != ONNX_NAMESPACE::TensorProto::FLOAT) {
    return false;
  }

  scale = *scale_initializer.data<float>();

  if (q_input_defs.size() != 3 || !q_input_defs[2]->Exists()) {
    int32_t output_dtype = ONNX_NAMESPACE::TensorProto::UNDEFINED;
    const auto& q_attrs = q_node.GetAttributes();
    if (auto it = q_attrs.find("output_dtype"); it != q_attrs.end()) {
      output_dtype = static_cast<int32_t>(it->second.i());
    }

    data_type =
        output_dtype == ONNX_NAMESPACE::TensorProto::UNDEFINED ? ONNX_NAMESPACE::TensorProto::UINT8 : output_dtype;
    zp = 0;
    return true;
  }

  const ONNX_NAMESPACE::TensorProto* zp_tensor_proto = graph.GetConstantInitializer(q_input_defs[2]->Name(), true);
  if (!zp_tensor_proto) {
    return false;
  }

  Initializer zp_initializer(*zp_tensor_proto, graph.ModelPath());
  if (zp_initializer.dims().size() != 0) {
    return false;
  }

  data_type = zp_initializer.data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::INT8:
      zp = static_cast<int32_t>(*zp_initializer.data<int8_t>());
      break;
    case ONNX_NAMESPACE::TensorProto::UINT8:
      zp = static_cast<int32_t>(*zp_initializer.data<uint8_t>());
      break;
    case ONNX_NAMESPACE::TensorProto::INT16:
      zp = static_cast<int32_t>(*zp_initializer.data<int16_t>());
      break;
    case ONNX_NAMESPACE::TensorProto::UINT16:
      zp = static_cast<int32_t>(*zp_initializer.data<uint16_t>());
      break;
    default:
      return false;
  }

  return true;
}

}  // namespace

bool IsClipMadeRedundantByQ(const Graph& graph, const Node& clip_node, const Node& q_node) {
  float scale = 0.0f;
  int32_t zp = 0;
  int32_t data_type = 0;
  if (!GetQScalarScaleZp(graph, q_node, scale, zp, data_type)) {
    return false;
  }

  int32_t data_type_min = 0;
  int32_t data_type_max = 0;
  if (!GetDataTypeMinMax(data_type, data_type_min, data_type_max)) {
    return false;
  }

  const std::string& clip_op_type = clip_node.OpType();
  if (clip_op_type == "Relu") {
    return zp == data_type_min;
  }

  if (clip_op_type == "Clip") {
    float clip_min = 0.0f;
    float clip_max = 0.0f;
    if (!optimizer_utils::GetClipConstantMinMax(graph, clip_node, clip_min, clip_max)) {
      return false;
    }

    int32_t q_clip_min = static_cast<int32_t>(::rint(clip_min / scale)) + zp;
    int32_t q_clip_max = static_cast<int32_t>(::rint(clip_max / scale)) + zp;

    // The Clip can be removed if its range entirely overlaps the quantization range.
    // QClip range:    [------------------]
    // Quant range:      [-------------]
    return q_clip_min <= data_type_min && q_clip_max >= data_type_max;
  }

  return false;
}

}  // namespace onnxruntime::QDQ
