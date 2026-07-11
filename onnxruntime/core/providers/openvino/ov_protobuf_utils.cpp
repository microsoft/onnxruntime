// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_protobuf_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace openvino_ep {
float get_float_initializer_data(const void* initializer) {
  const auto* tp = reinterpret_cast<const ONNX_NAMESPACE::TensorProto*>(initializer);
  ORT_ENFORCE((tp->has_data_type() && (tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));
  // ORT_ENFORCE(initializer.dims_size() == 1);
  return tp->float_data(0);
}
void set_float_initializer_data(const void* initializer, float data) {
  auto* tp = (ONNX_NAMESPACE::TensorProto*)(initializer);
  ORT_ENFORCE((tp->has_data_type() && (tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)));
  // ORT_ENFORCE(initializer.dims_size() == 1);
  tp->set_float_data(0, data);
}

void normalize_negative_resize_axes(void* model_proto, const TensorRankMap& tensor_ranks) {
  auto* model = reinterpret_cast<ONNX_NAMESPACE::ModelProto*>(model_proto);
  ORT_ENFORCE(model != nullptr);

  TensorRankMap resolved_tensor_ranks = tensor_ranks;
  const auto add_value_info_rank = [&resolved_tensor_ranks](const ONNX_NAMESPACE::ValueInfoProto& value_info) {
    if (value_info.has_type() && value_info.type().has_tensor_type() &&
        value_info.type().tensor_type().has_shape()) {
      resolved_tensor_ranks[value_info.name()] = value_info.type().tensor_type().shape().dim_size();
    }
  };

  const auto& graph = model->graph();
  for (const auto& input : graph.input()) {
    add_value_info_rank(input);
  }
  for (const auto& output : graph.output()) {
    add_value_info_rank(output);
  }
  for (const auto& value_info : graph.value_info()) {
    add_value_info_rank(value_info);
  }
  for (const auto& initializer : graph.initializer()) {
    resolved_tensor_ranks[initializer.name()] = initializer.dims_size();
  }

  int64_t onnx_opset = 0;
  for (const auto& opset_import : model->opset_import()) {
    if (opset_import.domain().empty() || opset_import.domain() == "ai.onnx") {
      onnx_opset = opset_import.version();
      break;
    }
  }

  if (onnx_opset < 18) {
    return;
  }

  for (auto& node : *model->mutable_graph()->mutable_node()) {
    if (node.op_type() != "Resize" ||
        (!node.domain().empty() && node.domain() != "ai.onnx") ||
        node.input_size() == 0) {
      continue;
    }

    auto* axes = static_cast<ONNX_NAMESPACE::AttributeProto*>(nullptr);
    for (auto& attribute : *node.mutable_attribute()) {
      if (attribute.name() == "axes") {
        axes = &attribute;
        break;
      }
    }

    if (axes == nullptr) {
      continue;
    }

    bool has_negative_axis = false;
    for (int64_t axis : axes->ints()) {
      has_negative_axis = has_negative_axis || axis < 0;
    }
    if (!has_negative_axis) {
      continue;
    }

    const auto rank_it = resolved_tensor_ranks.find(node.input(0));
    ORT_ENFORCE(rank_it != resolved_tensor_ranks.end(),
                "Cannot normalize negative Resize axes: input rank is unavailable for '", node.input(0), "'.");
    const int64_t rank = rank_it->second;
    ORT_ENFORCE(rank > 0, "Rank must be positive when axes is provided.");
    InlinedHashSet<int64_t> seen;
    for (int index = 0; index < axes->ints_size(); ++index) {
      const int64_t raw_axis = axes->ints(index);
      ORT_ENFORCE(raw_axis >= -rank && raw_axis < rank,
                  "axis ", raw_axis, " is not in valid range [-", rank, ",", rank - 1, "]");
      const int64_t normalized_axis = raw_axis < 0 ? raw_axis + rank : raw_axis;
      ORT_ENFORCE(seen.insert(normalized_axis).second,
                  "axes attribute contains duplicate axis ", normalized_axis,
                  " after negative-axis normalization (rank=", rank, ").");
      axes->set_ints(index, normalized_axis);
    }
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
