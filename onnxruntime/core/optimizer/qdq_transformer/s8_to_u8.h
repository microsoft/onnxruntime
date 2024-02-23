// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime::QDQ {

/**
 * @brief Convert the source int8_t TensorProto to a uint8_t one if the tensor
 *        contains values outside of [-64, 64]
 * @param src     The source tensor, must be type int8_t
 * @param dst     An empty tensor, will contain the converted tensor data
 * @param graph   Graph for generating tensor name or provide external
 *                data path
 * @param force   Perform conversion even when tensor values within [-64, 64]
 * @return        Whether the conversion happened.
 */
inline bool Int8TensorProto2Uint8(
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
  for (size_t i = 0; i < temp.size(); i++) {
    if (*p < -64 || *p > 64) {
      should_convert = true;
    }
    *p ^= 0x80;
    p++;
  }
  if (force || should_convert) {
    dst.set_raw_data(temp.data<int8_t>(), size_t(temp.size()));
    return true;
  }
  return false;
}

/**
 * @brief If the op_node has an single int8_t const weight tensor, convert it to uint8_t
 * @param graph
 * @param op_node
 * @param weights_idx     input index of the weight tensor
 * @param weight_zp_idx   input index of the weight zero point tensor
 * @return true when conversion happened.
 */
extern bool ConvertS8WeightToU8(Graph& graph, Node& op_node,
                                size_t weights_idx, size_t weight_zp_idx);

}  // namespace onnxruntime::QDQ
