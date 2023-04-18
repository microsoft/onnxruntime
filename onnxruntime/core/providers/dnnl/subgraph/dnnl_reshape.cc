// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_reshape.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace ort_dnnl {
DnnlReshape::DnnlReshape() {}

void DnnlReshape::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  // the input shape assumes OrtFormat so we get the memory in OrtFormat.
  auto data_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), dnnl_engine);
  dnnl::memory::dims data_dims = data_mem.get_desc().get_dims();

  auto shape_mem = sp.GetMemory(node.Input(IN_SHAPE));
  dnnl::memory::dims shape_dims = shape_mem.get_desc().get_dims();
  int64_t* shape_data = (int64_t*)shape_mem.get_data_handle();

  // Reshape helper will take input data_dims shape and the reshape_shape and replace the -1 and 0s with the calculated
  // Output values. The Reshape helper also does a lot of error checking to make sure the Reshape is possible.
  const auto data_dims_span = gsl::span<const int64_t>(data_dims.data(), data_dims.size());
  TensorShapeVector reshape_shape(shape_data, shape_data + shape_dims[0]);
  ReshapeHelper helper(TensorShape(data_dims_span), reshape_shape, GetAllowZero(node));

  dnnl::memory::dims reshape_shape_dims(reshape_shape.cbegin(), reshape_shape.cend());
  // the dnnl::memory::desc.reshape(shape) failed on some models so we instead create a new dnnl:memory::desc
  dnnl::memory::desc reshaped_md(reshape_shape_dims, node.Input(IN_DATA).Type(), sp.GetDnnlFormat(reshape_shape.size()));

  dnnl::memory reshaped_mem = dnnl::memory(reshaped_md, dnnl_engine, nullptr);
  sp.AddReshape(data_mem, reshaped_mem);

  sp.SetMemory(node.Output(OUT_RESHAPED), reshaped_mem, true);
}

bool DnnlReshape::GetAllowZero(DnnlNode& node) {
  auto attr = node.Attributes().find("allowzero");
  int64_t allowzero = 0;  // Default value according to ONNX spec
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
    allowzero = attr->second().i();
  }
  return !(allowzero == 0);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
