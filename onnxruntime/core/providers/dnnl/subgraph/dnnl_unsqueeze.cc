// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_unsqueeze.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace ort_dnnl {
DnnlUnsqueeze::DnnlUnsqueeze() {}

void DnnlUnsqueeze::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  // the input shape assumes OrtFormat so we get the memory in OrtFormat.
  auto data_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), dnnl_engine);
  bool data_is_scalar = sp.IsScalar(node.Input(IN_DATA));

  // The OneDNN execution provider automatically expands all scalar inputs to dim {1} tensors.
  // this will result in the data_dims.size() being 1 too large if the input is from a scalar.
  // To counter this data_dims is left empty if the input is from a scalar.
  dnnl::memory::dims data_dims;
  if (!data_is_scalar) {
    data_dims = data_mem.get_desc().dims();
  }

  std::vector<int64_t> axes_data;
  // ONNX Unsqueeze version 13+ the axes is an input tensor
  // ONNX Unsqueeze before version 13 axes comes from an Attribute.
  if (node.Input(IN_AXES).Exists()) {
    auto axes_mem = sp.GetMemory(node.Input(IN_AXES));
    dnnl::memory::dims axes_dims = axes_mem.get_desc().dims();
    int64_t* p_axes_data = (int64_t*)axes_mem.get_data_handle();
    axes_data = std::vector<int64_t>(p_axes_data, p_axes_data + axes_dims[0]);
  } else {
    axes_data = GetAxes(node);
  }

  std::vector<int64_t> output_shape(axes_data.size() + data_dims.size(), 0);
  // Set all axes indices to 1 in output_dims and check for duplicates
  for (int64_t axes : axes_data) {
    // Valid axis range is [0, output_rank - 1]
    axes = HandleNegativeAxis(axes, output_shape.size());
    if (axes < 0 || axes >= static_cast<int64_t>(output_shape.size()))
      ORT_ENFORCE("'axes' has an out of range axis");
    if (output_shape[axes] != 0)
      ORT_ENFORCE("'axes' has a duplicate axis");
    output_shape[axes] = 1;
  }

  // Now fill in the zero entries with the existing shape
  {
    auto begin = data_dims.cbegin();
    for (auto& axisSize : output_shape) {
      if (axisSize == 0)
        axisSize = *begin++;
    }
    assert(begin == data_dims.cend());
  }

  dnnl::memory::desc squeeze_md(output_shape, node.Input(IN_DATA).Type(), sp.GetDnnlFormat(output_shape.size()));

  dnnl::memory expanded_mem = dnnl::memory(squeeze_md, dnnl_engine, nullptr);
  sp.AddReshape(data_mem, expanded_mem);

  sp.SetMemory(node.Output(OUT_EXPANDED), expanded_mem, true);
}

std::vector<int64_t> DnnlUnsqueeze::GetAxes(DnnlNode& node) {
  auto attr = node.Attributes().find("axes");
  std::vector<int64_t> axes;
  if (attr != node.Attributes().end() && 
      attr->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS) {
    axes.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      axes.push_back(attr->second().ints(i));
    }
  } else {
    ORT_ENFORCE("Missing/Invalid 'axes' attribute value");
  } 
  return axes;
}
}  // namespace ort_dnnl
}  // namespace onnxruntime
