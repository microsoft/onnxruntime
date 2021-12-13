// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_squeeze.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace ort_dnnl {
DnnlSqueeze::DnnlSqueeze() {}

void DnnlSqueeze::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  // the input shape assumes OrtFormat so we get the memory in OrtFormat.
  auto data_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), dnnl_engine);
  dnnl::memory::dims data_dims = data_mem.get_desc().dims();

  std::vector<int64_t> axes_data;
  // ONNX Squeeze version 13+ the axes is an input tensor
  // ONNX Squeeze before version 13 axes comes from an Attribute.
  if (node.Input(IN_AXES).Exists()) {
    auto axes_mem = sp.GetMemory(node.Input(IN_AXES));
    dnnl::memory::dims axes_dims = axes_mem.get_desc().dims();
    int64_t* p_axes_data = (int64_t*)axes_mem.get_data_handle();
    axes_data = std::vector<int64_t>(p_axes_data, p_axes_data + axes_dims[0]);
  } else {
    axes_data = GetAxes(node);
  }

  // convert negative axis to the positive axis
  for (size_t i = 0; i < axes_data.size(); ++i) {
    axes_data[i] = HandleNegativeAxis(axes_data[i], data_dims.size());
  }

  // Handle out of order and repeating dims.
  std::sort(axes_data.begin(), axes_data.end());
  axes_data.erase(std::unique(axes_data.begin(), axes_data.end()), axes_data.end());

  std::vector<int64_t> output_shape;
  size_t j = 0;
  for (size_t i = 0; i < data_dims.size(); ++i) {
    if ((j < axes_data.size() && axes_data[j] == static_cast<int64_t>(i)) ||
        (axes_data.size() == 0 && data_dims[i] == 1)) {
      ORT_ENFORCE(data_dims[i] == 1, "Dimension of input ", i, " must be 1 instead of ", data_dims[i],
                  ". shape=", data_dims);
      ++j;
      continue;
    }
    output_shape.push_back(data_dims[i]);
  }

  dnnl::memory::desc squeeze_md(output_shape, node.Input(IN_DATA).Type(), sp.GetDnnlFormat(output_shape.size()));

  dnnl::memory squeeze_mem = dnnl::memory(squeeze_md, dnnl_engine, nullptr);
  sp.AddReshape(data_mem, squeeze_mem);

  sp.SetMemory(node.Output(OUT_SQUEEZED), squeeze_mem, true);
}

std::vector<int64_t> DnnlSqueeze::GetAxes(DnnlNode& node) {
  auto attr = node.Attributes().find("axes");
  std::vector<int64_t> axes;
  if (attr != node.Attributes().end() && 
      attr->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS) {
    axes.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      axes.push_back(attr->second().ints(i));
    }
  } 
  return axes;
}
}  // namespace ort_dnnl
}  // namespace onnxruntime
