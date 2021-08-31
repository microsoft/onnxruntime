// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_reshape.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace ort_dnnl {
DnnlReshape::DnnlReshape() { }

void DnnlReshape::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  auto data_mem = sp.GetMemory(node.Input(IN_DATA));
  dnnl::memory::dims data_dims = data_mem.get_desc().dims();
  auto data_md = data_mem.get_desc();


  if (!IsMemoryInExpectedOrtFormat(data_md)) {
    auto temp_md = dnnl::memory::desc(data_dims, node.Input(IN_DATA).Type(), sp.GetDnnlFormat(data_dims.size()));
    dnnl::memory temp_mem = dnnl::memory(temp_md, dnnl_engine);
    sp.AddPrimitive(dnnl::reorder(data_mem, temp_mem), {{DNNL_ARG_FROM, data_mem},
                                                           {DNNL_ARG_TO, temp_mem}});
    data_mem = temp_mem;
  } else {
    // If using GPU this will move the memory from the CPU to the GPU.
    data_mem = sp.GetMemoryAndReshape(node.Input(IN_DATA), data_md, dnnl_engine);
  }

  auto shape_mem = sp.GetMemory(node.Input(IN_SHAPE));
  dnnl::memory::dims shape_dims = shape_mem.get_desc().dims();
  int64_t* shape_data = (int64_t*)shape_mem.get_data_handle();

  dnnl::memory::dims reshape_shape(shape_data, shape_data + shape_dims[0]);
  // Reshape helper will take input data_dims shape and the reshape_shape and replace the -1 and 0s with the calculated
  // Output values. The Reshape helper also does a lot of error checking to make sure the Reshape is possible.
  ReshapeHelper helper(TensorShape(data_dims), reshape_shape, GetAllowZero(node));

  //the dnnl::memory::desc.reshape(shape) failed on some models so we instead create a new dnnl:memory::desc
  dnnl::memory::desc reshaped_md(reshape_shape, node.Input(IN_DATA).Type(), sp.GetDnnlFormat(reshape_shape.size()));

  dnnl::memory reshaped_mem = dnnl::memory(reshaped_md, dnnl_engine, nullptr);
  sp.AddReshape(data_mem, reshaped_mem);

  sp.SetMemory(node.Output(OUT_RESHAPED), reshaped_mem, true);
}

bool DnnlReshape::IsMemoryInExpectedOrtFormat(const dnnl::memory::desc& desc) {
  if (desc.data.format_kind != dnnl_blocked) {
    return false;
  }
  if (desc.data.format_desc.blocking.inner_nblks != 0) {
    return false;
  }
  auto strides = desc.data.format_desc.blocking.strides;
  // if a data format is dnnl_format::abcd... the stride will go from largest to smallest
  // if for example we have a shape {2,3,4} we expect a stride of {12, 4, 1} if it were
  // of dnnl_format::abc if instead the stride were {12, 1, 4} that would be dnnl_format::acb
  // which does not match what is expected from Onnxruntime.
  for (size_t i = 1; i < desc.dims().size(); ++i) {
    if (strides[i - 1] < strides[i]) {
      return false;
    }
  }
  return true;
}

bool DnnlReshape::GetAllowZero(DnnlNode& node) {
  auto attr = node.Attributes().find("allowzero");
  int64_t allowzero = 0;  //Default value according to ONNX spec
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
    allowzero = attr->second().i();
  }
  return !(allowzero == 0);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
