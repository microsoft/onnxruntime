// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License
#include "dnnl_reducemean.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {


DnnlReduceMean::DnnlReduceMean() {}

// assume all dims are available
void DnnlReduceMean::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {

  using namespace dnnl;

  // get the engine, currently only support either single gpu or single cpu device
  auto dnnl_engine = sp.GetEngine();

  auto axes = ReadAxes(node);
  
  auto reducemean_src_mem = sp.GetMemory(node.Input(IN_X));
  auto src_md = reducemean_src_mem.get_desc();

  //We need to calculate output tensor shape
  //First we initialize it with input shape and then we modify it based on the attribute values
  //This is because the DNNL primitive functionality is determined by the input and output shapes.
  
  auto src_dims = src_md.dims();
  auto ndim = src_dims.size();
  for (unsigned long int i = 0; i < ndim; i++) {
    if (axes.size() == 0)
      src_dims[i] = 1;  //If no axis is specified, then output shape is just all 1's
    else if (i < axes.size()) {
      if (axes[i] < 0)
        src_dims[ndim + axes[i]] = 1;
      else
        src_dims[axes[i]] = 1;
    }  //If there is axis, then make the respective dimensions 1, keeping the other dimension values untouched.
  }

  auto dst_shape = TensorShape(src_dims.data(), ndim);
  dnnl::memory::dims dst_dims_mkl(dst_shape.GetDims().begin(), dst_shape.GetDims().end());
  auto dst_md = dnnl::memory::desc({dst_dims_mkl}, src_md.data_type(), dnnl::memory::format_tag::any);

  auto reducemean_desc = dnnl::reduction::desc(dnnl::algorithm::reduction_mean, src_md, dst_md, 0.f, 0.f);
  auto reducemean_pd = dnnl::reduction::primitive_desc(reducemean_desc, dnnl_engine);

  // If using GPU this will move the memory from the CPU to the GPU.
  reducemean_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), reducemean_pd.src_desc(), dnnl_engine);
  auto reducemean_dst_mem = dnnl::memory(reducemean_pd.dst_desc(), dnnl_engine);

  auto reducemean_op = dnnl::reduction(reducemean_pd);
  sp.AddPrimitive(reducemean_op, {{DNNL_ARG_SRC, reducemean_src_mem},
                                  {DNNL_ARG_DST, reducemean_dst_mem}});

  sp.SetMemory(node.Output(OUT_Y), reducemean_dst_mem);
}

std::vector<int64_t> DnnlReduceMean::ReadAxes(DnnlNode& node) {
  auto attr = node.Attributes().find("axes");
  std::vector<int64_t> axes;
  if (attr != node.Attributes().end()) {
    auto& proto = attr->second();
    axes.reserve(proto.ints_size());
    for (int i = 0; i < proto.ints_size(); i++) {
      axes.push_back(proto.ints(i));
    }
  }
  return axes;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
