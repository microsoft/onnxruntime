// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_transpose.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

#include <iostream>

namespace onnxruntime {
namespace ort_dnnl {

DnnlTranspose::DnnlTranspose() {}

/*
Transpose: 
  Inputs:
    0) DATA - Input Tensor
  Outputs:
    0) TRANSPOSED - Output Tensor

    (DATA)     +-----------+ (TRANSPOSED)
    ---------->+ Transpose +-------------->
               +-----------+

Attributes (perm) - A list of integers. By default, reverse the dimensions,
                    otherwise permute the axes according to the values given.
*/
void DnnlTranspose::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {

  auto dnnl_engine = sp.GetEngine();

  auto data_mem = sp.GetMemory(node.Input(IN_DATA));
  auto data_dims = data_mem.get_desc().dims();
  auto ndata_dims = data_dims.size();

  auto perm = GetPerm(node);
  if (perm.size() == 0) {
    perm.reserve(ndata_dims);
    for (size_t i = 0; i < ndata_dims; ++i) {
      perm.push_back(static_cast<int64_t>(ndata_dims - i - 1));
    }
  }

  dnnl::memory::dims transposed_dims(ndata_dims, 0);
  dnnl::memory::dims strides(ndata_dims, 0);
  dnnl::memory::dim total_stride = 1;
  for (int i = (int)ndata_dims - 1 ; i >= 0; i--) {
    transposed_dims[i] = data_dims[perm[i]];
    strides[perm[i]] = total_stride;
    total_stride *= data_dims[perm[i]];
  }

  dnnl::memory::dims strides_inverse;
  strides_inverse.reserve(ndata_dims);
  for (size_t i = 0; i < ndata_dims; ++i) {
    strides_inverse.push_back(strides[ndata_dims - i - 1]);
  }

  // Memory descriptor describes the memory reorder but will not have the correct output dimentions or the correct dnnl::memory::format
  dnnl::memory::desc intermediate_md = dnnl::memory::desc(data_dims, node.Input(IN_DATA).Type(), strides);
  dnnl::memory intermediate_mem = dnnl::memory(intermediate_md, dnnl_engine);

  auto traspose_primitive = dnnl::reorder(data_mem, intermediate_mem);
  sp.AddPrimitive(traspose_primitive, {{DNNL_ARG_FROM, data_mem},
                                       {DNNL_ARG_TO, intermediate_mem}});

  // The reorder from above will get the memory in the right order. The next few lines will create a memory and memory descriptor
  // that will have the correct dimentions and correct memory::format
  dnnl::memory::desc transposed_md = dnnl::memory::desc(transposed_dims, node.Input(IN_DATA).Type(), sp.GetDnnlFormat(data_dims.size()));
  dnnl::memory transposed_mem = dnnl::memory(transposed_md, dnnl_engine, nullptr);
  void* handle = intermediate_mem.get_data_handle();
  transposed_mem.set_data_handle(handle);

  sp.SetMemory(node.Output(OUT_TRANSPOSED), transposed_mem, true);
}

std::vector<int64_t> DnnlTranspose::GetPerm(DnnlNode& node) {
  auto attr = node.Attributes().find("perm");
  std::vector<int64_t> perm;
  if (attr != node.Attributes().end()) {
    perm.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      perm.push_back(attr->second().ints(i));
    }
  }
  return perm;
}
}  // namespace ort_dnnl
}  // namespace onnxruntime
