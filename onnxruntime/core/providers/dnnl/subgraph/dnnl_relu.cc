// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_relu.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlRelu::DnnlRelu() {}

void DnnlRelu::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {

  auto dnnl_engine = sp.GetEngine();

  auto relu_src_mem = sp.GetMemory(node.Input(IN_X).Name());
  auto src_md = relu_src_mem.get_desc();

  auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_relu, src_md);
  auto relu_pd = dnnl::eltwise_forward::primitive_desc(relu_desc, dnnl_engine);

  // If using GPU this will move the memory from the CPU to the GPU.
  relu_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), relu_pd.src_desc(), dnnl_engine);
  auto relu_dst_mem = dnnl::memory(relu_pd.dst_desc(), dnnl_engine);

  auto relu_op = dnnl::eltwise_forward(relu_pd);
  sp.AddPrimitive(relu_op, {{DNNL_ARG_SRC, relu_src_mem},
                        {DNNL_ARG_DST, relu_dst_mem}});

  sp.SetMemory(node.Output(OUT_Y), relu_dst_mem);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
