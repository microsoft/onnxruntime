// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_matmul.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlMatMul::DnnlMatMul() {}

void DnnlMatMul::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();

  auto src_dims = sp.GetMemory(node.Input(IN_A)).get_desc().dims();
  auto weights_dims = sp.GetMemory(node.Input(IN_B)).get_desc().dims();

  if (src_dims.size() != weights_dims.size()) {
    while (src_dims.size() < weights_dims.size()) {
      src_dims.insert(src_dims.begin(), 1);
    }
    while (src_dims.size() > weights_dims.size()) {
      weights_dims.insert(weights_dims.begin(), 1);
    }
  }

  auto src_md = dnnl::memory::desc(src_dims, node.Input(IN_A).Type(), dnnl::memory::format_tag::any);
  auto weights_md = dnnl::memory::desc(weights_dims, node.Input(IN_B).Type(), dnnl::memory::format_tag::any);

  auto output_shape = src_dims;
  output_shape.pop_back();
  output_shape.emplace_back(weights_dims.back());
  for (size_t i = 0; i < output_shape.size() - 2; i++) {
    if (output_shape[i] == 1) {
      output_shape[i] = weights_dims[i];
    }
  }

  auto dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);

  auto matmul_d = dnnl::matmul::desc(src_md, weights_md, dst_md);
  auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, eng);

  auto matmul_src_mem = sp.GetMemoryAndReshape(node.Input(IN_A), matmul_pd.src_desc(), eng);
  auto matmul_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_B), matmul_pd.weights_desc(), eng);

  auto matmul_dst_mem = dnnl::memory(matmul_pd.dst_desc(), eng);
  auto matmul_prim = dnnl::matmul(matmul_pd);

  sp.AddPrimitive(matmul_prim, {{DNNL_ARG_SRC, matmul_src_mem},
                                {DNNL_ARG_WEIGHTS, matmul_weights_mem},
                                {DNNL_ARG_DST, matmul_dst_mem}});

  sp.SetMemory(node.Output(OUT_Y), matmul_dst_mem);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
