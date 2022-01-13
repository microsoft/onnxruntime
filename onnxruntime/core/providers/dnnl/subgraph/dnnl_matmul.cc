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

  bool has_add = false;
  if (node.OpType() == "MatMulAdd") {
    has_add = true;
    //if fused with add, need a third input
    assert(node.Input(IN_BINARY).Exists());
  }

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

  /*
  create a post op binary with possible unsqueezing in order to make sure onednn properly broadcast
  current limitation 
  1. is no unsqueeze for matmul output as it is not exposed due to post op fusion
  2. the third input has to be reordered to plain format (eg, no memory format propogation if the third input is internal to subgraph)
  3. adding 1s to front (unsqueeze/expand) in logical dims would possibly fail if physcial layout is not plain format
  */
  dnnl::primitive_attr attr;
  if (has_add) {
    dnnl::post_ops ops;
    auto ori_binary_mem_desc = sp.GetMemory(node.Input(IN_BINARY).Name()).get_desc();
    auto ori_binary_mem_dims = ori_binary_mem_desc.dims();
    auto binary_mem_dims = ori_binary_mem_dims;
    if (ori_binary_mem_dims.size() != output_shape.size()) {
      if (ori_binary_mem_dims.size() > output_shape.size()) {
        ORT_THROW("add fusion with matmul output broadcasting by unsqueezing is not supported");
      }
      //expand the third input (from the binary op) is possible
      while (binary_mem_dims.size() < output_shape.size()) {
        binary_mem_dims.insert(binary_mem_dims.begin(), 1);
      }
    }

    //expand the dims by 1s (should always be possible)
    //will throw exception if not possible
    auto binary_mem_desc = ori_binary_mem_desc.reshape(binary_mem_dims);
    //TODO: use format any to choose the best layout
    ops.append_binary(dnnl::algorithm::binary_add, binary_mem_desc);
    attr.set_post_ops(ops);
  }

  auto dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);

  auto matmul_d = dnnl::matmul::desc(src_md, weights_md, dst_md);
  auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, eng);

  auto matmul_src_mem = sp.GetMemoryAndReshape(node.Input(IN_A), matmul_pd.src_desc(), eng);
  auto matmul_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_B), matmul_pd.weights_desc(), eng);
  auto matmul_dst_mem = dnnl::memory(matmul_pd.dst_desc(), eng);
  auto matmul_prim = dnnl::matmul(matmul_pd);

  //a default memory map for matmul
  std::unordered_map<int, dnnl::memory> mem_map({{DNNL_ARG_SRC, matmul_src_mem},
                                                 {DNNL_ARG_WEIGHTS, matmul_weights_mem},
                                                 {DNNL_ARG_DST, matmul_dst_mem}});

  //add to memory map with extra third input if fused with add
  if (has_add) {
    dnnl::algorithm algo;
    dnnl::memory::desc binary_mem_desc;
    matmul_pd.get_primitive_attr().get_post_ops().get_params_binary(0, algo, binary_mem_desc);
    assert(algo == dnnl::algorithm::binary_add);
    auto binary_post_op_mem = sp.GetMemoryAndReshape(node.Input(IN_BINARY), binary_mem_desc, eng);
    mem_map[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = binary_post_op_mem;
  }

  sp.AddPrimitive(matmul_prim, mem_map);

  sp.SetMemory(node.Output(OUT_Y), matmul_dst_mem);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
