// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_gelu.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "dnnl_util.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlGelu::DnnlGelu() {}

void DnnlGelu::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  bool is_biased = node.Input(IN_BIAS).Exists();
  dnnl::memory src_mem;
  if (is_biased) {
    src_mem = sp.GetMemoryInOrtFormat(node.Input(IN_X), dnnl_engine);
  } else {
    src_mem = sp.GetMemory(node.Input(IN_X));
  }
  auto gelu_src_mem = src_mem;
  dnnl::memory dst_mem;
  if (is_biased) {
    auto bias_mem = sp.GetMemoryInOrtFormat(node.Input(IN_BIAS), dnnl_engine);
    auto src0_ori_md = src_mem.get_desc();
    auto src1_ori_md = bias_mem.get_desc();

    auto src0_dims = src0_ori_md.get_dims();
    auto src1_dims = src1_ori_md.get_dims();
    if (src0_dims.size() != src1_dims.size()) {
      while (src0_dims.size() < src1_dims.size()) {
        src0_dims.insert(src0_dims.begin(), 1);
      }
      while (src0_dims.size() > src1_dims.size()) {
        src1_dims.insert(src1_dims.begin(), 1);
      }
    }

    auto src0_md = src0_ori_md.reshape(src0_dims);
    auto src1_md = src1_ori_md.reshape(src1_dims);

    auto output_shape = src0_dims;
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (output_shape[i] == 1) {
        output_shape[i] = src1_dims[i];
      }
    }

    dnnl::primitive_attr attr;
    dnnl::post_ops ops;
    dnnl::algorithm algo = dnnl_util::OrtOperatorToDnnlAlgorithm(node.OpType());
    ops.append_eltwise(algo, 1.0f, 1.0f);
    attr.set_post_ops(ops);

    auto dst_md = dnnl::memory::desc(output_shape, node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);

    auto binary_pd = dnnl::binary::primitive_desc(dnnl_engine, dnnl::algorithm::binary_add, src0_md, src1_md, dst_md, attr);

    dst_mem = dnnl::memory(binary_pd.dst_desc(), dnnl_engine);
    auto binary_prim = dnnl::binary(binary_pd);

    sp.AddPrimitive(binary_prim, {{DNNL_ARG_SRC_0, src_mem},
                                  {DNNL_ARG_SRC_1, bias_mem},
                                  {DNNL_ARG_DST, dst_mem}});
  } else {
    auto dst_md = dnnl::memory::desc(src_mem.get_desc().get_dims(),
                                     node.Output(OUT_Y).Type(),
                                     dnnl::memory::format_tag::any);
    dnnl::algorithm algo = dnnl_util::OrtOperatorToDnnlAlgorithm(node.OpType());
    auto gelu_pd = dnnl::eltwise_forward::primitive_desc(dnnl_engine, dnnl::prop_kind::forward_inference, algo,
                                                         gelu_src_mem.get_desc(), dst_md);

    // If using GPU this will move the memory from the CPU to the GPU.
    gelu_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), gelu_pd.src_desc(), dnnl_engine);

    dst_mem = dnnl::memory(gelu_pd.dst_desc(), dnnl_engine);

    auto gelu_op = dnnl::eltwise_forward(gelu_pd);
    sp.AddPrimitive(gelu_op, {{DNNL_ARG_SRC, gelu_src_mem},
                              {DNNL_ARG_DST, dst_mem}});
  }

  if (sp.IsScalar(node.Input(IN_X))) {
    sp.SetMemory(node.Output(OUT_Y), dst_mem, false, true);
  } else {
    sp.SetMemory(node.Output(OUT_Y), dst_mem);
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
