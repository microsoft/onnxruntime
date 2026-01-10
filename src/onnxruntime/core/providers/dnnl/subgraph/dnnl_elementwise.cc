// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_elementwise.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include "dnnl_util.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlElementwise::DnnlElementwise() {}

void DnnlElementwise::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  auto elementwise_src_mem = sp.GetMemory(node.Input(IN_X));
  auto src_md = elementwise_src_mem.get_desc();
  dnnl::algorithm algo = dnnl_util::OrtOperatorToDnnlAlgorithm(node.OpType());
  bool requires_alpha = false;
  float alpha = 0.0;
  switch (algo) {
    case dnnl::algorithm::eltwise_elu: {
      requires_alpha = true;
      alpha = GetAlpha(node, /*default_alpha*/ 1.0f);
      break;
    }
    case dnnl::algorithm::eltwise_relu: {
      // Need to check operator since both Relu and LeakyRelu are covered by algorithm::eltwise_relu
      if (node.OpType() == "LeakyRelu") {
        requires_alpha = true;
        alpha = GetAlpha(node, /*default_alpha*/ 0.01f);
      } else {
        alpha = 0.0;
      }
      break;
    }
    case dnnl::algorithm::eltwise_soft_relu: {
      if (node.OpType() == "Softplus") {
        requires_alpha = true;
        alpha = 1.0f;
      }
      break;
    }
    default:
      alpha = 0.0;
  }

  // Generate a dst_md from the src data
  auto dst_md = dnnl::memory::desc(src_md.get_dims(), src_md.get_data_type(), dnnl::memory::format_tag::any);

  dnnl::eltwise_forward::primitive_desc elementwise_pd;
  if (requires_alpha) {
    elementwise_pd = dnnl::eltwise_forward::primitive_desc(dnnl_engine, dnnl::prop_kind::forward_inference, algo, src_md, dst_md, alpha);
  } else {
    elementwise_pd = dnnl::eltwise_forward::primitive_desc(dnnl_engine, dnnl::prop_kind::forward_inference, algo, src_md, dst_md);
  }

  // If using GPU this will move the memory from the CPU to the GPU.
  elementwise_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), elementwise_pd.src_desc(), dnnl_engine);
  auto elementwise_dst_mem = dnnl::memory(elementwise_pd.dst_desc(), dnnl_engine);

  auto elemenwise_primitive = dnnl::eltwise_forward(elementwise_pd);
  sp.AddPrimitive(elemenwise_primitive, {{DNNL_ARG_SRC, elementwise_src_mem},
                                         {DNNL_ARG_DST, elementwise_dst_mem}});
  if (sp.IsScalar(node.Input(IN_X))) {
    sp.SetMemory(node.Output(OUT_Y), elementwise_dst_mem, false, true);
  } else {
    sp.SetMemory(node.Output(OUT_Y), elementwise_dst_mem);
  }
}

float DnnlElementwise::GetAlpha(DnnlNode& node, float default_alpha) {
  auto attr = node.Attributes().find("alpha");
  if (attr != node.Attributes().end()) {
    return attr->second().f();
  }
  return default_alpha;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
