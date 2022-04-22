// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_elementwise.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlElementwise::DnnlElementwise() {}

void DnnlElementwise::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  auto elementwise_src_mem = sp.GetMemory(node.Input(IN_X));
  auto src_md = elementwise_src_mem.get_desc();
  dnnl::algorithm algo;
  bool requires_alpha = false;
  float alpha = 0.0;
  if (node.OpType() == "Abs") {
    algo = dnnl::algorithm::eltwise_abs;
  } else if (node.OpType() == "Elu") {
    requires_alpha = true;
    alpha = GetAlpha(node, /*default_alpha*/1.0f);
    algo = dnnl::algorithm::eltwise_elu; 
  } else if (node.OpType() == "Exp") {
    algo = dnnl::algorithm::eltwise_exp;
  } else if (node.OpType() == "LeakyRelu") {
    requires_alpha = true;
    alpha = GetAlpha(node, /*default_alpha*/ 0.01f);
    algo = dnnl::algorithm::eltwise_relu;
  } else if (node.OpType() == "Log") {
    algo = dnnl::algorithm::eltwise_log;
  } else if (node.OpType() == "Relu") {
    algo = dnnl::algorithm::eltwise_relu;
  } else if (node.OpType() == "Round") {
    algo = dnnl::algorithm::eltwise_round;
  } else if (node.OpType() == "Sigmoid") {
    // in OneDNN eltwise_logistic is defined as 1/(1 + exp(-x)) which matches the definition of "Sigmoid" in Onnx
    algo = dnnl::algorithm::eltwise_logistic;
  } else if (node.OpType() == "Softplus") {
    // in OneDNN eltwise_soft_relu is defined as ln(1 + exp(x)) which matches the definition of "Softplus" in Onnx
    algo = dnnl::algorithm::eltwise_soft_relu;
  } else if (node.OpType() == "Sqrt") {
    algo = dnnl::algorithm::eltwise_sqrt;
  } else if (node.OpType() == "Tanh") {
    algo = dnnl::algorithm::eltwise_tanh;
  } else {
    ORT_THROW("op type not supported");
  }
  dnnl::eltwise_forward::primitive_desc elementwise_pd;
  if (requires_alpha) {
    auto elementwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo, src_md, alpha);
    elementwise_pd = dnnl::eltwise_forward::primitive_desc(elementwise_desc, dnnl_engine);
  } else {
    auto elementwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algo, src_md);
    elementwise_pd = dnnl::eltwise_forward::primitive_desc(elementwise_desc, dnnl_engine);
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
