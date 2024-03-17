// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_softmax.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlSoftmax::DnnlSoftmax() {}

void DnnlSoftmax::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  using namespace dnnl;

  // get the engine, currently only support either single gpu or single cpu device
  auto dnnl_engine = sp.GetEngine();

  auto axis = ReadAxis(node);

  auto softmax_src_mem = sp.GetMemory(node.Input(IN_X));
  auto softmax_src_md = softmax_src_mem.get_desc();

  if (axis < 0) {
    axis = softmax_src_md.get_dims().size() + axis;
  }

  // Use different propagations for training and inference
  dnnl::prop_kind prop_kind;
#if defined(ENABLE_TRAINING)
  prop_kind = dnnl::prop_kind::forward_training;
#else
  prop_kind = dnnl::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

  // Generate the dst_md
  auto dst_md = dnnl::memory::desc(softmax_src_md.get_dims(),
                                   node.Output(OUT_Y).Type(),
                                   dnnl::memory::format_tag::any);

  auto softmax_pd = dnnl::softmax_forward::primitive_desc(dnnl_engine, prop_kind, dnnl::algorithm::softmax_accurate,
                                                          softmax_src_md, dst_md, static_cast<int>(axis));

  // If using GPU this will move the memory from the CPU to the GPU.
  softmax_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), softmax_pd.src_desc(), dnnl_engine);
  auto softmax_dst_mem = dnnl::memory(softmax_pd.dst_desc(), dnnl_engine);

  auto softmax_op = dnnl::softmax_forward(softmax_pd);
  sp.AddPrimitive(softmax_op, {{DNNL_ARG_SRC, softmax_src_mem},
                               {DNNL_ARG_DST, softmax_dst_mem}});
  if (sp.IsScalar(node.Input(IN_X))) {
    sp.SetMemory(node.Output(OUT_Y), softmax_dst_mem, false, true);
  } else {
    sp.SetMemory(node.Output(OUT_Y), softmax_dst_mem);
  }
}

int64_t DnnlSoftmax::ReadAxis(DnnlNode& node) {
  auto attr = node.Attributes().find("axis");
  int64_t axis = -1;  // Default value according to ONNX spec 13 but works with lower opset too
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
    axis = attr->second().i();
  }
  return axis;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
