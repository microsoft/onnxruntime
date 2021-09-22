#include "dnnl_softmaxgrad.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlSoftmaxGrad::DnnlSoftmaxGrad() {}

void DnnlSoftmaxGrad::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();

  //get what's available as input
  auto src_mem = sp.GetMemory(node.Input(IN_X).Name());
  auto diff_dst_mem = sp.GetMemory(node.Input(IN_dY).Name());

  //reorder if needed (gpu)
  auto softmax_bwd_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), src_mem.get_desc(), eng);
  auto softmax_bwd_diff_dst_mem = sp.GetMemoryAndReshape(node.Input(IN_dY), diff_dst_mem.get_desc(), eng);

  auto axis = ReadAxis(node);

  if (axis < 0)
    axis = src_mem.get_desc().dims().size() + axis;

  //create hints on the fly
  auto hints_d = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training, softmax_bwd_src_mem.get_desc(), (int) axis);
  auto hints_pd = dnnl::softmax_forward::primitive_desc(hints_d, eng);

  auto softmax_bwd_d = dnnl::softmax_backward::desc(softmax_bwd_diff_dst_mem.get_desc(), softmax_bwd_src_mem.get_desc(), (int) axis);

  auto softmax_bwd_pd = dnnl::softmax_backward::primitive_desc(softmax_bwd_d, eng, hints_pd);

  auto softmax_bwd_diff_src_mem = dnnl::memory(softmax_bwd_pd.diff_src_desc(), eng);

  auto softmax_bwd = dnnl::softmax_backward(softmax_bwd_pd);

  sp.AddPrimitive(softmax_bwd, {{DNNL_ARG_DST, softmax_bwd_src_mem},
                                {DNNL_ARG_DIFF_DST, softmax_bwd_diff_dst_mem},
                                {DNNL_ARG_DIFF_SRC, softmax_bwd_diff_src_mem}});

  sp.SetMemory(node.Output(OUT_dX), softmax_bwd_diff_src_mem);
}

int64_t DnnlSoftmaxGrad::ReadAxis(DnnlNode& node) {
  auto attr = node.Attributes().find("axis");
  int64_t axis = -1;  //Default value according to ONNX spec 13 but works with lower opsets too
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
    axis = attr->second().i();
  }
  return axis;
}
}  // namespace ort_dnnl
}  // namespace onnxruntime
