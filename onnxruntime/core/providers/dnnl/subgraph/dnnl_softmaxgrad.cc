#include "dnnl_softmaxgrad.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlSoftmaxGrad::DnnlSoftmaxGrad() {}

void DnnlSoftmaxGrad::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();

  // get what's available as input
  auto src_mem = sp.GetMemory(node.Input(IN_X));
  auto diff_dst_mem = sp.GetMemory(node.Input(IN_dY));

  // reorder if needed (gpu)
  auto softmax_bwd_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), src_mem.get_desc(), eng);
  auto softmax_bwd_diff_dst_mem = sp.GetMemoryAndReshape(node.Input(IN_dY), diff_dst_mem.get_desc(), eng);

  int axis;
  {
    auto axis64 = ReadAxis(node);
    if (axis64 < 0)
      axis64 = src_mem.get_desc().get_dims().size() + axis64;

    axis = static_cast<int>(axis64);
  }

  auto fws_dst_md = dnnl::memory::desc(diff_dst_mem.get_desc().get_dims(),
                                       diff_dst_mem.get_desc().get_data_type(),
                                       dnnl::memory::format_tag::any);

  // create hints on the fly
  auto hints_pd = dnnl::softmax_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                                                        dnnl::algorithm::softmax_accurate,
                                                        softmax_bwd_src_mem.get_desc(), fws_dst_md, axis);

  auto softmax_bwd_pd = dnnl::softmax_backward::primitive_desc(eng, dnnl::algorithm::softmax_accurate,
                                                               fws_dst_md, softmax_bwd_diff_dst_mem.get_desc(),
                                                               softmax_bwd_src_mem.get_desc(), axis, hints_pd);

  auto softmax_bwd_diff_src_mem = dnnl::memory(softmax_bwd_pd.diff_src_desc(), eng);

  auto softmax_bwd = dnnl::softmax_backward(softmax_bwd_pd);

  sp.AddPrimitive(softmax_bwd, {{DNNL_ARG_DST, softmax_bwd_src_mem},
                                {DNNL_ARG_DIFF_DST, softmax_bwd_diff_dst_mem},
                                {DNNL_ARG_DIFF_SRC, softmax_bwd_diff_src_mem}});

  sp.SetMemory(node.Output(OUT_dX), softmax_bwd_diff_src_mem);
}

int64_t DnnlSoftmaxGrad::ReadAxis(DnnlNode& node) {
  auto attr = node.Attributes().find("axis");
  int64_t axis = -1;  // Default value according to ONNX spec 13 but works with lower opsets too
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
    axis = attr->second().i();
  }
  return axis;
}
}  // namespace ort_dnnl
}  // namespace onnxruntime
