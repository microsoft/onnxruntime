#include "dnnl_relugrad.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlReluGrad::DnnlReluGrad() {}

void DnnlReluGrad::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();

  //get what's available as input
  auto src_mem = sp.GetMemory(node.Input(IN_X).Name());
  auto diff_dst_mem = sp.GetMemory(node.Input(IN_dY).Name());

  //reorder if needed (gpu)
  auto relu_bwd_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), src_mem.get_desc(), eng);
  auto relu_bwd_diff_dst_mem = sp.GetMemoryAndReshape(node.Input(IN_dY), diff_dst_mem.get_desc(), eng);

  //create hints on the fly
  auto hints_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, dnnl::algorithm::eltwise_relu, relu_bwd_src_mem.get_desc(), 0.0, 0.0);
  auto hints_pd = dnnl::eltwise_forward::primitive_desc(hints_d, eng);

  auto relu_bwd_d = dnnl::eltwise_backward::desc(dnnl::algorithm::eltwise_relu, relu_bwd_diff_dst_mem.get_desc(), relu_bwd_src_mem.get_desc(), 0.0, 0.0);

  auto relu_bwd_pd = dnnl::eltwise_backward::primitive_desc(relu_bwd_d, eng, hints_pd);

  auto relu_bwd_diff_src_mem = dnnl::memory(relu_bwd_pd.diff_src_desc(), eng);

  auto relu_bwd = dnnl::eltwise_backward(relu_bwd_pd);

  sp.AddPrimitive(relu_bwd, {{DNNL_ARG_SRC, relu_bwd_src_mem},
                         {DNNL_ARG_DIFF_DST, relu_bwd_diff_dst_mem},
                         {DNNL_ARG_DIFF_SRC, relu_bwd_diff_src_mem}});

  sp.SetMemory(node.Output(OUT_dX), relu_bwd_diff_src_mem);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
