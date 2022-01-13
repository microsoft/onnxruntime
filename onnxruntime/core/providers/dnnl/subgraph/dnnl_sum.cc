// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_sum.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlSum::DnnlSum() {}

void DnnlSum::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  std::vector<dnnl::memory> src_mems;
  for (size_t i = IN_DATA_0; i < node.InputCount(); ++i) {
    src_mems.push_back(sp.GetMemory(node.Input(static_cast<int>(IN_DATA_0 + i))));
  }

  std::vector<float> scales;
  std::vector<dnnl::memory::desc> srcs_pd;
  for (size_t i = 0; i < src_mems.size(); ++i) {
    srcs_pd.push_back(src_mems[i].get_desc());
    scales.push_back(1.0f);
  }

  auto dst_dims = srcs_pd[0].dims();
  auto dst_md =  dnnl::memory::desc({dst_dims}, node.Input(IN_DATA_0).Type(), dnnl::memory::format_tag::any);

  auto sum_pd = dnnl::sum::primitive_desc(dst_md, scales, srcs_pd, dnnl_engine);

  for (size_t i = 0; i < src_mems.size(); ++i) {
    src_mems[i] = sp.GetMemoryAndReshape(node.Input(static_cast<int>(IN_DATA_0 + i)), sum_pd.src_desc(), dnnl_engine);
  }
  auto sum_dst_mem = dnnl::memory(sum_pd.dst_desc(), dnnl_engine);
  
  auto sum_op = dnnl::sum(sum_pd);
  
  std::unordered_map<int, dnnl::memory> sum_args;
  sum_args.insert({DNNL_ARG_DST, sum_dst_mem});
  for (int i = 0; i < static_cast<int>(src_mems.size()); ++i) {
    sum_args.insert({DNNL_ARG_MULTIPLE_SRC + i, src_mems[i]});
  }

  sp.AddPrimitive(sum_op, sum_args);

  sp.SetMemory(node.Output(OUT_SUM), sum_dst_mem);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime