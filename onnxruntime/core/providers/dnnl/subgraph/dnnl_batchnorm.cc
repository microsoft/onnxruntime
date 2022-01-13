// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_batchnorm.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {


DnnlBatchNorm::DnnlBatchNorm() {}


void DnnlBatchNorm::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {

  using namespace dnnl;

  // get the engine, currently only support either single gpu or single cpu device
  auto dnnl_engine = sp.GetEngine();

  auto epsilon = ReadEpsilon(node);
  
  auto batchnorm_src_mem = sp.GetMemory(node.Input(IN_X));
  auto src_md = batchnorm_src_mem.get_desc();

  auto batchnorm_scale_mem = sp.GetMemory(node.Input(IN_SCALE));
  auto scale_md = batchnorm_scale_mem.get_desc();
  auto scale_dims = scale_md.dims();

  auto batchnorm_bias_mem = sp.GetMemory(node.Input(IN_B));
  auto bias_md = batchnorm_bias_mem.get_desc();

  auto batchnorm_mean_mem = sp.GetMemory(node.Input(IN_MEAN));
  auto mean_md = batchnorm_mean_mem.get_desc();

  auto batchnorm_var_mem = sp.GetMemory(node.Input(IN_VAR));
  auto var_md = batchnorm_var_mem.get_desc();


  std::vector<memory::desc> src_mds;
  src_mds.push_back(scale_md);
  src_mds.push_back(bias_md);
  const int axis = 0;

  //To make the inputs compatible with OneDNN, we need to concatenate scale and bias into a single tensor of length 2XC
  //Then, we create the batchnorm pd and feed in the inputs.
  auto concat_pd = dnnl::concat::primitive_desc(axis, src_mds, dnnl_engine);

  //If using GPU this will move the memory from the CPU to the GPU.
  batchnorm_scale_mem = sp.GetMemoryAndReshape(node.Input(IN_SCALE), concat_pd.src_desc(), dnnl_engine);
  batchnorm_bias_mem = sp.GetMemoryAndReshape(node.Input(IN_B), concat_pd.src_desc(), dnnl_engine);
  batchnorm_mean_mem = sp.GetMemoryAndReshape(node.Input(IN_MEAN), mean_md, dnnl_engine);
  batchnorm_var_mem = sp.GetMemoryAndReshape(node.Input(IN_VAR), var_md, dnnl_engine);
  auto batchnorm_scale_shift_mem = dnnl::memory(concat_pd.dst_desc(), dnnl_engine);

  auto batchnorm_desc = dnnl::batch_normalization_forward::desc(dnnl::prop_kind::forward_inference, src_md, epsilon, 
      dnnl::normalization_flags::use_scale_shift | dnnl::normalization_flags::use_global_stats);
  auto batchnorm_pd = dnnl::batch_normalization_forward::primitive_desc(batchnorm_desc, dnnl_engine);

  // If using GPU this will move the memory from the CPU to the GPU.
  batchnorm_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), batchnorm_pd.src_desc(), dnnl_engine);
  auto batchnorm_dst_mem = dnnl::memory(batchnorm_pd.dst_desc(), dnnl_engine);

  auto concat_op = dnnl::concat(concat_pd);
  sp.AddPrimitive(concat_op, {{DNNL_ARG_MULTIPLE_SRC, batchnorm_scale_mem},
                              {DNNL_ARG_MULTIPLE_SRC+1, batchnorm_bias_mem},
                              {DNNL_ARG_DST, batchnorm_scale_shift_mem}});

  auto batchnorm_op = dnnl::batch_normalization_forward(batchnorm_pd);
  sp.AddPrimitive(batchnorm_op, {{DNNL_ARG_SRC, batchnorm_src_mem},
                                 {DNNL_ARG_MEAN, batchnorm_mean_mem},
                                 {DNNL_ARG_VARIANCE, batchnorm_var_mem},
                                 {DNNL_ARG_SCALE_SHIFT, batchnorm_scale_shift_mem},
                                 {DNNL_ARG_DST, batchnorm_dst_mem}});

  sp.SetMemory(node.Output(OUT_Y), batchnorm_dst_mem);
}

float DnnlBatchNorm::ReadEpsilon(DnnlNode& node) {
  auto attr = node.Attributes().find("epsilon");
  float epsilon = 1e-05f;  //Default value according to ONNX spec
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
    epsilon = attr->second().f();
  }
  return epsilon;
}


}  // namespace ort_dnnl
}  // namespace onnxruntime
