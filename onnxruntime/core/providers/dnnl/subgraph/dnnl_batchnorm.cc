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
  auto scale_dims = scale_md.get_dims();

  auto batchnorm_bias_mem = sp.GetMemory(node.Input(IN_B));
  auto bias_md = batchnorm_bias_mem.get_desc();

  auto batchnorm_mean_mem = sp.GetMemory(node.Input(IN_MEAN));
  auto mean_md = batchnorm_mean_mem.get_desc();

  auto batchnorm_var_mem = sp.GetMemory(node.Input(IN_VAR));
  auto var_md = batchnorm_var_mem.get_desc();

  // Primitive desc info
  auto dst_md = dnnl::memory::desc(src_md.get_dims(), src_md.get_data_type(), dnnl::memory::format_tag::any);
  auto flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift | dnnl::normalization_flags::use_global_stats;

  auto batchnorm_pd =
      dnnl::batch_normalization_forward::primitive_desc(dnnl_engine, dnnl::prop_kind::forward_inference,
                                                        src_md, dst_md, epsilon, flags);

  // If using GPU this will move the memory from the CPU to the GPU.
  batchnorm_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), batchnorm_pd.src_desc(), dnnl_engine);
  batchnorm_scale_mem = sp.GetMemoryAndReshape(node.Input(IN_SCALE), scale_md, dnnl_engine);
  batchnorm_bias_mem = sp.GetMemoryAndReshape(node.Input(IN_B), bias_md, dnnl_engine);
  batchnorm_mean_mem = sp.GetMemoryAndReshape(node.Input(IN_MEAN), mean_md, dnnl_engine);
  batchnorm_var_mem = sp.GetMemoryAndReshape(node.Input(IN_VAR), var_md, dnnl_engine);
  auto batchnorm_dst_mem = dnnl::memory(batchnorm_pd.dst_desc(), dnnl_engine);

  auto batchnorm_op = dnnl::batch_normalization_forward(batchnorm_pd);
  sp.AddPrimitive(batchnorm_op, {{DNNL_ARG_SRC, batchnorm_src_mem},
                                 {DNNL_ARG_MEAN, batchnorm_mean_mem},
                                 {DNNL_ARG_VARIANCE, batchnorm_var_mem},
                                 {DNNL_ARG_SCALE, batchnorm_scale_mem},
                                 {DNNL_ARG_SHIFT, batchnorm_bias_mem},
                                 {DNNL_ARG_DST, batchnorm_dst_mem}});

  sp.SetMemory(node.Output(OUT_Y), batchnorm_dst_mem);
}

float DnnlBatchNorm::ReadEpsilon(DnnlNode& node) {
  auto attr = node.Attributes().find("epsilon");
  float epsilon = 1e-05f;  // Default value according to ONNX spec
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
    epsilon = attr->second().f();
  }
  return epsilon;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
