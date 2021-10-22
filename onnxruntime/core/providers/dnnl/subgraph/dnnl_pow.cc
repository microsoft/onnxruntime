// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_pow.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlPow::DnnlPow() {}

void DnnlPow::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  auto elementwise_src_mem = sp.GetMemory(node.Input(IN_X));
  auto src_md = elementwise_src_mem.get_desc();

  auto exponent_src_mem = sp.GetMemory(node.Input(IN_Y));

  float beta = 1.0;
  switch (node.Input(IN_Y).Type()) {
    case dnnl::memory::data_type::f32: {
      beta = static_cast<float>(*(float*)exponent_src_mem.get_data_handle());
      break;
    }
    case dnnl::memory::data_type::s32: {
      beta = static_cast<float>(*(int32_t*)exponent_src_mem.get_data_handle());
      break;
    }
    case dnnl::memory::data_type::s8: {
      beta = static_cast<float>(*(int8_t*)exponent_src_mem.get_data_handle());
      break;
    }
    case dnnl::memory::data_type::u8: {
      beta = static_cast<float>(*(uint8_t*)exponent_src_mem.get_data_handle());
      break;
    }
    default:
      ORT_THROW("Pow exponent data type not supported");
  }

  // DNNL eltwise_pow is defined as alpha*x^beta. We don't use alpha so it is hard coded to 1.0
  dnnl::eltwise_forward::desc elementwise_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_pow, src_md, 1.0, beta);
  dnnl::eltwise_forward::primitive_desc elementwise_pd(elementwise_desc, dnnl_engine);

  // If using GPU this will move the memory from the CPU to the GPU.
  elementwise_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), elementwise_pd.src_desc(), dnnl_engine);
  auto elementwise_dst_mem = dnnl::memory(elementwise_pd.dst_desc(), dnnl_engine);

  auto elemenwise_primitive = dnnl::eltwise_forward(elementwise_pd);
  sp.AddPrimitive(elemenwise_primitive, {{DNNL_ARG_SRC, elementwise_src_mem},
                                         {DNNL_ARG_DST, elementwise_dst_mem}});

  sp.SetMemory(node.Output(OUT_Z), elementwise_dst_mem);
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
