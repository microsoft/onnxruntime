// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_dynamicquantizelinear.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {


/*
x_min = np.minimum(0, np.min(X))
x_max = np.maximum(0, np.max(X))
Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)
*/
void DnnlDynamicQuantizeLinear::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto eng = sp.GetEngine();
  auto x_memory = sp.GetMemory(node.Input(IN_X).Name());
  x_memory = sp.GetMemoryAndReshape(node.Input(IN_X), x_memory.get_desc(), eng);
  auto x_memory_desc = x_memory.get_desc();
  auto x_memory_dims = x_memory_desc.dims();
  auto x_memory_dt = x_memory_desc.data_type();

  //dims of all ones
  dnnl::memory::dims min_max_dst_dims(x_memory_dims.size(), 1);

  auto min_max_dst_mem_desc = dnnl::memory::desc(min_max_dst_dims, x_memory_dt, sp.GetDnnlFormat(x_memory_dims.size()));

  //max_reduction responsible for producing scale
  auto max_reduction_d = dnnl::reduction::desc(
      dnnl::algorithm::reduction_max, x_memory_desc, min_max_dst_mem_desc, 0.f, 0.f);
  auto min_reduction_d = dnnl::reduction::desc(
      dnnl::algorithm::reduction_min, x_memory_desc, min_max_dst_mem_desc, 0.f, 0.f);

  //prepare a zero memory, used for adding 0 to data range for min max operation
  auto zero_mem = dnnl::memory(min_max_dst_mem_desc, eng);
  WriteZeroToMem(zero_mem);

  //max(x) with 0 added to range -> sub min(x) -> div 255
  dnnl::primitive_attr max_reduction_attr;
  {
    dnnl::post_ops sub_min_div_255;
    //max(0,reduce_max(x))
    sub_min_div_255.append_binary(dnnl::algorithm::binary_max, zero_mem.get_desc());
    //max - min
    sub_min_div_255.append_binary(dnnl::algorithm::binary_sub, min_max_dst_mem_desc);
    // /255
    sub_min_div_255.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f / 255.0f, 0.0f);
    max_reduction_attr.set_post_ops(sub_min_div_255);
  }

  //add 0 to reduce min range
  dnnl::primitive_attr min_reduction_attr;
  {
    dnnl::post_ops add_0_to_range;
    add_0_to_range.append_binary(dnnl::algorithm::binary_min, zero_mem.get_desc());
    min_reduction_attr.set_post_ops(add_0_to_range);
  }

  auto max_reduction_pd = dnnl::reduction::primitive_desc(max_reduction_d, max_reduction_attr, eng);
  auto min_reduction_pd = dnnl::reduction::primitive_desc(min_reduction_d, min_reduction_attr, eng);

  auto max_reduction_prim = dnnl::reduction(max_reduction_pd);
  auto min_reduction_prim = dnnl::reduction(min_reduction_pd);

  auto y_scale_mem = dnnl::memory(min_max_dst_mem_desc, eng);
  auto min_reduction_dst_mem = dnnl::memory(min_max_dst_mem_desc, eng);

  std::unordered_map<int, dnnl::memory> min_reduction_args = {{DNNL_ARG_SRC, x_memory}, {DNNL_ARG_DST, min_reduction_dst_mem}};
  min_reduction_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = zero_mem;

  std::unordered_map<int, dnnl::memory> max_reduction_args = {{DNNL_ARG_SRC, x_memory}, {DNNL_ARG_DST, y_scale_mem}};
  max_reduction_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1] = zero_mem;
  max_reduction_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1] = min_reduction_dst_mem;

  //compute min first since max_reduction needs min dst as post op arg
  // compute x min
  sp.AddPrimitive(min_reduction_prim, min_reduction_args);
  // compute y scale f32
  sp.AddPrimitive(max_reduction_prim, max_reduction_args);


  //prepare y zero point kernel
  auto y_zero_point_d = dnnl::binary::desc(dnnl::algorithm::binary_div, min_reduction_dst_mem.get_desc(), y_scale_mem.get_desc(), min_reduction_dst_mem.get_desc());

  dnnl::primitive_attr y_zero_point_attr;
  {
    y_zero_point_attr.set_scales(DNNL_ARG_SRC_0, 0, {-1.0f});
    dnnl::post_ops div_saturate_round;
    div_saturate_round.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
    //clip might not be necessary as reorder cast will saturate on lower precision
    //might still need it as compute y needs saturated zero point already
    div_saturate_round.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip_v2, 0.0f, 255.0f);
    y_zero_point_attr.set_post_ops(div_saturate_round);
  }
  auto y_zero_point_pd = dnnl::binary::primitive_desc(y_zero_point_d, y_zero_point_attr, eng);
  auto y_zero_point_prim = dnnl::binary(y_zero_point_pd);

  auto y_zero_point_dst_mem = dnnl::memory(y_zero_point_pd.dst_desc(), eng);
  std::unordered_map<int, dnnl::memory> y_zero_point_args = {{DNNL_ARG_SRC_0, min_reduction_dst_mem}, {DNNL_ARG_SRC_1, y_scale_mem}, {DNNL_ARG_DST, y_zero_point_dst_mem}};

  //y zero point f32
  //np.clip(round((0 - x_min) / Y_Scale), 0, 255)
  sp.AddPrimitive(y_zero_point_prim, y_zero_point_args);


  //prepare y kernel
  //x/y -> round() -> + y_zp -> clip 0,255
  auto y_d = dnnl::binary::desc(dnnl::algorithm::binary_div, x_memory.get_desc(), y_scale_mem.get_desc(), x_memory.get_desc());
  dnnl::primitive_attr y_attr;
  {
    dnnl::post_ops round_zp_saturate;
    round_zp_saturate.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
    round_zp_saturate.append_binary(dnnl::algorithm::binary_add, y_zero_point_dst_mem.get_desc());
    //clip might not be necessary as reorder cast will saturate on lower precision
    round_zp_saturate.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip_v2, 0.0f, 255.0f);
    y_attr.set_post_ops(round_zp_saturate);
  }
  auto y_pd = dnnl::binary::primitive_desc(y_d, y_attr, eng);
  auto y_prim = dnnl::binary(y_pd);

  auto y_mem = dnnl::memory(y_pd.dst_desc(), eng);
  std::unordered_map<int, dnnl::memory> y_args = {{DNNL_ARG_SRC_0, x_memory}, {DNNL_ARG_SRC_1, y_scale_mem}, {DNNL_ARG_DST, y_mem}};
  y_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1] = y_zero_point_dst_mem;
  
  // x/y -> round() -> + y_zp -> clip 0,255
  // quantized output tensor f32
  sp.AddPrimitive(y_prim, y_args);


  //set output y scale
  sp.SetMemory(node.Output(OUT_Y_SCALE), y_scale_mem, false, true);

  //data type change for y_zp and set memory
  //data type change is needed for onnxruntime spec
  //zp for onednn is currently in s32, any downstream node might need to convert from u8 to s32 
  auto y_zero_point_dst_md_uint8 = ChangeMemoryDescDataType(y_zero_point_dst_mem.get_desc(), dnnl::memory::data_type::u8);
  auto y_zero_point_dst_mem_uint8 = dnnl::memory(y_zero_point_dst_md_uint8, eng);
  sp.AddPrimitive(dnnl::reorder(y_zero_point_dst_mem, y_zero_point_dst_mem_uint8), {{DNNL_ARG_FROM, y_zero_point_dst_mem}, {DNNL_ARG_TO, y_zero_point_dst_mem_uint8}});
  sp.SetMemory(node.Output(OUT_Y_ZP), y_zero_point_dst_mem_uint8, false, true);

  //data type change for y and set memory
  auto y_md_uint8 = ChangeMemoryDescDataType(y_mem.get_desc(), dnnl::memory::data_type::u8);
  auto y_mem_uint8 = dnnl::memory(y_md_uint8, eng);
  sp.AddPrimitive(dnnl::reorder(y_mem, y_mem_uint8), {{DNNL_ARG_FROM, y_mem}, {DNNL_ARG_TO, y_mem_uint8}});
  sp.SetMemory(node.Output(OUT_Y), y_mem_uint8);

}

//change md to targeted data type of cast op dst
dnnl::memory::desc DnnlDynamicQuantizeLinear::ChangeMemoryDescDataType(dnnl::memory::desc md, dnnl::memory::data_type dt) {
  auto dims = md.dims();
  auto strides = md.data.format_desc.blocking.strides;
  dnnl::memory::dims strides_vec;
  for (size_t i = 0; i < dims.size(); i++) {
    strides_vec.push_back(strides[i]);
  }
  auto result = dnnl::memory::desc(dims, dt, strides_vec);
  return result;
}

//write zero to memory
void DnnlDynamicQuantizeLinear::WriteZeroToMem(dnnl::memory& mem) {
  bool on_gpu = false;
  if (mem.get_engine().get_kind() == dnnl::engine::kind::gpu) {
    on_gpu = true;
  }
  if (!on_gpu) {
    auto dst = mem.get_data_handle();
    size_t size = mem.get_desc().get_size();
    memset(dst, 0, size);
  } else {
    //create a memory on cpu and do a reorder to gpu
    auto cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    auto cpu_memory = dnnl::memory(mem.get_desc(),cpu_engine);
    memset(cpu_memory.get_data_handle(),0,cpu_memory.get_desc().get_size());
    dnnl::stream s{mem.get_engine()};
    //mem now contains all zero
    dnnl::reorder(cpu_memory, mem).execute(s, cpu_memory, mem);
    //wait for reorder to complete
    s.wait();
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
