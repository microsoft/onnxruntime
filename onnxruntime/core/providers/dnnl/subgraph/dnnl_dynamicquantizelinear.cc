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
  // Get engine
  auto eng = sp.GetEngine();

  // Get src mem
  auto x_mem = sp.GetMemory(node.Input(IN_X));
  auto x_md = x_mem.get_desc();
  auto x_size = x_md.dims().size();
  auto x_format = sp.GetDnnlFormat(x_size);
  x_mem = sp.GetMemoryAndReshape(node.Input(IN_X), x_md, eng);

  // Dims for one dimensional tensor
  dnnl::memory::dims one_dim(x_size, 1);

  // Y_SCALE COMPUTATION
  // Create descriptor for reduction max and min
  auto y_scale_md = dnnl::memory::desc(one_dim, x_md.data_type(), x_format);
  auto max_reduction_d = dnnl::reduction::desc(dnnl::algorithm::reduction_max, x_md, y_scale_md, 0.f, 0.f);
  auto min_reduction_d = dnnl::reduction::desc(dnnl::algorithm::reduction_min, x_md, y_scale_md, 0.f, 0.f);

  // Fill memory with 0's, needed for min and max binary
  auto zero_mem = dnnl::memory(y_scale_md, eng);
  WriteZeroToMem(zero_mem);

  // Generate post ops to calc y_scale
  dnnl::primitive_attr max_reduction_attr;
  {
    // y_scale = ((x_max - x_min) / (255 - 0)).astype(np.float32) # uint8->[0, 255]
    dnnl::post_ops calc_y_scale;
    // x_max = max(0, reduce_max(x))
    calc_y_scale.append_binary(dnnl::algorithm::binary_max, zero_mem.get_desc());
    // y_scale = x_max - x_min
    calc_y_scale.append_binary(dnnl::algorithm::binary_sub, y_scale_md);
    // y_scale =/ 255
    calc_y_scale.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f / 255.0f, 0.0f);
    max_reduction_attr.set_post_ops(calc_y_scale);
  }

  // x_min = min(0, reduce_min(x))
  dnnl::primitive_attr min_reduction_attr;
  {
    dnnl::post_ops calc_min;
    calc_min.append_binary(dnnl::algorithm::binary_min, zero_mem.get_desc());
    min_reduction_attr.set_post_ops(calc_min);
  }

  // Create reduction primitive
  auto max_reduction_prim = dnnl::reduction(dnnl::reduction::primitive_desc(max_reduction_d, max_reduction_attr, eng));
  auto min_reduction_prim = dnnl::reduction(dnnl::reduction::primitive_desc(min_reduction_d, min_reduction_attr, eng));

  // Create y_scale and min memory
  auto y_scale_mem = dnnl::memory(y_scale_md, eng);
  auto min_reduction_mem = dnnl::memory(y_scale_md, eng);

  // Compute min first since max_reduction needs min as input
  sp.AddPrimitive(min_reduction_prim, {{DNNL_ARG_SRC, x_mem},
                                       {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, zero_mem},
                                       {DNNL_ARG_DST, min_reduction_mem}});

  // Compute y_scale in fp32
  sp.AddPrimitive(max_reduction_prim, {{DNNL_ARG_SRC, x_mem},
                                       {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, zero_mem},
                                       {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, min_reduction_mem},
                                       {DNNL_ARG_DST, y_scale_mem}});


  // Y_ZERO_POINT COMPUTATION
  // Create memory and primitive descriptors
  auto y_zp_md = dnnl::memory::desc(one_dim, dnnl::memory::data_type::u8, x_format);
  auto zp_prim_d = dnnl::binary::desc(dnnl::algorithm::binary_div, y_scale_md, y_scale_md, y_zp_md);

  // Add round and clip post ops
  dnnl::primitive_attr zp_prim_attr;
  {
    zp_prim_attr.set_scales(DNNL_ARG_SRC_0, 0, {-1.0f});
    dnnl::post_ops div_saturate_round;
    div_saturate_round.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
    zp_prim_attr.set_post_ops(div_saturate_round);
  }

  // Create primitives
  auto zp_prim_pd = dnnl::binary::primitive_desc(zp_prim_d, zp_prim_attr, eng);
  auto zp_prim = dnnl::binary(zp_prim_pd);

  // Create zp memory dst
  auto y_zp_mem = dnnl::memory(zp_prim_pd.dst_desc(), eng);

  // Calc zp
  sp.AddPrimitive(zp_prim,{{DNNL_ARG_SRC_0, min_reduction_mem},
                           {DNNL_ARG_SRC_1, y_scale_mem},
                           {DNNL_ARG_DST, y_zp_mem}});

  // Y COMPUTATION
  // Create y md and binary desc
  auto y_md = dnnl::memory::desc(x_md.dims(), dnnl::memory::data_type::u8, x_format);
  auto y_bin_d = dnnl::binary::desc(dnnl::algorithm::binary_div, x_mem.get_desc(), y_scale_mem.get_desc(), y_md);
  // Add post ops
  dnnl::primitive_attr y_bin_attr;
  {
    dnnl::post_ops round_add;
    round_add.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
    round_add.append_binary(dnnl::algorithm::binary_add, y_zp_mem.get_desc());
    y_bin_attr.set_post_ops(round_add);
  }
  // Create binary primitive with post ops
  auto y_pd = dnnl::binary::primitive_desc(y_bin_d, y_bin_attr, eng);
  auto y_prim = dnnl::binary(y_pd);
  // Create y_dst mem
  auto y_mem = dnnl::memory(y_pd.dst_desc(), eng);
  // Compute y
  sp.AddPrimitive(y_prim, {{DNNL_ARG_SRC_0, x_mem},
                           {DNNL_ARG_SRC_1, y_scale_mem},
                           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, y_zp_mem},
                           {DNNL_ARG_DST, y_mem}});

  // Set outputs
  sp.SetMemory(node.Output(OUT_Y), y_mem);
  sp.SetMemory(node.Output(OUT_Y_SCALE), y_scale_mem, false, true);
  sp.SetMemory(node.Output(OUT_Y_ZP), y_zp_mem, false, true);
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
