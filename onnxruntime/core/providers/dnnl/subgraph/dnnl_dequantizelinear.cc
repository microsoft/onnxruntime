// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_dequantizelinear.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

/*
 y = (x - x_zero_point) * x_scale.
 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
 for per-tensor or per layer quantization, or a 1-D tensor for per-axis quantization.
 'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape.
 In the case of dequantizing int32, there's no zero point (zero point is supposed to be 0).
*/
void DnnlDequantizeLinear::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  // Get engine
  auto dnnl_engine = sp.GetEngine();

  // Validate dims and datatypes
  ValidateDims(sp, node);
  ValidateType(sp, node);

  // Check if scale and zp are scalars
  bool isScalar = sp.IsScalar(node.Input(IN_X_SCALE));
  // Check if zp is needed
  bool isZeroPointUseful = false;
  if (node.Input(IN_X_ZERO_POINT).Exists()) {
    // If zp exists then it's needed
    isZeroPointUseful = true;
    // If it's constant then we can evaluate if zp == 0
    if (node.Input(IN_X_ZERO_POINT).IsConstant()) {
      // if zp == 0 then isZeroPointUseful = false; else isZeroPointUseful = true
      auto mem = sp.GetMemory(node.Input(IN_X_ZERO_POINT));
      isZeroPointUseful = isZeroPointNonZero(&mem);
    }
  }

  // Get the x and scale mem
  auto x_mem = sp.GetMemory(node.Input(IN_X));
  auto x_scale_mem = sp.GetMemory(node.Input(IN_X_SCALE));
  // Move to GPU if available
  x_mem = sp.GetMemoryAndReshape(node.Input(IN_X), x_mem.get_desc(), dnnl_engine);
  x_scale_mem = sp.GetMemoryAndReshape(node.Input(IN_X_SCALE), x_scale_mem.get_desc(), dnnl_engine);
  // Get descs
  auto x_md = x_mem.get_desc();
  auto x_scale_md = x_scale_mem.get_desc();
  auto x_dims = x_md.dims().size();

  // Fix scale dims
  int64_t axis = GetAxis(node, x_dims);
  // Check if axis is negative and fix it
  if (axis < 0) {
    axis += x_dims;
  }
  // Prepare the scale to prevent broacasting errors
  if (isScalar) {
    // For scalar scale
    Padd(&x_scale_md, x_dims, false);
  } else {
    // For N-D scale
    Padd(&x_scale_md, static_cast<uint64_t>(axis) + 1, x_dims);
  }

  // Create dst mem
  auto dst_md = dnnl::memory::desc(x_md.dims(), node.Output(OUT_Y).Type(), dnnl::memory::format_tag::any);
  dnnl::memory dst_mem;

  // If zero point exists and we are NOT dequantizing int32, then substract zp from x and scale
  if (isZeroPointUseful && (x_mem.get_desc().data_type() != dnnl::memory::data_type::s32)) {
    // Get Zero point
    auto x_zp_mem = sp.GetMemory(node.Input(IN_X_ZERO_POINT));
    // Get mds for operands
    auto x_zp_md = x_zp_mem.get_desc();

    // Prepare the zp to prevent broacasting errors
    if (isScalar) {
      // For scalar zp
      Padd(&x_zp_md, x_dims, false);
    } else {
      // For N-D zp
      Padd(&x_zp_md, static_cast<uint64_t>(axis) + 1, x_dims);
    }

    // Create binary desc
    auto binary_d = dnnl::binary::desc(dnnl::algorithm::binary_sub, x_md, x_zp_md, dst_md);
    // Add post op scale
    dnnl::primitive_attr binary_attr;
    {
      dnnl::post_ops binary_ops;
      binary_ops.append_binary(dnnl::algorithm::binary_mul, x_scale_md);
      binary_attr.set_post_ops(binary_ops);
    }
    // Add post op to scale result
    auto binary_pd = dnnl::binary::primitive_desc(binary_d, binary_attr, dnnl_engine);
    // Move to GPU if available
    x_zp_mem = sp.GetMemoryAndReshape(node.Input(IN_X_ZERO_POINT), x_zp_md, dnnl_engine);
    // Create primitive and set dst mem
    dst_mem = dnnl::memory(binary_pd.dst_desc(), dnnl_engine);
    auto binary_prim = dnnl::binary(binary_pd);

    sp.AddPrimitive(binary_prim, {{DNNL_ARG_SRC_0, x_mem},
                                  {DNNL_ARG_SRC_1, x_zp_mem},
                                  {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, x_scale_mem},
                                  {DNNL_ARG_DST, dst_mem}});

    // If zp doesn't exists or we are dequantizing from int32, only need to scale
  } else {
    // Create binary and primitive desc
    auto binary_d = dnnl::binary::desc(dnnl::algorithm::binary_mul, x_md, x_scale_md, dst_md);
    auto binary_pd = dnnl::binary::primitive_desc(binary_d, dnnl_engine);

    // Create primitive
    dst_mem = dnnl::memory(binary_pd.dst_desc(), dnnl_engine);
    auto binary_prim = dnnl::binary(binary_pd);

    sp.AddPrimitive(binary_prim, {{DNNL_ARG_SRC_0, x_mem},
                                  {DNNL_ARG_SRC_1, x_scale_mem},
                                  {DNNL_ARG_DST, dst_mem}});
  }

  // Set the output mem
  if (sp.IsScalar(node.Input(IN_X))) {
    sp.SetMemory(node.Output(OUT_Y), dst_mem, false, true);
  } else {
    sp.SetMemory(node.Output(OUT_Y), dst_mem);
  }
}

bool DnnlDequantizeLinear::isZeroPointNonZero(dnnl::memory* zp_mem) {
  // Because zp will always be int8, uint8 or int32, this cast is always valid
  auto zp_data = static_cast<uint8_t*>(zp_mem->get_data_handle());
  //  Adjust the iteration num
  auto topline = zp_mem->get_desc().dims().size();
  if (zp_mem->get_desc().data_type() == dnnl::memory::data_type::s32) {
    topline *= 4;
  }
  // ZP is either a scalar or a 1-D vector so iterate over all the dimensions
  // and search for a zp != 0
  for (size_t i = 0; i < topline; ++i) {
    if (zp_data[i] != 0) {
      return true;
    }
  }
  // If ZP is full of zeros then it is not needed
  return false;
}

void DnnlDequantizeLinear::Padd(dnnl::memory::desc* target_md, size_t front_pad, size_t back_pad) {
  // Pads an input to broadcast the op correctly
  auto target_dims = target_md->dims();

  // Add front padding
  while (target_dims.size() < front_pad) {
    target_dims.insert(target_dims.begin(), 1);
  }
  // Add back padd
  while (target_dims.size() < back_pad) {
    target_dims.insert(target_dims.end(), 1);
  }

  *target_md = target_md->reshape(target_dims);
}

int64_t DnnlDequantizeLinear::GetAxis(DnnlNode& node, size_t x_dims) {
  // We need to do sign comparisons so we have to cast
  int64_t sig_x_dims = static_cast<uint64_t>(x_dims);
  auto attr = node.Attributes().find("axis");
  // If axis is provided, make sure axis is an integer and
  // has a range of [-r, r]
  if (attr != node.Attributes().end()) {
    int64_t axis2 = attr->second().i();
    if (attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT &&
        (((axis2 <= 0) && (axis2 >= -sig_x_dims)) ||
         ((axis2 >= 0) && (axis2 <= (sig_x_dims - 1))))) {
      return attr->second().i();
    }
  }
  // Return the default value
  return 1;
}

void DnnlDequantizeLinear::ValidateDims(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  // We only need to validate when zp is provided
  if (node.Input(IN_X_ZERO_POINT).Exists()) {
    auto x_scale_dims = sp.GetMemory(node.Input(IN_X_SCALE)).get_desc().dims();
    auto x_zp_dims = sp.GetMemory(node.Input(IN_X_ZERO_POINT)).get_desc().dims();

    if (x_zp_dims != x_scale_dims) {
      ORT_THROW("x_scale and x_zero_point dimensions does not match");
    }
  }
}

void DnnlDequantizeLinear::ValidateType(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  // If zp exists check its dataype
  if (node.Input(IN_X_ZERO_POINT).Exists()) {
    auto x_md = sp.GetMemory(node.Input(IN_X)).get_desc();
    auto x_zp_md = sp.GetMemory(node.Input(IN_X_ZERO_POINT)).get_desc();

    if (x_md.data_type() != x_zp_md.data_type()) {
      ORT_THROW("x and x_zero_point have different datatypes");
    }
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime