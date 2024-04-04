// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_convgrad.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include <cassert>

namespace onnxruntime {
namespace ort_dnnl {

DnnlConvGrad::DnnlConvGrad() {}

/*
ConvGrad: (According to OnnxRuntime discovered using code inspection and Onnx documentation)
  Inputs:
    0) dY - Gradient of output Y
    1) X - Input Tensor
    2) W - Weight Tensor
  Outputs:
    0) dX - Gradient of Input X
    1) dW - Gradient of (W) Weight
    2) dB - Gradient of (B) Bias
                    +-----------------+
    (dY) diff_dst   |                 | (dX optional output ) diff_src
    --------------->+                 +----------------->
    (X) src         |                 | (dW optional output ) diff_weights
    --------------->+    ConvGrad     +----------------->
    (W) weights     |                 | (dB optional output ) diff_bias
    --------------->+                 +----------------->
                    |                 |
                    +-----------------+

  diff_dst = DNNL_ARG_DIFF_DST
  src = DNNL_ARG_SRC
  weights = DNNL_ARG_WEIGHTS
  diff_src = DNNL_ARG_DIFF_SRC
  diff_weights = DNNL_ARG_DIFF_WEIGHTS
  diff_bias = DNNL_ARG_DIFF_BIAS

Attributes (auto_pad, dilations, group, kernel_shap, pads, and strides) should be the same as the forward pass Conv operator

To acheive Everything specified in the OnnxRuntime ConvGrad we must use both:
1) dnnl::convolution_backward_data - used to calculate (dX) diff_src
2) dnnl::convolution_backward_weights - used to calculate (dW) diff_weights and (dB) diff_bias
*/
void DnnlConvGrad::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  auto dy_mem = sp.GetMemory(node.Input(IN_DY));
  auto dy_md = dy_mem.get_desc();
  auto dy_dims = dy_mem.get_desc().get_dims();

  auto x_mem = sp.GetMemory(node.Input(IN_X));
  auto x_md = x_mem.get_desc();
  auto x_dims = x_mem.get_desc().get_dims();

  auto w_mem = sp.GetMemory(node.Input(IN_W));
  auto w_md = w_mem.get_desc();
  auto w_dims_original = w_mem.get_desc().get_dims();
  auto w_dims = w_dims_original;

  bool dx_required = node.Output(OUT_DX).Exists();
  bool dw_required = node.Output(OUT_DW).Exists();
  bool db_required = node.Output(OUT_DB).Exists();

  auto kernel_shape = GetKernelShape(node);
  ConvShape shape = static_cast<ConvShape>(kernel_shape.size());

  auto pads = GetPads(node, shape);
  auto padding_left = GetPaddingLeft(pads, shape);
  auto padding_right = GetPaddingRight(pads, shape);

  auto dilations = GetDilations(node, shape);
  auto strides = GetStrides(node, shape);
  auto group = GetGroup(node);
  if (group != 1) {
    w_dims.insert(w_dims.begin(), group);
    w_dims[1] = static_cast<int64_t>(w_dims_original[0] / group);
    dnnl::memory::format_tag format = dnnl::memory::format_tag::any;
    switch (shape) {
      case onnxruntime::ort_dnnl::DnnlConvGrad::SHAPE_UNKNOWN: {
        // use format_tag::any
        break;
      }
      case onnxruntime::ort_dnnl::DnnlConvGrad::SHAPE_1D: {
        format = dnnl::memory::format_tag::goiw;
        break;
      }
      case onnxruntime::ort_dnnl::DnnlConvGrad::SHAPE_2D: {
        format = dnnl::memory::format_tag::goihw;
        break;
      }
      case onnxruntime::ort_dnnl::DnnlConvGrad::SHAPE_3D: {
        format = dnnl::memory::format_tag::goidhw;
        break;
      }
      default:
        // use format_tag::any
        break;
    }
    w_md = dnnl::memory::desc({w_dims}, node.Input(IN_W).Type(), format);
  }

  // dX memory desc is the same as the input X
  dnnl::memory::desc dx_md = x_md;

  // dW has the same memory desc as the input W
  dnnl::memory::desc dw_md = w_md;
  //
  dnnl::memory::dims db_dims({w_dims[0]});
  dnnl::memory::desc db_md({db_dims}, node.Output(OUT_DB).Type(), dnnl::memory::format_tag::x);

  // fwd Y output memory desc is the same as dY bwd input memory desc
  dnnl::memory::desc fwd_y_md = dy_md;
  // fwd X input memory desc is the same as X bwd input memory desc
  dnnl::memory::desc fwd_x_md = x_md;
  // fwd B input memory desc is the same as dB bwd output memory desc
  dnnl::memory::desc fwd_b_md = db_md;

  // Reproduce the forward convolution pd.
  dnnl::convolution_forward::primitive_desc conv_forward_pd;
  if (db_required) {
    conv_forward_pd = dnnl::convolution_forward::primitive_desc(dnnl_engine, dnnl::prop_kind::forward_training,
                                                                dnnl::algorithm::convolution_direct,
                                                                fwd_x_md, w_md, fwd_b_md, fwd_y_md,
                                                                strides, dilations, padding_left, padding_right);
  } else {
    conv_forward_pd = dnnl::convolution_forward::primitive_desc(dnnl_engine, dnnl::prop_kind::forward_training,
                                                                dnnl::algorithm::convolution_direct,
                                                                fwd_x_md, w_md, fwd_y_md,
                                                                strides, dilations, padding_left, padding_right);
  }

  // Create the convolution backward data primitive desc
  auto conv_backward_data_pd =
      dnnl::convolution_backward_data::primitive_desc(dnnl_engine, dnnl::algorithm::convolution_direct,
                                                      dx_md, w_md, dy_md, strides, dilations, padding_left,
                                                      padding_right, conv_forward_pd);

  // Create the convolution backward weights primitve desc
  dnnl::convolution_backward_weights::primitive_desc conv_backward_weights_pd;
  if (db_required) {
    conv_backward_weights_pd =
        dnnl::convolution_backward_weights::primitive_desc(dnnl_engine, dnnl::algorithm::convolution_direct,
                                                           x_md, dw_md, db_md, dy_md,
                                                           strides, dilations, padding_left,
                                                           padding_right, conv_forward_pd);
  } else {
    conv_backward_weights_pd =
        dnnl::convolution_backward_weights::primitive_desc(dnnl_engine, dnnl::algorithm::convolution_direct,
                                                           x_md, dw_md, dy_md,
                                                           strides, dilations, padding_left, padding_right,
                                                           conv_forward_pd);
  }

  // check if memory needs to be moved to GPU
  dy_mem = sp.GetMemoryAndReshape(node.Input(IN_DY), conv_backward_data_pd.diff_dst_desc(), dnnl_engine);
  x_mem = sp.GetMemoryAndReshape(node.Input(IN_X), conv_backward_weights_pd.src_desc(), dnnl_engine);
  w_mem = sp.GetMemoryAndReshape(node.Input(IN_W), conv_backward_data_pd.weights_desc(), dnnl_engine);

  // Create output memory.
  auto dx_mem_desc = conv_backward_data_pd.diff_src_desc();
  dnnl::memory dx_mem(dx_mem_desc, dnnl_engine);

  auto dw_mem_desc = conv_backward_weights_pd.diff_weights_desc();
  dnnl::memory dw_mem(dw_mem_desc, dnnl_engine);

  auto db_mem_desc = conv_backward_weights_pd.diff_bias_desc();
  dnnl::memory db_mem(db_mem_desc, dnnl_engine);

  if (dx_required) {
    auto conv_backward_data_op = dnnl::convolution_backward_data(conv_backward_data_pd);
    sp.AddPrimitive(conv_backward_data_op, {{DNNL_ARG_DIFF_DST, dy_mem},
                                            {DNNL_ARG_WEIGHTS, w_mem},
                                            {DNNL_ARG_DIFF_SRC, dx_mem}});
  }

  if (dw_required || db_required) {
    auto conv_backward_weights_op = dnnl::convolution_backward_weights(conv_backward_weights_pd);
    sp.AddPrimitive(conv_backward_weights_op, {{DNNL_ARG_DIFF_DST, dy_mem},
                                               {DNNL_ARG_SRC, x_mem},
                                               {DNNL_ARG_DIFF_WEIGHTS, dw_mem},
                                               {DNNL_ARG_DIFF_BIAS, db_mem}});
  }

  if (dx_required) {
    sp.SetMemory(node.Output(OUT_DX), dx_mem);
  }
  if (dw_required) {
    sp.SetMemory(node.Output(OUT_DW), dw_mem);
  }
  if (db_required) {
    sp.SetMemory(node.Output(OUT_DB), db_mem);
  }
}

std::vector<int64_t> DnnlConvGrad::GetKernelShape(DnnlNode& node) {
  auto attr = node.Attributes().find("kernel_shape");
  std::vector<int64_t> kernel_shape;
  if (attr != node.Attributes().end()) {
    kernel_shape.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      kernel_shape.push_back(attr->second().ints(i));
    }
    return kernel_shape;
  }
  // Infer the Kernel shape from the input weights
  auto weight_dims = node.Input(IN_W).Dim();
  kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
  return kernel_shape;
}

std::vector<int64_t> DnnlConvGrad::GetPads(DnnlNode& node, ConvShape shape) {
  std::vector<int64_t> pads;
  auto attr = node.Attributes().find("pads");
  if (attr != node.Attributes().end()) {
    pads.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      pads.push_back(attr->second().ints(i));
    }

    return dnnl::memory::dims(pads.begin(), pads.end());
  }

  pads.resize(shape * 2, 0);
  return dnnl::memory::dims(pads.begin(), pads.end());
  ;
}

dnnl::memory::dims DnnlConvGrad::GetPaddingLeft(const std::vector<int64_t>& onnx_padding, ConvShape shape) {
  assert(onnx_padding.size() == shape * 2);
  dnnl::memory::dims padding_left;
  padding_left.assign(onnx_padding.begin(), onnx_padding.begin() + shape);
  return padding_left;
}

dnnl::memory::dims DnnlConvGrad::GetPaddingRight(const std::vector<int64_t>& onnx_padding, ConvShape shape) {
  assert(onnx_padding.size() == shape * 2);
  dnnl::memory::dims padding_right;
  padding_right.assign(onnx_padding.begin() + shape, onnx_padding.end());
  return padding_right;
}

dnnl::memory::dims DnnlConvGrad::GetDilations(DnnlNode& node, ConvShape shape) {
  auto attr = node.Attributes().find("dilations");
  std::vector<int64_t> dilations;
  if (attr != node.Attributes().end()) {
    dilations.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      // OneDNN dilations are always one less than Onnx dilations
      dilations.push_back(attr->second().ints(i) - 1);
    }
  } else {
    dilations.resize(shape, 0);
  }
  return dnnl::memory::dims(dilations.begin(), dilations.end());
}

dnnl::memory::dims DnnlConvGrad::GetStrides(DnnlNode& node, ConvShape shape) {
  auto attr = node.Attributes().find("strides");
  std::vector<int64_t> strides;
  if (attr != node.Attributes().end()) {
    strides.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      strides.push_back(attr->second().ints(i));
    }
  } else {
    strides.resize(shape, 1);
  }
  return dnnl::memory::dims(strides.begin(), strides.end());
}

int64_t DnnlConvGrad::GetGroup(DnnlNode& node) {
  auto attr = node.Attributes().find("group");
  if (attr != node.Attributes().end()) {
    return attr->second().i();
  }
  return 1;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime