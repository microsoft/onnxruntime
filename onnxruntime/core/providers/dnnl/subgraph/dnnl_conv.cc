// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_conv.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include <cassert>

namespace onnxruntime {
namespace ort_dnnl {

DnnlConv::DnnlConv() {}

void DnnlConv::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  bool has_relu = false;
  if (node.OpType() == "ConvRelu") {
    has_relu = true;
  }

  auto dnnl_engine = sp.GetEngine();

  auto conv_src_mem = sp.GetMemory(node.Input(IN_X));
  auto src_md = conv_src_mem.get_desc();
  auto src_dims = conv_src_mem.get_desc().dims();

  auto conv_weights_mem = sp.GetMemory(node.Input(IN_W));
  auto weight_md = conv_weights_mem.get_desc();
  auto weight_dims_original = conv_weights_mem.get_desc().dims();
  dnnl::memory::dims weight_dims = weight_dims_original;

  bool bias_exists = node.Input(IN_B).Exists();
  dnnl::memory conv_bias_mem;
  dnnl::memory::desc bias_md;
  if (bias_exists) {
    conv_bias_mem = sp.GetMemory(node.Input(IN_B));
    bias_md = conv_bias_mem.get_desc();
  }

  /*
  * Get any inputs required for the dnnl::convolution_forward::desc
  * beyond the dnnl:memory::desc:
  *  -dilations
  *  - strides
  *  - padding_left and padding_right
  */
  auto kernel_shape = GetKernelShape(node);
  ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
  assert(shape != SHAPE_UNKNOWN);

  auto group = GetGroup(node);
  if (group != 1) {
    weight_dims.insert(weight_dims.begin(), group);
    weight_dims[1] = static_cast<int64_t>(weight_dims_original[0] / group);
    dnnl::memory::format_tag format = dnnl::memory::format_tag::any;
    switch (shape) {
      case onnxruntime::ort_dnnl::DnnlConv::SHAPE_UNKNOWN: {
        // use format_tag::any
        break;
      }
      case onnxruntime::ort_dnnl::DnnlConv::SHAPE_1D: {
        format = dnnl::memory::format_tag::goiw;
        break;
      }
      case onnxruntime::ort_dnnl::DnnlConv::SHAPE_2D: {
        format = dnnl::memory::format_tag::goihw;
        break;
      }
      case onnxruntime::ort_dnnl::DnnlConv::SHAPE_3D: {
        format = dnnl::memory::format_tag::goidhw;
        break;
      }
      default:
        // use format_tag::any
        break;
    }
    weight_md = dnnl::memory::desc({weight_dims}, node.Input(IN_W).Type(), format);
  }

  auto strides = GetStrides(node, shape);
  auto dilations = GetDilations(node, shape);
  // Use GetInferedPads here instead of GetPads since this will acount for the `auto_pad` attribute in its return value
  auto padding = GetInferedPads(node, src_dims, dilations, kernel_shape, strides);
  auto padding_left = GetPaddingLeft(padding, shape);
  auto padding_right = GetPaddingRight(padding, shape);

  // Figure out the output shape based on the inputs
  auto dst_mem_dims = InferOutputShape(node, src_dims, weight_dims_original, kernel_shape, strides, dilations, padding);
  dnnl::memory::desc dst_md = dnnl::memory::desc({dst_mem_dims}, node.Input(IN_X).Type(), dnnl::memory::format_tag::any);

#ifdef ENABLE_TRAINING
  auto prop_kind = dnnl::prop_kind::forward_training;
#else
  auto prop_kind = dnnl::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

  dnnl::primitive_attr attr;
  if (has_relu) {
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f;
    const float ops_beta = 0.f;
    dnnl::post_ops ops;
    ops.append_eltwise(ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
    attr.set_post_ops(ops);
  }

  dnnl::convolution_forward::primitive_desc conv_pd;
  if (bias_exists) {
    auto conv_desc = dnnl::convolution_forward::desc(
        prop_kind, dnnl::algorithm::convolution_direct,
        src_md, weight_md, bias_md, dst_md,
        strides, dilations, padding_left, padding_right);
    conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, attr, dnnl_engine);
  } else {
    auto conv_desc = dnnl::convolution_forward::desc(
        prop_kind, dnnl::algorithm::convolution_direct,
        src_md, weight_md, dst_md,
        strides, dilations, padding_left, padding_right);
    conv_pd = dnnl::convolution_forward::primitive_desc(conv_desc, attr, dnnl_engine);
  }

  // If using GPU this will move the memory from the CPU to the GPU.
  conv_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), conv_pd.src_desc(), dnnl_engine);
  conv_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_W), conv_pd.weights_desc(), dnnl_engine);
  if (bias_exists) {
    conv_bias_mem = sp.GetMemoryAndReshape(node.Input(IN_B), conv_pd.bias_desc(), dnnl_engine);
  }
  auto conv_dst_mem = dnnl::memory(conv_pd.dst_desc(), dnnl_engine);

  // Add the convolution layer to the subgraph
  auto conv_op = dnnl::convolution_forward(conv_pd);
  if (bias_exists) {
    sp.AddPrimitive(conv_op, {{DNNL_ARG_SRC, conv_src_mem},
                              {DNNL_ARG_WEIGHTS, conv_weights_mem},
                              {DNNL_ARG_BIAS, conv_bias_mem},
                              {DNNL_ARG_DST, conv_dst_mem}});
  } else {
    sp.AddPrimitive(conv_op, {{DNNL_ARG_SRC, conv_src_mem},
                              {DNNL_ARG_WEIGHTS, conv_weights_mem},
                              {DNNL_ARG_DST, conv_dst_mem}});
  }

  sp.SetMemory(node.Output(OUT_Y), conv_dst_mem);
}

std::vector<int64_t> DnnlConv::GetInferedPads(DnnlNode& node,
                                              const dnnl::memory::dims& src_dims,
                                              const dnnl::memory::dims& dilations,
                                              const std::vector<int64_t>& kernel_shape,
                                              const dnnl::memory::dims& strides) {
  AutoPadType auto_pad = GetAutoPad(node);
  ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
  std::vector<int64_t> pads;
  if (auto_pad == AutoPadType::NOTSET) {
    pads = GetPads(node);
    if (pads.empty()) {
      // 'shape * 2' because we want the pad at the start and end of each dim.
      pads.resize(shape * 2, 0);
    }
    return pads;
  }

  pads.resize(shape * 2, 0);

  int64_t pad_head = 0;
  int64_t pad_tail = 0;
  assert(src_dims.size() == shape + 2);
  for (size_t i = 0; i < shape; ++i) {
    if (ComputePad(src_dims[2 + i], strides[i], kernel_shape[i], (dilations[i] + 1), auto_pad, pad_head, pad_tail)) {
      pads[i] = pad_head;
      pads[shape + i] = pad_tail;
    }
  }
  return pads;
}

dnnl::memory::dims DnnlConv::GetPaddingLeft(const std::vector<int64_t>& onnx_padding, ConvShape shape) {
  assert(onnx_padding.size() == shape * 2);
  dnnl::memory::dims padding_left;
  padding_left.assign(onnx_padding.begin(), onnx_padding.begin() + shape);
  return padding_left;
}

dnnl::memory::dims DnnlConv::GetPaddingRight(const std::vector<int64_t>& onnx_padding, ConvShape shape) {
  assert(onnx_padding.size() == shape * 2);
  dnnl::memory::dims padding_right;
  padding_right.assign(onnx_padding.begin() + shape, onnx_padding.end());
  return padding_right;
}

AutoPadType DnnlConv::GetAutoPad(DnnlNode& node) {
  std::string auto_pad;
  auto attr = node.Attributes().find("auto_pad");
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
    auto_pad = attr->second().s();
  }
  return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

dnnl::memory::dims DnnlConv::GetDilations(DnnlNode& node, ConvShape shape) {
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
int64_t DnnlConv::GetGroup(DnnlNode& node) {
  auto attr = node.Attributes().find("group");
  if (attr != node.Attributes().end()) {
    return attr->second().i();
  }
  return 1;
}

std::vector<int64_t> DnnlConv::GetKernelShape(DnnlNode& node) {
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

std::vector<int64_t> DnnlConv::GetPads(DnnlNode& node) {
  auto attr = node.Attributes().find("pads");
  if (attr != node.Attributes().end()) {
    std::vector<int64_t> pads;
    pads.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      pads.push_back(attr->second().ints(i));
    }
    return pads;
  }
  return {};
}

dnnl::memory::dims DnnlConv::GetStrides(DnnlNode& node, ConvShape shape) {
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

// ComputePad is copy/paste of a the ComputePad found in core/providers/common.h
// With some minor modifications.
// ComputePad is not exposed to the shared library so this copy is used instead.
bool DnnlConv::ComputePad(const int64_t in_dim,
                          const int64_t stride,
                          const int64_t kernel,
                          const int64_t dilation,
                          AutoPadType pad_type,
                          int64_t& pad_head, /* output param */
                          int64_t& pad_tail, /* output param */
                          bool force_symmetric_auto_padding /*= false*/) {
  pad_head = 0;
  pad_tail = 0;
  switch (pad_type) {
    case AutoPadType::NOTSET:
      break;
    case AutoPadType::VALID:
      break;
    case AutoPadType::SAME_UPPER:
      //[[fallthrough]] //fallthrough attribute requires C++17
    case AutoPadType::SAME_LOWER: {
      if (1 != dilation) {
        LOGS_DEFAULT(ERROR) << "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.";
        return false;
      }

      // The ONNX spec says if `auto_pad` attribute is set, pad until the `legacy_target_size`
      // is `ceil (in_dim / stride)`. The following line of code is essentially just that and
      // is retained as is
      int64_t legacy_target_size = (in_dim + stride - 1) / stride;
      int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
      // make sure padding is symmetric
      if (force_symmetric_auto_padding) {
        // Inlining math::roundUpPow2() from util/math.h to avoid bringing in the transitive dependencies.
        pad_needed = (pad_needed + 1) & ~1;
      }

      if (pad_type == AutoPadType::SAME_LOWER) {
        pad_head = (pad_needed + 1) / 2;
      } else {
        pad_head = pad_needed / 2;
      }
      pad_tail = pad_needed - pad_head;
    } break;
    default:
      LOGS_DEFAULT(ERROR) << "ComputePad: pad_type attribute not supported.";
      return false;
  }
  return true;
}

dnnl::memory::dims DnnlConv::InferOutputShape(DnnlNode& node,
                                              const dnnl::memory::dims& src_dims,
                                              const dnnl::memory::dims& weight_dims,
                                              const std::vector<int64_t>& kernel_shape,
                                              const dnnl::memory::dims& strides,
                                              const dnnl::memory::dims& dilations,
                                              const std::vector<int64_t>& pads) {
  auto pad_type = GetAutoPad(node);
  ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
  dnnl::memory::dims output_shape;

  output_shape.push_back(src_dims[0]);
  output_shape.push_back(weight_dims[0]);
  for (size_t dim = 0; dim < shape; ++dim) {
    if (dim >= strides.size() || dim >= kernel_shape.size() ||
        dim >= dilations.size() || dim >= pads.size() ||
        shape + dim >= pads.size()) {
      LOGS_DEFAULT(ERROR) << "Out of bound access to array";
      return {};
    }
    int64_t dkernel = (dilations[dim] + 1) * (kernel_shape[dim] - 1) + 1;
    switch (pad_type) {
      case onnxruntime::AutoPadType::NOTSET: {
        output_shape.push_back(static_cast<int64_t>(static_cast<float>(src_dims[dim + 2] + pads[dim] + pads[dim + shape] - dkernel) / strides[dim] + 1));
      } break;
      case onnxruntime::AutoPadType::VALID: {
        output_shape.push_back((src_dims[dim + 2] - dkernel) / strides[dim] + 1);
      } break;
      case onnxruntime::AutoPadType::SAME_UPPER: {
        if (dilations[dim] != 0) {
          LOGS_DEFAULT(ERROR) << "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.";
          return {};
        }
        int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) / strides[dim];
        int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim] - src_dims[dim + 2];
        output_shape.push_back((src_dims[dim + 2] + pad_needed - dkernel) / strides[dim] + 1);
      } break;
      case onnxruntime::AutoPadType::SAME_LOWER: {
        int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) / strides[dim];
        int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim] - src_dims[dim + 2];
        output_shape.push_back((src_dims[dim + 2] + pad_needed - dkernel) / strides[dim] + 1);
      } break;
      default:
        break;
    }
  }
  return output_shape;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime