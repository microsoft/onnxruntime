// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_poolgrad.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"
#include <cassert>

namespace onnxruntime {
namespace ort_dnnl {

DnnlPoolGrad::DnnlPoolGrad() {}

/*
MaxPoolGrad: (According to OnnxRuntime discovered using code inspection and Onnx documentation)
  Inputs:
    0) dY - Gradient of output Y
    1) indices - indices
  Outputs:
    0) dX - Gradient of Input

                        +-----------------+
    (dY) diff_dst       |                 |
    ------------------->+                 | (dX ) diff_src
    (indices) workspace | MaxPoolGrad     +----------------->
    ------------------->+                 |
                        |                 |
                        +-----------------+

  diff_dst  = DNNL_ARG_DIFF_DST
  workspace = DNNL_ARG_WORKSPACE

  diff_src  = DNNL_ARG_DIFF_SRC

Attributes (auto_pad, ceil_mode, dilations, kernel_shap, pads, storage_order, and strides) should be the same as the forward pass Pool operator

The indices must come from the forward pool operator the indices input from OnnxRuntime will be ignored. For that reason the
forward and backward operators must run using dnnl endpoint.

AveragePoolGrad:
  Inputs:
    0) dY - Gradient of output Y
  Outputs:
    0) dX - Gradient of Input

                        +-----------------+
    (dY) diff_dst       |                 | (dX ) diff_src
    ------------------->+ AveragePoolGrad +----------------->
                        |                 |
                        +-----------------+

  diff_dst  = DNNL_ARG_DIFF_DST
  diff_src  = DNNL_ARG_DIFF_SRC

Attributes (auto_pad, ceil_mode, count_include_pad, kernel_shap, pads, and strides) should be the same as the forward pass Pool operator
*/
void DnnlPoolGrad::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();

  auto dy_mem = sp.GetMemory(node.Input(IN_DY));
  auto dy_md = dy_mem.get_desc();
  auto dy_dims = dy_mem.get_desc().get_dims();

  dnnl::memory indices_mem;
  dnnl::memory::desc indices_md;
  dnnl::memory::dims indices_dims;
  bool maxpoolgrad_optype = (node.OpType() == "MaxPoolGrad");

  if (maxpoolgrad_optype) {
    indices_mem = sp.GetMemory(node.Input(IN_INDICES));
    indices_md = indices_mem.get_desc();
    indices_dims = indices_mem.get_desc().get_dims();
  }

  auto dx_dims = node.Output(OUT_DX).Dim();
  dnnl::memory::desc dx_md(dx_dims, node.Input(IN_DY).Type(), dnnl::memory::format_tag::any);
  dnnl::memory::desc fwd_dx_md(dx_dims, node.Input(IN_DY).Type(), sp.GetDnnlFormat(dx_dims.size()));

  // Read the attributes
  auto kernel_shape = GetKernelShape(node);
  PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
  auto strides = GetStrides(node, shape);
  auto padding = GetPadding(node, shape);
  auto padding_left = GetPaddingLeft(padding);
  auto padding_right = GetPaddingRight(padding);

  dnnl::algorithm algo = dnnl::algorithm::pooling_max;
  if (node.OpType() == "AveragePoolGrad" /*|| node.OpType() == "GlobalAveragePoolGrad"*/) {
    algo = dnnl::algorithm::pooling_avg_exclude_padding;
    if (GetCountIncludePadding(node) != 0) {
      algo = dnnl::algorithm::pooling_avg_include_padding;
    }
  }

  // Default dilatation to 0
  auto dilatation = dnnl::memory::dims(kernel_shape.size(), 0);

  dnnl::pooling_forward::primitive_desc pool_forward_pd(dnnl_engine, dnnl::prop_kind::forward, algo, fwd_dx_md, dy_md,
                                                        strides, kernel_shape, dilatation, padding_left, padding_right);

  dnnl::pooling_backward::primitive_desc pool_backward_pd(dnnl_engine, algo, dx_md, dy_md, strides, kernel_shape,
                                                          dilatation, padding_left, padding_right, pool_forward_pd);

  dnnl::pooling_backward pool_backward_op(pool_backward_pd);

  dy_mem = sp.GetMemoryAndReshape(node.Input(IN_DY), pool_backward_pd.diff_dst_desc(), dnnl_engine);
  if (maxpoolgrad_optype) {
    indices_mem = sp.GetMemoryAndReshape(node.Input(IN_INDICES), pool_backward_pd.workspace_desc(), dnnl_engine);
  }

  dnnl::memory dx_mem(pool_backward_pd.diff_src_desc(), dnnl_engine);
  if (maxpoolgrad_optype) {
    sp.AddPrimitive(pool_backward_op, {{DNNL_ARG_DIFF_DST, dy_mem},
                                       {DNNL_ARG_DIFF_SRC, dx_mem},
                                       {DNNL_ARG_WORKSPACE, indices_mem}});
  } else {
    sp.AddPrimitive(pool_backward_op, {{DNNL_ARG_DIFF_DST, dy_mem},
                                       {DNNL_ARG_DIFF_SRC, dx_mem}});
  }
  sp.SetMemory(node.Output(OUT_DX), dx_mem);
}

AutoPadType DnnlPoolGrad::GetAutoPad(DnnlNode& node) {
  std::string auto_pad;
  auto attr = node.Attributes().find("auto_pad");
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
    auto_pad = attr->second().s();
  }
  return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

int64_t DnnlPoolGrad::GetCeilMode(DnnlNode& node) {
  auto attr = node.Attributes().find("ceil_mode");
  if (attr != node.Attributes().end()) {
    return attr->second().i();
  }
  return false;
}

int64_t DnnlPoolGrad::GetCountIncludePadding(DnnlNode& node) {
  auto attr = node.Attributes().find("count_include_pad");
  if (attr != node.Attributes().end()) {
    return attr->second().i();
  }
  return 0;
}

dnnl::memory::dims DnnlPoolGrad::GetDilations(DnnlNode& node, PoolShape shape) {
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

dnnl::memory::dims DnnlPoolGrad::GetKernelShape(DnnlNode& node) {
  auto attr = node.Attributes().find("kernel_shape");
  std::vector<int64_t> kernel_shape;
  if (attr != node.Attributes().end()) {
    kernel_shape.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      kernel_shape.push_back(attr->second().ints(i));
    }
    return kernel_shape;
  }
  return {};
}

std::vector<int64_t> DnnlPoolGrad::InferPadding(DnnlNode& node, const dnnl::memory::dims& src_dims, const dnnl::memory::dims& kernel_shape, const dnnl::memory::dims& strides) {
  auto auto_pad = GetAutoPad(node);
  PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
  std::vector<int64_t> padding;
  switch (auto_pad) {
    case onnxruntime::AutoPadType::NOTSET: {
      padding = GetPadding(node, shape);
      return padding;
      break;
    }
    case onnxruntime::AutoPadType::VALID: {
      padding.resize(shape * 2, 0);
      return padding;
      break;
    }
    case onnxruntime::AutoPadType::SAME_UPPER: {
      padding.resize(shape * 2, 0);
      for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
        int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) / strides[dim];
        int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim] - src_dims[dim + 2];
        int64_t pad_head = pad_needed / 2;
        int64_t pad_tail = pad_needed - pad_head;
        padding[dim] = pad_head;
        padding[dim + shape] = pad_tail;
      }
      return padding;
      break;
    }
    case onnxruntime::AutoPadType::SAME_LOWER: {
      padding.resize(shape * 2, 0);
      for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
        int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) / strides[dim];
        int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim] - src_dims[dim + 2];
        int64_t pad_head = (pad_needed + 1) / 2;
        int64_t pad_tail = pad_needed - pad_head;
        padding[dim] = pad_head;
        padding[dim + shape] = pad_tail;
      }
      return padding;
      break;
    }
    default:
      ORT_THROW("Unsupported AutoPad Type.");
      break;
  }
}

std::vector<int64_t> DnnlPoolGrad::GetPadding(DnnlNode& node, PoolShape shape) {
  auto attr = node.Attributes().find("pads");
  std::vector<int64_t> pads;
  if (attr != node.Attributes().end() && !IsGlobalPooling(node)) {
    pads.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      pads.push_back(attr->second().ints(i));
    }
  }
  if (pads.empty()) {
    // 'shape * 2' because we want the pad at the start and end of each dim.
    pads.resize(shape * 2, 0);
  }
  return pads;
}

dnnl::memory::dims DnnlPoolGrad::GetPaddingLeft(const std::vector<int64_t> padding) {
  return dnnl::memory::dims(padding.begin(), padding.begin() + (padding.size() / 2));
}

dnnl::memory::dims DnnlPoolGrad::GetPaddingRight(const std::vector<int64_t> padding) {
  return dnnl::memory::dims(padding.begin() + (padding.size() / 2), padding.end());
}

int64_t DnnlPoolGrad::GetStorageOrder(DnnlNode& node) {
  auto attr = node.Attributes().find("storage_order");
  if (attr != node.Attributes().end()) {
    return static_cast<int>(attr->second().i());
  }
  return 0;
}

dnnl::memory::dims DnnlPoolGrad::GetStrides(DnnlNode& node, PoolShape shape) {
  auto attr = node.Attributes().find("strides");
  std::vector<int64_t> strides;
  if (attr != node.Attributes().end() && !IsGlobalPooling(node)) {
    strides.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      strides.push_back(attr->second().ints(i));
    }
  } else {
    strides.resize(shape, 1);
  }
  return dnnl::memory::dims(strides.begin(), strides.end());
}

dnnl::memory::dims DnnlPoolGrad::InferOutputDims(DnnlNode& node, const dnnl::memory::dims& src_dims, const dnnl::memory::dims& kernel_shape, const dnnl::memory::dims& strides) {
  ORT_ENFORCE(src_dims.size() >= 2);

  dnnl::memory::dims output_dims;
  output_dims.push_back(src_dims[0]);
  output_dims.push_back(src_dims[1]);
  if (IsGlobalPooling(node)) {
    for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
      output_dims.push_back(1);
    }
    return output_dims;
  }

  auto auto_pad = GetAutoPad(node);
  switch (auto_pad) {
    case onnxruntime::AutoPadType::NOTSET: {
      PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
      std::vector<int64_t> padding = GetPadding(node, shape);
      for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
        output_dims.push_back(static_cast<int64_t>(static_cast<float>(src_dims[dim + 2] + padding[dim] + padding[dim + shape] - kernel_shape[dim]) / strides[dim] + 1));
      }
      return output_dims;
      break;
    }
    case onnxruntime::AutoPadType::VALID: {
      for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
        output_dims.push_back((src_dims[dim + 2] - kernel_shape[dim]) / strides[dim] + 1);
      }
      return output_dims;
      break;
    }
    case onnxruntime::AutoPadType::SAME_UPPER: {
      for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
        int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) / strides[dim];
        int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim] - src_dims[dim + 2];
        int64_t out_size = (src_dims[dim + 2] + pad_needed - kernel_shape[dim]) / strides[dim] + 1;
        output_dims.push_back(out_size);
      }
      return output_dims;
      break;
    }
    case onnxruntime::AutoPadType::SAME_LOWER: {
      for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
        int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) / strides[dim];
        int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim] - src_dims[dim + 2];
        int64_t out_size = (src_dims[dim + 2] + pad_needed - kernel_shape[dim]) / strides[dim] + 1;
        output_dims.push_back(out_size);
      }
      return output_dims;
      break;
    }
    default:
      ORT_THROW("Unsupported AutoPad Type.");
      break;
  }
}

// Note OneDNN does not yet support LpPool or GlobalLpPool even though GlobalLpPool is included here.
bool DnnlPoolGrad::IsGlobalPooling(DnnlNode& node) const {
  return (node.OpType() == "GlobalAveragePool" || node.OpType() == "GlobalMaxPool" || node.OpType() == "GlobalLpPool");
}

}  // namespace ort_dnnl
}  // namespace onnxruntime