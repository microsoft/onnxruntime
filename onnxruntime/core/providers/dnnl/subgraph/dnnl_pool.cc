// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_pool.h"
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlPool::DnnlPool() {}

void DnnlPool::CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node) {
  auto dnnl_engine = sp.GetEngine();
#ifdef ENABLE_TRAINING
  // When using training the memory needs to be in a format known to pool_forward and the
  // pool_backward primitives. Since we don't currently have a way to pass the memory format
 // from pool_forward to pool_backward; we are choosing to use Onnxruntime's memory format
 // as the common memory format to be used by both forward and the backward primitives.
 auto pool_src_mem = sp.GetMemoryInOrtFormat(node.Input(IN_X), dnnl_engine);
#else
  auto pool_src_mem = sp.GetMemory(node.Input(IN_X));
#endif  // ENABLE_TRAINING
  auto src_md = pool_src_mem.get_desc();
  auto src_dims = pool_src_mem.get_desc().dims();

  #ifdef ENABLE_TRAINING
  auto prop_kind = dnnl::prop_kind::forward;
#else
  auto prop_kind = dnnl::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

  dnnl::algorithm algo = dnnl::algorithm::pooling_max;
  if (node.OpType() == "AveragePool" || node.OpType() == "GlobalAveragePool") {
    algo = dnnl::algorithm::pooling_avg_exclude_padding;
    if (GetCountIncludePadding(node) != 0) {
      algo = dnnl::algorithm::pooling_avg_include_padding;
    }
  }

  auto kernel_shape = GetKernelShape(src_dims, node);
  PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
  auto strides = GetStrides(node, shape);

  auto dst_mem_dims = InferOutputDims(node, src_dims, kernel_shape, strides);
  dnnl::memory::desc dst_md = dnnl::memory::desc(dst_mem_dims, node.Input(IN_X).Type(), dnnl::memory::format_tag::any);

  auto padding = InferPadding(node, src_dims, kernel_shape, strides);
  auto padding_left = GetPaddingLeft(padding);
  auto padding_right = GetPaddingRight(padding);



  auto pool_desc = dnnl::pooling_forward::desc(prop_kind, algo,
                                               src_md, dst_md,
                                               strides, kernel_shape,
                                               padding_left, padding_right);

  auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, dnnl_engine);

#ifndef ENABLE_TRAINING
  // If using GPU this will move the memory from the CPU to the GPU.
  pool_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), pool_pd.src_desc(), dnnl_engine);
#endif
  dnnl::memory pool_dst_mem = dnnl::memory(pool_pd.dst_desc(), dnnl_engine);

  auto pool_op = dnnl::pooling_forward(pool_pd);
#ifdef ENABLE_TRAINING
  auto pool_workspace_mem = dnnl::memory(pool_pd.workspace_desc(), dnnl_engine);

  sp.AddPrimitive(pool_op, {{DNNL_ARG_SRC, pool_src_mem},
                            {DNNL_ARG_WORKSPACE, pool_workspace_mem},
                            {DNNL_ARG_DST, pool_dst_mem}});
#else
  sp.AddPrimitive(pool_op, {{DNNL_ARG_SRC, pool_src_mem},
                            {DNNL_ARG_DST, pool_dst_mem}});
#endif  //ENABLE_TRAINING


  sp.SetMemory(node.Output(OUT_Y), pool_dst_mem);
#ifdef ENABLE_TRAINING
  if (node.OutputCount() == 2) {
    sp.SetMemory(node.Output(OUT_INDICES), pool_workspace_mem);
  }
#endif  // ENABLE_TRAINING
}


AutoPadType DnnlPool::GetAutoPad(DnnlNode& node) {
  std::string auto_pad;
  auto attr = node.Attributes().find("auto_pad");
  if (attr != node.Attributes().end() &&
      attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
    auto_pad = attr->second().s();
  }
  return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

int64_t DnnlPool::GetCeilMode(DnnlNode& node) {
  auto attr = node.Attributes().find("ceil_mode");
  if (attr != node.Attributes().end()) {
    return attr->second().i();
  }
  return false;
}

int64_t DnnlPool::GetCountIncludePadding(DnnlNode& node) {
  auto attr = node.Attributes().find("count_include_pad");
  if (attr != node.Attributes().end()) {
    return attr->second().i();
  }
  return 0;
}

dnnl::memory::dims DnnlPool::GetDilations(DnnlNode& node, PoolShape shape) {
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

dnnl::memory::dims DnnlPool::GetKernelShape(const dnnl::memory::dims& src_dims, DnnlNode& node) {
  auto attr = node.Attributes().find("kernel_shape");
  std::vector<int64_t> kernel_shape;
  if (attr != node.Attributes().end()) {
    kernel_shape.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      kernel_shape.push_back(attr->second().ints(i));
    }
    return kernel_shape;
  }

  kernel_shape = std::vector<int64_t>(src_dims.begin() + 2, src_dims.end());
  return kernel_shape;
}

std::vector<int64_t> DnnlPool::InferPadding(DnnlNode& node, const dnnl::memory::dims& src_dims, const dnnl::memory::dims& kernel_shape, const dnnl::memory::dims& strides) {
  auto auto_pad = GetAutoPad(node);
  PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
  std::vector<int64_t> padding;
  switch (auto_pad) {
    case onnxruntime::AutoPadType::NOTSET:{
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

std::vector<int64_t> DnnlPool::GetPadding(DnnlNode& node, PoolShape shape) {
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

dnnl::memory::dims DnnlPool::GetPaddingLeft(const std::vector<int64_t> padding) {
  return dnnl::memory::dims(padding.begin(), padding.begin() + (padding.size() / 2));
}

dnnl::memory::dims DnnlPool::GetPaddingRight(const std::vector<int64_t> padding) {
  return dnnl::memory::dims(padding.begin() + (padding.size() / 2), padding.end());
}

int64_t DnnlPool::GetStorageOrder(DnnlNode& node) {
  auto attr = node.Attributes().find("storage_order");
  if (attr != node.Attributes().end()) {
    return static_cast<int>(attr->second().i());
  }
  return 0;
}

dnnl::memory::dims DnnlPool::GetStrides(DnnlNode& node, PoolShape shape) {
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

dnnl::memory::dims DnnlPool::InferOutputDims(DnnlNode& node, const dnnl::memory::dims& src_dims, const dnnl::memory::dims& kernel_shape, const dnnl::memory::dims& strides) {
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
        output_dims.push_back(static_cast<int64_t>( static_cast<float>(src_dims[dim + 2] + padding[dim] + padding[dim + shape] - kernel_shape[dim]) / strides[dim] + 1));
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
bool DnnlPool::IsGlobalPooling(DnnlNode& node) const {
  return (node.OpType() == "GlobalAveragePool" || node.OpType() == "GlobalMaxPool" || node.OpType() == "GlobalLpPool");
}

}  // namespace ort_dnnl
}  // namespace onnxruntime