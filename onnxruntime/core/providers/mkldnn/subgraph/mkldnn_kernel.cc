// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "mkldnn_kernel.h"

namespace onnxruntime {
namespace mkl_dnn {

void MklDnnKernel::InitDstReorderOutput(mkldnn::engine& cpu_engine,
                                        mkldnn::memory::data_type& data_type,
                                        std::vector<mkldnn::primitive>& net,
                                        std::vector<std::unordered_map<int, mkldnn::memory>>& net_args) {
  // Allocate dst buffer if reorder is necessary
  if (primitive_dst_desc_ != ort_source_desc_) {
    // reorder to ONNXRuntime format
    mkldnn::memory::dims dst_dims_mkl(
        primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());
    mkldnn::memory::desc dst_des = mkldnn::memory::desc(dst_dims_mkl,
                                                        data_type, ort_source_format_);
    reorder_dst_mem_to_ = onnxruntime::make_unique<mkldnn::memory>(
        mkldnn::memory(dst_des, cpu_engine));
    net.push_back(mkldnn::reorder(*primitive_dst_mem_, *reorder_dst_mem_to_));
    net_args.push_back({{MKLDNN_ARG_FROM, *primitive_dst_mem_},
                        {MKLDNN_ARG_TO, *reorder_dst_mem_to_}});
  }
}

mkldnn::memory::format_tag MklDnnKernel::GetSourceFormat(int dim_size) {
  mkldnn::memory::format_tag source_format = mkldnn::memory::format_tag::any;
  switch (dim_size) {
    case 1: {
      source_format = mkldnn::memory::format_tag::x;
      break;
    }
    case 2: {
      source_format = mkldnn::memory::format_tag::nc;
      break;
    }
    case 3: {
      source_format = mkldnn::memory::format_tag::ntc;
      break;
    }
    case 4: {
      source_format = mkldnn::memory::format_tag::nchw;
      break;
    }
    case 5: {
      source_format = mkldnn::memory::format_tag::ncdhw;
      break;
    }
    default: {
      source_format = mkldnn::memory::format_tag::any;
      break;
    }
  }

  return source_format;
}

}  // namespace mkl_dnn
}  // namespace onnxruntime