// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "mkldnn_kernel.h"

namespace onnxruntime {
namespace mkl_dnn {

void MklDnnKernel::InitDstReorderOutput(mkldnn::engine& cpu_engine,
                                        mkldnn::memory::data_type& data_type,
                                        std::vector<mkldnn::primitive>& net) {
  // Allocate dst buffer if reorder is necessary
  if (primitive_dst_format_ != ort_source_format_) {
    // reorder to ONNXRuntime format
    mkldnn::memory::dims dst_dims_mkl(
        primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());
    mkldnn::memory::desc dst_des = mkldnn::memory::desc(dst_dims_mkl,
                                                        data_type, ort_source_format_);
    reorder_dst_mem_to_.reset(new mkldnn::memory({dst_des, cpu_engine}, nullptr));
    net.push_back(mkldnn::reorder(*primitive_dst_mem_, *reorder_dst_mem_to_));
  }
}

mkldnn::memory::format MklDnnKernel::GetSourceFormat(int dim_size) {
  mkldnn::memory::format source_format = mkldnn::memory::format::any;
  switch (dim_size) {
    case 1: {
      source_format = mkldnn::memory::format::x;
      break;
    }
    case 2: {
      source_format = mkldnn::memory::format::nc;
      break;
    }
    case 3: {
      source_format = mkldnn::memory::format::ntc;
      break;
    }
    case 4: {
      source_format = mkldnn::memory::format::nchw;
      break;
    }
    case 5: {
      source_format = mkldnn::memory::format::ncdhw;
      break;
    }
    default: {
      source_format = mkldnn::memory::format::any;
      break;
    }
  }

  return source_format;
}

}  // namespace mkl_dnn
}  // namespace onnxruntime