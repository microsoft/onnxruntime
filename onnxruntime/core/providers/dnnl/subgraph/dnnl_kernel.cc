// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "dnnl_kernel.h"

namespace onnxruntime {
namespace ort_dnnl {

void DnnlKernel::InitDstReorderOutput(dnnl::engine& cpu_engine,
                                      dnnl::memory::data_type& data_type,
                                      std::vector<dnnl::primitive>& net,
                                      std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
                                      bool gpu_available) {
  // Allocate dst buffer if reorder is necessary
  if (primitive_dst_desc_ != ort_source_desc_ || gpu_available) {
    // reorder to ONNXRuntime format
    dnnl::memory::dims dst_dims_mkl(
        primitive_dst_shape_.GetDims().begin(), primitive_dst_shape_.GetDims().end());
    dnnl::memory::desc dst_des = dnnl::memory::desc(dst_dims_mkl,
                                                    data_type, ort_source_format_);
    reorder_dst_mem_to_ = std::make_unique<dnnl::memory>(
        dnnl::memory(dst_des, cpu_engine));
    net.push_back(dnnl::reorder(*primitive_dst_mem_, *reorder_dst_mem_to_));
    net_args.push_back({{DNNL_ARG_FROM, *primitive_dst_mem_},
                        {DNNL_ARG_TO, *reorder_dst_mem_to_}});
  }
}

dnnl::memory::format_tag DnnlKernel::GetSourceFormat(int dim_size) {
  dnnl::memory::format_tag source_format = dnnl::memory::format_tag::any;
  switch (dim_size) {
    case 1: {
      source_format = dnnl::memory::format_tag::x;
      break;
    }
    case 2: {
      source_format = dnnl::memory::format_tag::nc;
      break;
    }
    case 3: {
      source_format = dnnl::memory::format_tag::ncw;
      break;
    }
    case 4: {
      source_format = dnnl::memory::format_tag::nchw;
      break;
    }
    case 5: {
      source_format = dnnl::memory::format_tag::ncdhw;
      break;
    }
    default: {
      source_format = dnnl::memory::format_tag::any;
      break;
    }
  }

  return source_format;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
