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
/*
void MklDnnKernel::AllocateMemoryAndReorderIfNeeded(ONNXRunTimeTensor* const output_tensors, const DType& dtype) {
  // End of sub-graph. Allocate memory and get the output
  auto& y_dims = primitive_dst_shape_.GetDims();
  AllocateOutputTensor(output_tensors, mklnode_ptr_->output_index,
                       &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()),
                       dtype);
  if (primitive_dst_format_ != ort_source_format_) {
    reorder_dst_mem_to_->set_data_handle(output_tensors[mklnode_ptr_->output_index].data);
  } else {
    primitive_dst_mem_->set_data_handle(output_tensors[mklnode_ptr_->output_index].data);
  }
}

void MklDnnKernel::AllocateOutputTensor(ONNXRunTimeTensor* const output_tensors,
                                        int index, const int64_t* shape, size_t dim) {
  output_tensors[index].dtype = dtype;
  output_tensors[index].ndim = dim;
  output_tensors[index].shape = new int64_t[dim];
  memcpy(output_tensors[index].shape, shape, sizeof(int64_t) * dim);
  int64_t data_size = 1;
  for (auto j = 0; j < output_tensors[index].ndim; j++)
    data_size *= output_tensors[index].shape[j];
  output_tensors[index].data = (*(mkl_context_->allocate_func))(mkl_context_->allocator, sizeof(double) * data_size, 64);
}
*/

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