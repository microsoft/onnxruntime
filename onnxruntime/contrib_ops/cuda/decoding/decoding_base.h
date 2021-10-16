// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/fastertransformer/utils/common.h"


namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class DecodingBase : public CudaKernel
{
public:
  explicit DecodingBase(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info)  {
    ORT_TRY
    {
      check_cuda_error(cublasCreate(&cublas_handle_));
      check_cuda_error(cublasLtCreate(&cublaslt_handle_));
    }
    ORT_CATCH(std::runtime_error& e)
    {
      ORT_THROW(e.what());
    }
  };

  template<typename DataType_>
  void get_tensor(OpKernelContext* context, int tensor_id, const DataType_** tensor_ptr, int offset = 0) const {
    const Tensor* tensor = context->Input<Tensor>(tensor_id);
    const DataType_* data = reinterpret_cast<const DataType_*>(tensor->DataRaw());
    *tensor_ptr = data + offset;

    ORT_ENFORCE(*tensor_ptr != nullptr);
  }

  cublasHandle_t get_cublas_handler() { return cublas_handle_; }
  cublasLtHandle_t get_cublaslt_handler() { return cublaslt_handle_; }

  ~DecodingBase()
  {
    cublasDestroy(cublas_handle_);
    cublasLtDestroy(cublaslt_handle_);
  }

private:
  cublasHandle_t cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
