// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template<typename DataType>
int64_t ReadIndex(const Tensor& tensor, const char* type_name) {
  DataType data{0};
  if (!CUDA_CALL(cudaMemcpy(&data, tensor.Data<DataType>(),
                            sizeof(DataType), cudaMemcpyDeviceToHost))) {
    ORT_THROW("Cuda: Failed to read tensor data as type: ", type_name, ".");
  }
  return static_cast<int64_t>(data);
}

class SequenceAt final: public CudaKernel {
 public:
  SequenceAt(const OpKernelInfo& info): CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
    ORT_ENFORCE(X != nullptr, "SequenceAt GPU: Got nullptr for sequence input.");
    const Tensor* I = context->Input<Tensor>(1);
    ORT_ENFORCE(I != nullptr, "SequenceAt GPU: Got nullptr input for index tensor.");

    int64_t idx = I->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_INT32 ?
                  ReadIndex<int32_t>(*I, "int32_t") : ReadIndex<int64_t>(*I, "int64_t");
    int64_t sequence_size = static_cast<int64_t>(X->Size());
    if (idx < 0) {
      idx = sequence_size + idx;
    }
    ORT_ENFORCE(idx >= 0 && idx < sequence_size, "SequenceAt GPU: Invalid sequence index.");

    const Tensor& source_tensor = X->Get(idx);
    auto source_type = source_tensor.DataType();
    const void* source_addr = source_tensor.DataRaw(source_type);
 
    Tensor* target_tensor = context->Output(0, source_tensor.Shape());
    ORT_ENFORCE(target_tensor != nullptr, "SequenceAt GPU: Got nullptr for output tensor.");
    void* target_addr = target_tensor->MutableDataRaw(source_type);
 
    if (source_addr != target_addr) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_addr,
                                           source_addr,
                                           source_tensor.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
    }
    return Status::OK();
  }
}; // SequenceAt

class SequenceConstruct final: public CudaKernel {
 public:
  SequenceConstruct(const OpKernelInfo& info): CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override {
    TensorSeq* Y = context->Output<TensorSeq>(0);
    ORT_ENFORCE(Y != nullptr, "SequenceConstruct GPU: Got nullptr for output sequence.");

    AllocatorPtr alloc;
    ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
                "SequenceConstruct GPU: Unable to get an allocator.");

    int32_t at = 0;
    const Tensor* source_tensor = nullptr;
    while (nullptr != (source_tensor = context->Input<Tensor>(at++))) {
      if (1 == at) {
        Y->SetType(source_tensor->DataType());
      }
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor->DataType(),
                                                             source_tensor->Shape(), alloc);
      ORT_ENFORCE(target_tensor, "SequenceConstruct GPU: Failed to allocate new tensor.");
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor->DataRaw(),
                                           source_tensor->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
      Y->Add(std::move(*target_tensor)); // Add will check type consistency inside
    }
    return Status::OK();
  }
}; // SequenceConstruct

class SequenceEmpty final: public CudaKernel {
 public:
  SequenceEmpty(const OpKernelInfo& info): CudaKernel(info) {
    info.GetAttr("dtype", &dtype_);
  }
  Status ComputeInternal(OpKernelContext* context) const override {
    TensorSeq* Y = context->Output<TensorSeq>(0);
    ORT_ENFORCE(Y != nullptr, "SequenceEmpty GPU: Failed to allocate output tensor sequence.");
    Y->SetType(DataTypeImpl::GetTypeFromOnnxType(dtype_));
    return Status::OK();
  }
 private:
  int64_t dtype_ = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}; // SequenceEmpty

class SequenceLength final: public CudaKernel {
 public:
  SequenceLength(const OpKernelInfo& info): CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
    ORT_ENFORCE(X != nullptr, "SequenceLength GPU: Input tensor is nullptr.");
    Tensor* Y = context->Output(0, {});
    ORT_ENFORCE(Y != nullptr, "SequenceLength GPU: Failed to allocate output tensor sequence.");
    auto X_size = static_cast<int64_t>(X->Size());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableDataRaw(),
                                         &X_size,
                                         sizeof(int64_t),
                                         cudaMemcpyHostToDevice, Stream()));
    return Status::OK();
  } // SequenceLength
};
 
} // namespace cuda
} // namespace onnxruntime
