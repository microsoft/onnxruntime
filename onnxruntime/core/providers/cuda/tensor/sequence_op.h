// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/tensor/concat.h"
#include "core/providers/cuda/tensor/concat_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename DataType>
int64_t ReadIndex(const Tensor& tensor, const char* type_name) {
  DataType data{0};
  if (!CUDA_CALL(cudaMemcpy(&data, tensor.Data<DataType>(),
                            sizeof(DataType), cudaMemcpyDeviceToHost))) {
    ORT_THROW("Cuda: Failed to read tensor data as type: ", type_name, ".");
  }
  return static_cast<int64_t>(data);
}

class SequenceAt final : public CudaKernel {
 public:
  SequenceAt(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
    ORT_ENFORCE(X != nullptr, "SequenceAt GPU: Got nullptr for sequence input.");
    const Tensor* I = context->Input<Tensor>(1);
    ORT_ENFORCE(I != nullptr, "SequenceAt GPU: Got nullptr input for index tensor.");

    int64_t idx = I->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_INT32 ? ReadIndex<int32_t>(*I, "int32_t") : ReadIndex<int64_t>(*I, "int64_t");
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
};  // SequenceAt

class SequenceConstruct final : public CudaKernel {
 public:
  SequenceConstruct(const OpKernelInfo& info) : CudaKernel(info) {}
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
      Y->Add(std::move(*target_tensor));  // Add will check type consistency inside
    }
    return Status::OK();
  }
};  // SequenceConstruct

class SequenceEmpty final : public CudaKernel {
 public:
  SequenceEmpty(const OpKernelInfo& info) : CudaKernel(info) {
    if (!info.GetAttr("dtype", &dtype_).IsOK()) {
      dtype_ = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }
  }
  Status ComputeInternal(OpKernelContext* context) const override {
    TensorSeq* Y = context->Output<TensorSeq>(0);
    ORT_ENFORCE(Y != nullptr, "SequenceEmpty GPU: Failed to allocate output tensor sequence.");
#ifdef SHARED_PROVIDER
    Y->SetType(DataTypeImpl::GetTypeFromOnnxType(static_cast<int>(dtype_)));
#else
    Y->SetType(DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(dtype_))->GetElementType());
#endif
    return Status::OK();
  }

 private:
  int64_t dtype_{};
};  // SequenceEmpty

class SequenceLength final : public CudaKernel {
 public:
  SequenceLength(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
    ORT_ENFORCE(X != nullptr, "SequenceLength GPU: Input tensor sequence is nullptr.");
    Tensor* Y = context->Output(0, {});
    ORT_ENFORCE(Y != nullptr, "SequenceLength GPU: Failed to allocate output tensor sequence.");
    auto X_size = static_cast<int64_t>(X->Size());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y->MutableDataRaw(),
                                         &X_size,
                                         sizeof(int64_t),
                                         cudaMemcpyHostToDevice, Stream()));
    return Status::OK();
  }
};  // SequenceLength

class ConcatFromSequence final : public CudaKernel, public ConcatBase {
 public:
  ConcatFromSequence(const OpKernelInfo& info) : CudaKernel(info), ConcatBase(info, true) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
    ORT_ENFORCE(X != nullptr, "ConcatFromSequence GPU: Input tensor sequence is nullptr.");
    int64_t input_count = static_cast<int64_t>(X->Size());
    std::vector<const Tensor*> input_tensors;
    for (int64_t i = 0; i < input_count; ++i) {
      input_tensors.push_back(&X->Get(i));
    }
    Prepare p;
    ORT_RETURN_IF_ERROR(PrepareForCompute(context, input_tensors, p));
    if (0 == p.output_num_elements) {
      return Status::OK();
    }

    int64_t initial_output_offset = 0;
    auto element_bytes = p.output_tensor->DataType()->Size();
    for (int input_index = 0; input_index < input_count; input_index++) {
      const auto& prep = p.inputs[input_index];
      if (prep.num_elements == 0) {
        continue;
      }
      auto input_axis_pitch = prep.axis_pitch;
      const uint8_t* input = static_cast<const uint8_t*>(prep.tensor->DataRaw());

      auto input_size = prep.num_elements;
      uint8_t* output = static_cast<uint8_t*>(p.output_tensor->MutableDataRaw());
      int64_t cur_out_offset = 0;
      int64_t cur_in_offset = 0;
      for (size_t idx_copy = 0, end = input_size / input_axis_pitch; idx_copy < end; ++idx_copy) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
            output + (initial_output_offset + cur_out_offset) * element_bytes,
            input + cur_in_offset * element_bytes, input_axis_pitch * element_bytes,
            cudaMemcpyHostToDevice, Stream()));
        cur_out_offset += p.output_axis_pitch;
        cur_in_offset += input_axis_pitch;
      }
      initial_output_offset += input_axis_pitch;
    }
    return Status::OK();
  }
};  // ConcatFromSequence

class SequenceErase final : public CudaKernel {
 public:
  SequenceErase(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
    ORT_ENFORCE(X != nullptr, "SequenceAt GPU: Got nullptr for sequence input.");
    int64_t X_size = static_cast<int64_t>(X->Size());
    int64_t idx = X_size - 1;
    const Tensor* I = context->Input<Tensor>(1);
    if (I != nullptr) {
      idx = I->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_INT32 ? ReadIndex<int32_t>(*I, "int32_t") : ReadIndex<int64_t>(*I, "int64_t");
      if (idx < 0) {
        idx = X_size + idx;
      }
      ORT_ENFORCE(idx >= 0 && idx < X_size, "SequenceErase GPU: Invalid sequence index.");
    }

    AllocatorPtr alloc;
    ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
                "SequenceErase GPU: Unable to get an allocator.");
    TensorSeq* Y = context->Output<TensorSeq>(0);
    ORT_ENFORCE(Y != nullptr, "SequenceErase GPU: Failed to allocate output tensor sequence.");
    Y->SetType(X->DataType());
    for (int64_t i = 0; i < X_size; ++i) {
      if (i == idx) {
        continue;
      }
      const Tensor& source_tensor = X->Get(i);
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                             source_tensor.Shape(), alloc);

      ORT_ENFORCE(target_tensor, "SequenceErase GPU: Failed to allocate new tensor.");
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor.DataRaw(),
                                           source_tensor.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
      Y->Add(std::move(*target_tensor));
    }
    return Status::OK();
  }
};  // SequenceErase

class SequenceInsert final : public CudaKernel {
 public:
  SequenceInsert(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* S = context->Input<TensorSeq>(0);
    ORT_ENFORCE(S != nullptr, "SequenceInsert GPU: Got nullptr for sequence input.");
    int64_t S_size = static_cast<int64_t>(S->Size());
    int64_t idx = S_size;
    const Tensor* I = context->Input<Tensor>(2);
    if (I != nullptr) {
      idx = I->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_INT32 ? ReadIndex<int32_t>(*I, "int32_t") : ReadIndex<int64_t>(*I, "int64_t");
      if (idx < 0) {
        idx = S_size + idx;
      }
      ORT_ENFORCE(idx >= 0 && idx <= S_size, "SequenceInsert GPU: Invalid sequence index.");
    }
    const Tensor* X = context->Input<Tensor>(1);
    ORT_ENFORCE(X != nullptr, "SequenceInsert GPU: Got nullptr for tensor input.");

    AllocatorPtr alloc;
    ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
                "SequenceConstruct GPU: Unable to get an allocator.");

    TensorSeq* Y = context->Output<TensorSeq>(0);
    ORT_ENFORCE(Y != nullptr, "SequenceInsert GPU: Failed to allocate output tensor sequence.");
    Y->SetType(S->DataType());
    for (int64_t i = 0; i < S_size; ++i) {
      if (i == idx) {
        std::unique_ptr<Tensor> target_tensor = Tensor::Create(X->DataType(),
                                                               X->Shape(), alloc);
        ORT_ENFORCE(target_tensor, "SequenceInsert GPU: Failed to allocate new tensor.");
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                             X->DataRaw(), X->SizeInBytes(),
                                             cudaMemcpyDeviceToDevice, Stream()));
        Y->Add(std::move(*target_tensor));
      }
      const Tensor& source_tensor = S->Get(i);
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                             source_tensor.Shape(), alloc);
      ORT_ENFORCE(target_tensor, "SequenceInsert GPU: Failed to allocate new tensor.");
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor.DataRaw(),
                                           source_tensor.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
      Y->Add(std::move(*target_tensor));  // Add will check type consistency inside
    }                                     // for
    if (idx == S_size) {
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(X->DataType(),
                                                             X->Shape(), alloc);
      ORT_ENFORCE(target_tensor, "SequenceInsert GPU: Failed to allocate new tensor.");
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           X->DataRaw(), X->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
      Y->Add(std::move(*target_tensor));
    }
    return Status::OK();
  }
};  // SequenceInsert

}  // namespace cuda
}  // namespace onnxruntime
