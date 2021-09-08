// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/tensor/concat.h"
#include "core/providers/cuda/tensor/concat_impl.h"

namespace onnxruntime {
namespace cuda {

class SequenceAt final : public CudaKernel {
 public:
  SequenceAt(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
    const Tensor* I = context->Input<Tensor>(1);

    int64_t idx = -1;
    if (I->IsDataType<int32_t>()) {
      idx = static_cast<int64_t>(I->Data<int32_t>()[0]);
    } else {
      idx = I->Data<int64_t>()[0];
    }

    int64_t sequence_size = static_cast<int64_t>(X->Size());
    if (idx < 0) {
      idx = sequence_size + idx;
    }
    ORT_ENFORCE(idx >= 0 && idx < sequence_size, "SequenceAt GPU: Invalid sequence index.");

    const Tensor& source_tensor = X->Get(idx);
    auto source_type = source_tensor.DataType();
    const void* source_addr = source_tensor.DataRaw(source_type);

    Tensor* target_tensor = context->Output(0, source_tensor.Shape());
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
    auto num_inputs = Node().InputArgCount().front();
    ORT_ENFORCE(num_inputs >= 1, "Must have 1 or more inputs");

    MLDataType first_dtype = context->Input<Tensor>(0)->DataType();

    AllocatorPtr alloc;
    ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
                "SequenceConstruct GPU: Unable to get an allocator.");

    TensorSeq* Y = context->Output<TensorSeq>(0);
    Y->SetType(first_dtype);
    Y->Reserve(num_inputs);

    for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
      const auto* source_tensor = context->Input<Tensor>(input_idx);

      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor->DataType(),
                                                             source_tensor->Shape(), alloc);

      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor->DataRaw(),
                                           source_tensor->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));

      Y->Add(std::move(*target_tensor));  // Add will check for type consistency
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
    Tensor* Y = context->Output(0, {});
    Y->MutableData<int64_t>()[0] = static_cast<int64_t>(X->Size());
    return Status::OK();
  }
};  // SequenceLength

class ConcatFromSequence final : public CudaKernel, public ConcatBase {
 public:
  ConcatFromSequence(const OpKernelInfo& info) : CudaKernel(info), ConcatBase(info, true) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* X = context->Input<TensorSeq>(0);
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
    int64_t X_size = static_cast<int64_t>(X->Size());
    int64_t idx = X_size - 1;
    const Tensor* I = context->Input<Tensor>(1);
    if (I != nullptr) {
      if (I->IsDataType<int32_t>()) {
        idx = static_cast<int64_t>(I->Data<int32_t>()[0]);
      } else {
        idx = I->Data<int64_t>()[0];
      }

      if (idx < 0) {
        idx = X_size + idx;
      }
      ORT_ENFORCE(idx >= 0 && idx < X_size, "SequenceErase GPU: Invalid sequence index.");
    }

    AllocatorPtr alloc;
    ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
                "SequenceErase GPU: Unable to get an allocator.");

    TensorSeq* Y = context->Output<TensorSeq>(0);
    Y->SetType(X->DataType());
    Y->Reserve(X_size - 1);

    for (int64_t i = 0; i < X_size; ++i) {
      if (i == idx) {
        continue;
      }
      const Tensor& source_tensor = X->Get(i);
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                             source_tensor.Shape(), alloc);

      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor.DataRaw(),
                                           source_tensor.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
      Y->Add(std::move(*target_tensor));  // Add will check for type consistency
    }

    return Status::OK();
  }
};  // SequenceErase

class SequenceInsert final : public CudaKernel {
 public:
  SequenceInsert(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override {
    const TensorSeq* S = context->Input<TensorSeq>(0);
    int64_t S_size = static_cast<int64_t>(S->Size());
    int64_t idx = S_size;
    const Tensor* I = context->Input<Tensor>(2);
    if (I != nullptr) {
      if (I->IsDataType<int32_t>()) {
        idx = static_cast<int64_t>(I->Data<int32_t>()[0]);
      } else {
        idx = I->Data<int64_t>()[0];
      }

      if (idx < 0) {
        idx = S_size + idx;
      }
      ORT_ENFORCE(idx >= 0 && idx <= S_size, "SequenceInsert GPU: Invalid sequence index.");
    }
    const Tensor* X = context->Input<Tensor>(1);

    AllocatorPtr alloc;
    ORT_ENFORCE(context->GetTempSpaceAllocator(&alloc).IsOK(),
                "SequenceInsert GPU: Unable to get an allocator.");

    std::unique_ptr<Tensor> tensor_to_be_inserted = Tensor::Create(X->DataType(),
                                                                   X->Shape(), alloc);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(tensor_to_be_inserted->MutableDataRaw(),
                                         X->DataRaw(), X->SizeInBytes(),
                                         cudaMemcpyDeviceToDevice, Stream()));

    TensorSeq* Y = context->Output<TensorSeq>(0);
    Y->SetType(S->DataType());
    Y->Reserve(S_size + 1);

    for (int64_t i = 0; i < S_size; ++i) {
      if (i == idx) {
        Y->Add(std::move(*tensor_to_be_inserted));  // Add will check for type consistency
      }
      const Tensor& source_tensor = S->Get(i);
      std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                             source_tensor.Shape(), alloc);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(),
                                           source_tensor.DataRaw(),
                                           source_tensor.SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
      Y->Add(std::move(*target_tensor));  // Add will check for type consistency
    }

    if (idx == S_size) {
      Y->Add(std::move(*tensor_to_be_inserted));  // Add will check for type consistency
    }

    return Status::OK();
  }
};  // SequenceInsert

}  // namespace cuda
}  // namespace onnxruntime
