// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_execution_provider.h"
#include "core/providers/cann/cann_inc.h"
#include "core/providers/cann/cann_call.h"
#include "core/providers/cann/cann_allocator.h"
#include "core/providers/cann/cann_fence.h"
#include "core/providers/cann/cann_fwd.h"
#include "core/providers/cann/npu_data_transfer.h"

using onnxruntime::common::Status;

namespace onnxruntime {

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    auto X_type = ctx->InputType(0);
    if (X_type->IsTensorType()) {
      const auto* X = ctx->Input<Tensor>(0);
      ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
      Tensor* Y = ctx->Output(0, X->Shape());
      ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
      return Info().GetDataTransferManager().CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());
    } else if (X_type->IsSparseTensorType()) {
      const auto* X = ctx->Input<SparseTensor>(0);
      ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
      SparseTensor* Y = ctx->OutputSparse(0, X->DenseShape());
      ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output sparse tensor.");
      return X->Copy(Info().GetDataTransferManager(), Info().GetKernelDef().ExecQueueId(), *Y);
    } else if (X_type->IsTensorSequenceType()) {
      const TensorSeq* X = ctx->Input<TensorSeq>(0);
      ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor sequence is nullptr.");
      TensorSeq* Y = ctx->Output<TensorSeq>(0);
      ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor sequence.");
      auto X_dtype = X->DataType();
      Y->SetType(X_dtype);
      AllocatorPtr alloc;

      if (Node().OpType() == "MemcpyFromHost") {
        auto status = ctx->GetTempSpaceAllocator(&alloc);
        if (!status.IsOK()) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "Memcpy cann: unable to get an allocator.");
        }
      } else {
        auto status = ctx->GetTempSpaceCPUAllocator(&alloc);
        if (!status.IsOK()) {
          return Status(common::ONNXRUNTIME, common::FAIL,
                        "Memcpy cann: unable to get the CPU allocator.");
        }
      }
      auto X_size = X->Size();
      for (size_t i = 0; i < X_size; ++i) {
        const Tensor& source_tensor = X->Get(i);
        std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(), source_tensor.Shape(), alloc);
        Status retval = Info().GetDataTransferManager().CopyTensor(source_tensor, *target_tensor,
                                                                   Info().GetKernelDef().ExecQueueId());
        if (!retval.IsOK()) {
          return retval;
        }
        Y->Add(std::move(*target_tensor));
      }
      return Status::OK();
    }
    return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
  }
};

namespace cann {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .ExecQueueId(kCannStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .ExecQueueId(kCannStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

}  // namespace cann

namespace cann {

// op 1-9
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, double, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 8, float, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, uint8_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, uint16_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, uint32_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, uint64_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, int8_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, int16_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, int32_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, int64_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 10, float, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      7, 8, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      7, 8, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      9, 13, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      9, 13, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      7, 9, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 10, double, Conv);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 12, Identity);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, int32_t, MatMul);

// op 10
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      10, 10, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 10, 11, Dropout);

// op 11
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, uint8_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, uint16_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, uint32_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, uint64_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, int8_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, int16_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, int32_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, int64_t, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, MLFloat16, Flatten);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, 12, float, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, MLFloat16, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, double, Conv);

// op 12
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 12, 12, Dropout);

// op 13
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int32_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int64_t, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, double, Div);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Gemm);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint8_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint16_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint32_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint64_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int8_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int16_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int32_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int64_t, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Flatten);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Flatten);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, Identity);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int32_t, MatMul);

// op 14
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      14, 14, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      14, 14, float, BatchNormalization);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, Identity);

// op 15
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 15, MLFloat16, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 15, float, BatchNormalization);

Status RegisterCANNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // op 1-9
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 8, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 8, MLFloat16, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 8, float, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, uint8_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, uint16_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, uint32_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, uint64_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, int8_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, int16_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, int32_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, int64_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, MLFloat16, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 10, float, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 8, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 13, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 13, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, double, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 10, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 10, double, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      7, 9, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      1, 12, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 8, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 8, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, int32_t, MatMul)>,

      // op 10
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      10, 11, Dropout)>,

      // op 11
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, uint8_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, uint16_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, uint32_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, uint64_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, int8_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, int16_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, int32_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, int64_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, MLFloat16, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            11, 12, float, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, double, Conv)>,

      // op 12
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      12, 12, Dropout)>,

      // op 13
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint8_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint16_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint32_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint64_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int8_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int16_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int32_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int64_t, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      13, 13, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int32_t, MatMul)>,

      // op 14
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, uint8_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int8_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int16_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, uint8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int16_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int8_t, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int16_t, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int32_t, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            14, 14, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            14, 14, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, Identity)>,

      // op 15
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  15, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  15, float, BatchNormalization)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}

}  // namespace cann

CANNExecutionProvider::CANNExecutionProvider(const CANNExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kCannExecutionProvider}, info_{info} {
  CANN_CALL_THROW(aclrtSetDevice(info_.device_id));
  CANN_CALL_THROW(aclrtCreateStream(&stream_));
}

CANNExecutionProvider::~CANNExecutionProvider() {
  CANN_CALL_THROW(aclrtDestroyStream(stream_));
}

// All threads share the same context and stream
Status CANNExecutionProvider::OnRunStart() {
  CANN_CALL_THROW(aclrtSetDevice(info_.device_id));

  return Status::OK();
}

Status CANNExecutionProvider::OnRunEnd(bool sync_stream) {
  if (sync_stream) {
    CANN_CALL_THROW(aclrtSynchronizeStream(stream_));
  }

  return Status::OK();
}

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void InitializeRegistry() {
  s_kernel_registry = KernelRegistry::Create();
  ORT_THROW_IF_ERROR(cann::RegisterCANNKernels(*s_kernel_registry));

  CANN_CALL_THROW(aclInit(nullptr));
}

void DeleteRegistry() {
  s_kernel_registry.reset();

  CANN_CALL_THROW(aclFinalize());
}

std::shared_ptr<KernelRegistry> CANNExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

std::unique_ptr<onnxruntime::IDataTransfer> CANNExecutionProvider::GetDataTransfer() const {
  return std::make_unique<onnxruntime::NPUDataTransfer>(static_cast<aclrtStream>(GetComputeStream()),
                                                        info_.do_copy_in_default_stream);
}

std::vector<std::unique_ptr<ComputeCapability>>
CANNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                     const IKernelLookup& kernel_lookup) const {
  InlinedVector<NodeIndex> candidates;
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      continue;
    }

    const KernelCreateInfo* cann_kernel_def = kernel_lookup.LookUpKernel(node);
    if (cann_kernel_def == nullptr) {
      LOGS_DEFAULT(INFO) << "CANN kernel not found in registries for Op type: " << node.OpType()
                         << " node name: " << node.Name();
      continue;
    }

    candidates.push_back(node.Index());
  }

  auto cpu_nodes = GetCpuPreferredNodes(graph, kernel_lookup, candidates);
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    if (cpu_nodes.count(node_index) > 0)
      continue;

    auto sub_graph = IndexedSubGraph::Create();
    sub_graph->Nodes().push_back(node_index);
    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }

  return result;
}

AllocatorPtr CANNExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return IExecutionProvider::GetAllocator(info_.device_id, mem_type);
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

void CANNExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
  OrtDevice cann_device{OrtDevice::NPU, OrtDevice::MemType::DEFAULT, info_.device_id};
  OrtDevice pinned_device{OrtDevice::CPU, OrtDevice::MemType::CANN_PINNED, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};
  OrtDevice cpu_device{OrtDevice::CPU, OrtDevice::MemType::DEFAULT, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};

  auto cann_alloc = IExecutionProvider::GetAllocator(cann_device.Id(), OrtMemTypeDefault);
  if (!cann_alloc) {
    cann_alloc = allocator_manager.GetAllocator(OrtMemTypeDefault, cann_device);

    if (!cann_alloc) {
      AllocatorCreationInfo default_memory_info(
          [](OrtDevice::DeviceId id) {
            return std::make_unique<CANNAllocator>(id, CANN);
          },
          cann_device.Id(),
          true,
          {info_.default_memory_arena_cfg ? *info_.default_memory_arena_cfg
                                          : OrtArenaCfg(info_.npu_mem_limit,
                                                        static_cast<int>(info_.arena_extend_strategy), -1, -1, -1)});

      cann_alloc = CreateAllocator(default_memory_info);
      allocator_manager.InsertAllocator(cann_alloc);
    }

    InsertAllocator(cann_alloc);
  }

  auto cann_pinned_alloc = IExecutionProvider::GetAllocator(pinned_device.Id(), OrtMemTypeCPUOutput);
  if (!cann_pinned_alloc) {
    cann_pinned_alloc = allocator_manager.GetAllocator(OrtMemTypeCPUOutput, pinned_device);

    if (!cann_pinned_alloc) {
      AllocatorCreationInfo pinned_memory_info(
          [](OrtDevice::DeviceId device_id) {
            return std::make_unique<CANNPinnedAllocator>(device_id, CANN_PINNED);
          },
          pinned_device.Id());

      cann_pinned_alloc = CreateAllocator(pinned_memory_info);
      allocator_manager.InsertAllocator(cann_pinned_alloc);
    }

    InsertAllocator(cann_pinned_alloc);
  }

  auto cann_cpu_alloc = IExecutionProvider::GetAllocator(cpu_device.Id(), OrtMemTypeCPUInput);
  if (!cann_cpu_alloc) {
    cann_cpu_alloc = allocator_manager.GetAllocator(OrtMemTypeCPUInput, cpu_device);

    if (!cann_cpu_alloc) {
      AllocatorCreationInfo cpu_memory_info(
          [](int device_id) {
            return std::make_unique<CPUAllocator>(
                OrtMemoryInfo("CANN_CPU", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), device_id,
                              OrtMemTypeCPUInput));
          },
          cpu_device.Id());

      cann_cpu_alloc = CreateAllocator(cpu_memory_info);
      allocator_manager.InsertAllocator(cann_cpu_alloc);
    }

    InsertAllocator(cann_cpu_alloc);
  }
}

}  // namespace onnxruntime
