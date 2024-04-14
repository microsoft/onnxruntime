// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <utility>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <map>
#include <unordered_set>

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/cann/cann_execution_provider.h"
#include "core/providers/cann/cann_inc.h"
#include "core/providers/cann/cann_call.h"
#include "core/providers/cann/cann_allocator.h"
#include "core/providers/cann/cann_fwd.h"
#include "core/providers/cann/cann_stream_handle.h"
#include "core/providers/cann/npu_data_transfer.h"

using onnxruntime::cann::BuildONNXModel;
using onnxruntime::cann::CannModelPreparation;
using onnxruntime::cann::ParserONNXModel;
using onnxruntime::cann::SupportONNXModel;
using onnxruntime::common::Status;

namespace onnxruntime {

// Models can only be parsed and built serially in the same process
OrtMutex g_mutex;

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
      auto* npu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(X->Location().device,
                                                                                Y->Location().device);
      ORT_RETURN_IF_ERROR(npu_data_transfer->CopyTensorAsync(*X, *Y, *ctx->GetComputeStream()));
      return Status::OK();
    } else {
      if (X_type->IsSparseTensorType()) {
        aclrtSynchronizeStream(static_cast<aclrtStream>(ctx->GetComputeStream()->GetHandle()));
        const auto* X = ctx->Input<SparseTensor>(0);
        ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
        SparseTensor* Y = ctx->OutputSparse(0, X->DenseShape());
        ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output sparse tensor.");
        return X->Copy(Info().GetDataTransferManager(), *Y);
      } else if (X_type->IsTensorSequenceType()) {
        const TensorSeq* X = ctx->Input<TensorSeq>(0);
        ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor sequence is nullptr.");
        TensorSeq* Y = ctx->Output<TensorSeq>(0);
        ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor sequence.");
        auto X_dtype = X->DataType();
        Y->SetType(X_dtype);
        AllocatorPtr alloc;

        // If we are copying contents to CANN, the allocator to use
        // to allocate the buffers of the new tensors in the sequence
        // can be temp space allocator associated with the CANN EP
        if (Node().OpType() == "MemcpyFromHost") {
          auto status = ctx->GetTempSpaceAllocator(&alloc);
          if (!status.IsOK()) {
            return Status(common::ONNXRUNTIME, common::FAIL,
                          "Memcpy cann: unable to get an allocator.");
          }
        } else {
          // If we are copying contents to CPU (op type is "MemcpyToHost"),
          // the allocator to use to allocate the buffers of the new tensors
          // in the sequence will be the allocator from the CPU EP
          auto status = ctx->GetTempSpaceCPUAllocator(&alloc);
          if (!status.IsOK()) {
            return Status(common::ONNXRUNTIME, common::FAIL,
                          "Memcpy cann: unable to get the CPU allocator.");
          }
        }
        auto X_size = X->Size();
        for (size_t i = 0; i < X_size; ++i) {
          const Tensor& source_tensor = X->Get(i);
          std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(),
                                                                 source_tensor.Shape(),
                                                                 alloc);
          auto* npu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(source_tensor.Location().device,
                                                                                    target_tensor->Location().device);
          ORT_RETURN_IF_ERROR(npu_data_transfer->CopyTensorAsync(source_tensor,
                                                                 *target_tensor,
                                                                 *ctx->GetComputeStream()));
          Y->Add(std::move(*target_tensor));
        }
        return Status::OK();
      }
      return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy: Unsupported input type.");
    }
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
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
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
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 12, double, Mul);
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
                                                      7, 8, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      9, 13, float, BatchNormalization);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      7, 9, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 12, Identity);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, int32_t, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 8, bool, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, uint8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, uint16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, uint32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, uint64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, int8_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, int16_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, int32_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, int64_t, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, float, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, double, Cast);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, bool, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 4, Reshape);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 5, 12, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, uint8_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, uint16_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, uint32_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, uint64_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, int8_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, int16_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, int32_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, int64_t, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, MLFloat16, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, float, Transpose);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, bool, Transpose);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, float, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, int32_t, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, float, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, int8_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, int32_t, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Neg);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, float, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Floor);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, float, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, float, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      6, 12, MLFloat16, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Reciprocal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, float, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Sqrt);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, float, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, float, Log);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 1, 5, MLFloat16, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Exp);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, MLFloat16, Erf);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 9, 12, float, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, MLFloat16, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, float, Sin);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, MLFloat16, Cos);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 7, float, Cos);

// op 10
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      10, 10, MLFloat16, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 10, 10, double, AveragePool);
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
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, double, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, MLFloat16, Round);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 11, float, Round);

// op 12
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 12, 12,
                                                      MLFloat16_MLFloat16, Dropout);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 12, 12,
                                                      float_float, Dropout);

// op 13
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int32_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int64_t, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, double, Add);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int32_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int64_t, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, double, Sub);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int32_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, int64_t, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, MLFloat16, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, float, Mul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, double, Mul);
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
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16_MLFloat16, Dropout);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float_float, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, Identity);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, BFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int32_t, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, uint64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, double, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, bool, Cast);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, 13, Reshape);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, MLFloat16, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 13, float, Erf);

// op 14
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint8_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint16_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint16_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint8_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, uint16_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int8_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int16_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int32_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, int64_t, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, double, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                      14, 14, float, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                14, 18, Identity);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, Reshape);

// op 15
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 15, float, BatchNormalization);

// op 19
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 19, Identity);

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
                                                                            7, 12, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 12, double, Mul)>,
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
                                                                            7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 13, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, double, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            7, 9, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 7, double, MaxPool)>,
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
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 8, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      1, 4, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      5, 12, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, uint8_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, uint16_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, uint32_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, uint64_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, int8_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, int16_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, int32_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, int64_t, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, MLFloat16, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, float, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  1, bool, Transpose)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            1, 5, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            6, 12, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            9, 12, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  7, MLFloat16, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  7, float, Sin)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  7, MLFloat16, Cos)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  7, float, Cos)>,

      // op 10
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            10, 10, double, AveragePool)>,
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
                                                                  11, double, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, MLFloat16, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, MLFloat16, Round)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  11, float, Round)>,

      // op 12
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            12, 12, MLFloat16_MLFloat16, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            12, 12, float_float, Dropout)>,

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
                                                                            13, 13, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            13, 13, double, Mul)>,
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
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16_MLFloat16, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float_float, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      13, 13, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, BFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int32_t, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      13, 13, Reshape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  13, float, Erf)>,

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
                                                                  14, uint8_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, uint16_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int8_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int16_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, uint8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, uint16_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int8_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int16_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, uint8_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, uint16_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int8_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, int16_t, Div)>,
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
                                                                  14, int64_t, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  14, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                            14, 14, float, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                      14, 18, Identity)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 14, Reshape)>,

      // op 15
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain,
                                                                  15, float, BatchNormalization)>,

      // op 19
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCannExecutionProvider, kOnnxDomain, 19, Identity)>,
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
    : IExecutionProvider{onnxruntime::kCannExecutionProvider, OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, info.device_id)}, info_{info} {
  InitProviderOrtApi();

  CANN_CALL_THROW(aclrtSetDevice(info_.device_id));

  soc_name_ = aclrtGetSocName();
  ORT_ENFORCE(soc_name_ != nullptr, "aclrtGetSocName return nullptr");
  metadef_id_generator_ = ModelMetadefIdGenerator::Create();
}

CANNExecutionProvider::~CANNExecutionProvider() {
  for (auto modelID : modelIDs_) {
    CANN_CALL_THROW(aclmdlUnload(modelID.second));
  }
}

// All threads share the same context and stream
Status CANNExecutionProvider::OnRunStart(const onnxruntime::RunOptions& /*run_options*/) {
  CANN_RETURN_IF_ERROR(aclrtSetDevice(info_.device_id));

  return Status::OK();
}

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void InitializeRegistry() {
  CANN_CALL_THROW(aclInit(nullptr));

  s_kernel_registry = KernelRegistry::Create();
  ORT_THROW_IF_ERROR(cann::RegisterCANNKernels(*s_kernel_registry));
}

void DeleteRegistry() {
  s_kernel_registry.reset();

  ge::aclgrphBuildFinalize();

  CANN_CALL_THROW(aclFinalize());
}

std::shared_ptr<KernelRegistry> CANNExecutionProvider::GetKernelRegistry() const {
  return s_kernel_registry;
}

std::unique_ptr<onnxruntime::IDataTransfer> CANNExecutionProvider::GetDataTransfer() const {
  return std::make_unique<onnxruntime::NPUDataTransfer>();
}

std::unique_ptr<IndexedSubGraph> CANNExecutionProvider::GetSubGraph(
    const std::vector<std::size_t>& graph_nodes_index,
    const GraphViewer& graph_viewer) const {
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.size());
  for (const auto& index : graph_nodes_index) {
    node_set.insert(index);
  }

  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph_viewer.GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }

  // Find inputs and outputs of the subgraph
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::IndexedSubGraph::Create();
  std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add, graph_outputs_to_add;
  std::unordered_set<const NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  std::vector<std::string> initializers;
  for (const auto& index : graph_nodes_index) {
    sub_graph->Nodes().push_back(index);
    const auto& node = graph_viewer.GetNode(index);
    for (const auto& input : node->InputDefs()) {
      if (graph_viewer.IsConstantInitializer(input->Name(), true)) {
        initializers.push_back(input->Name());
        continue;
      }
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    for (const auto& input : node->ImplicitInputDefs()) {
      if (graph_viewer.IsConstantInitializer(input->Name(), true)) {
        initializers.push_back(input->Name());
        continue;
      }
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    // For output searching, there are two special cases,
    // One is, if node's OutputEdges are more than its outputs, meaning certain output is used more than once,
    // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
    // to the output list
    // The other one is, if subgraph's node output is parent graph's output. the node output should
    // be also added to the subgraph's output list
    if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        const auto& node_idx = it->GetNode().Index();
        const onnxruntime::NodeArg* output;
        if (it->GetDstArgIndex() < static_cast<int>(it->GetNode().InputDefs().size())) {
          output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];
        } else {
          auto index = it->GetDstArgIndex() - static_cast<int>(it->GetNode().InputDefs().size());
          output = (it->GetNode()).ImplicitInputDefs()[index];
        }
        if (node_set.find(node_idx) != node_set.end()) {
          const auto& iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
              graph_outputs_to_add[output] = output_order;
            }
            fused_outputs[output] = output_order++;
          }
        } else {
          fused_outputs_to_add[output] = output_order++;
        }
      }
    } else {
      for (const auto& output : node->OutputDefs()) {
        const auto& it = fused_inputs.find(output);
        if (it != fused_inputs.end()) {
          fused_inputs.erase(it);
          erased.insert(output);
        } else {
          // Only when output is neither in input list nor erased list, add the output to output list
          if (erased.find(output) == erased.end()) {
            if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
              graph_outputs_to_add[output] = output_order;
            }
            fused_outputs[output] = output_order++;
          }
        }
      }
    }
  }

  fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
  fused_outputs.insert(graph_outputs_to_add.begin(), graph_outputs_to_add.end());

  // Sort inputs and outputs by the order they were added
  std::multimap<int, const NodeArg*> inputs, outputs;
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // Generate unique kernel name for CANN subgraph
  HashValue model_hash = 0;
  int id = metadef_id_generator_->GenerateId(graph_viewer, model_hash);
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->name() = graph_viewer.Name() + "_" + std::to_string(model_hash) + "_" + std::to_string(id);

  // Assign inputs and outputs to subgraph's meta_def
  for (const auto& input : inputs) {
    if (input.second->Exists()) {
      meta_def->inputs().push_back(input.second->Name());
    }
  }

  for (const auto& initializer : initializers) {
    meta_def->constant_initializers().push_back(initializer);
  }

  for (const auto& output : outputs) {
    if (output.second->Exists()) {
      meta_def->outputs().push_back(output.second->Name());
    }
  }

  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  sub_graph->SetMetaDef(std::move(meta_def));

  return sub_graph;
}

std::vector<std::vector<NodeIndex>>
GetSubGraphPartition(const std::vector<NodeIndex>& topological_order, const std::vector<NodeIndex>& unsupported_nodes) {
  std::vector<std::vector<NodeIndex>> partitions;

  if (topological_order.size() == unsupported_nodes.size())
    return partitions;

  auto prev = topological_order.begin();
  for (const auto& node : unsupported_nodes) {
    auto next = std::find(prev, topological_order.end(), node);
    std::vector<NodeIndex> partition{prev, next};
    if (!partition.empty()) {
      partitions.push_back(std::move(partition));
    }

    prev = ++next;
  }

  std::vector<NodeIndex> partition{prev, topological_order.end()};
  if (!partition.empty()) {
    partitions.push_back(std::move(partition));
  }

  return partitions;
}

std::vector<std::unique_ptr<ComputeCapability>>
CANNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                     const IKernelLookup& kernel_lookup) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // TODO(FFFrog): Feature Enhancement
  // After the subgraph is divided, the remaining single operators should first fall back to
  // the single operator operation mode of CANN
  if (info_.enable_cann_graph) {
    std::vector<NodeIndex>&& unsupported_nodes = SupportONNXModel(graph_viewer);

    if (unsupported_nodes.empty()) {
      auto sub_graph = GetSubGraph(graph_viewer.GetNodesInTopologicalOrder(), graph_viewer);
      result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    } else {
      auto partitions = GetSubGraphPartition(graph_viewer.GetNodesInTopologicalOrder(), unsupported_nodes);

      for (const auto& partition : partitions) {
        auto sub_graph = GetSubGraph(partition, graph_viewer);
        result.push_back(ComputeCapability::Create(std::move(sub_graph)));
      }
    }
  } else {
    InlinedVector<NodeIndex> candidates;

    for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      const auto* p_node = graph_viewer.GetNode(node_index);
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

    auto cpu_nodes = GetCpuPreferredNodes(graph_viewer, kernel_lookup, candidates);
    for (auto& node_index : candidates) {
      if (cpu_nodes.count(node_index) > 0)
        continue;

      auto sub_graph = IndexedSubGraph::Create();
      sub_graph->Nodes().push_back(node_index);
      result.push_back(ComputeCapability::Create(std::move(sub_graph)));
    }
  }

  return result;
}

Status CANNExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                      std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    const std::string node_name = fused_node.Name();

    std::unordered_map<size_t, std::string> index2name;
    const auto& input_defs = fused_node.InputDefs();
    index2name.reserve(input_defs.size());
    for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
      index2name[i] = input_defs[i]->Name();
    }
    names_[node_name] = index2name;

    std::string string_model;
    auto model = cann::CreateModel(graph_body_viewer, *GetLogger());
    auto model_proto = model->ToProto();
    graph_body_viewer.ToProto(*model_proto->mutable_graph(), true, true);
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    model_proto->SerializeToString(string_model);
    models_[node_name] = string_model;

    if (info_.dump_graphs) {
      std::fstream dump(fused_node.Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
      model_proto->SerializeToOstream(dump);
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<CannFuncState> p = std::make_unique<CannFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, context->node_name};
      *state = p.release();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<CannFuncState*>(state);
    };

    compute_info.compute_func = [this](FunctionState state, const OrtApi*, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);

      CannFuncState* cann_state = reinterpret_cast<CannFuncState*>(state);
      std::string& string_model = models_[cann_state->node_name];
      std::unordered_map<size_t, std::string>& index2name = names_[cann_state->node_name];

      std::string input_shape = [&ctx, &index2name]() -> std::string {
        std::string res;
        for (size_t i = 0; i < ctx.GetInputCount(); i++) {
          auto&& shape = ctx.GetInput(i).GetTensorTypeAndShapeInfo().GetShape();

          std::string s = index2name[i] + ":";
          for (auto& d : shape) {
            s += std::to_string(d) + ",";
          }
          s[s.length() - 1] = ';';
          res += s;
        }

        return res.substr(0, res.length() - 1);
      }();

      // Since the name of the input tensor of the sub-graph may exceed the maximum length required by Linux,
      // and may also contain various special characters, such as "/". So, it is reasonable to convert it to HashValue.
      HashValue hash;
      cann::GenerateHashValue(input_shape, hash);
      std::string filename = cann_state->node_name + "_" + std::to_string(hash);
      std::string filename_with_suffix = filename + ".om";

      // TODO(FFFrog): Resource Management
      // It is very necessary to provide a new mechanism for memory reclamation to avoid inference failure caused by
      // device memory exhaustion
      uint32_t modelID;
      if (modelIDs_.find(filename) != modelIDs_.end()) {
        modelID = modelIDs_[filename];
      } else {
        std::lock_guard<OrtMutex> lock(g_mutex);

        if (cann::FileExist(filename_with_suffix)) {
          CANN_RETURN_IF_ERROR(aclmdlLoadFromFile(filename_with_suffix.c_str(), &modelID));
        } else {
          ge::Graph graph{cann_state->node_name.c_str()};
          ORT_RETURN_IF_ERROR(ParserONNXModel(string_model, graph));

          ge::ModelBufferData model;
          ORT_RETURN_IF_ERROR(BuildONNXModel(graph, input_shape, soc_name_, filename, info_, model));

          CANN_RETURN_IF_ERROR(aclmdlLoadFromMem(model.data.get(), model.length, &modelID));
        }

        modelIDs_.emplace(filename, modelID);
      }

      CannModelPreparation prepare(modelID);

      ORT_TRY {
        for (size_t i = 0; i < aclmdlGetNumInputs(prepare.modelDesc_); i++) {
          auto input = ctx.GetInput(i);
          CANN_MODEL_PREPARE_INPUTBUFFER(prepare,
                                         const_cast<void*>(input.GetTensorRawData()),
                                         aclmdlGetInputSizeByIndex(prepare.modelDesc_, i));
        }

        for (size_t i = 0; i < aclmdlGetNumOutputs(prepare.modelDesc_); i++) {
          aclmdlIODims dims;
          CANN_CALL_THROW(aclmdlGetOutputDims(prepare.modelDesc_, i, &dims));
          std::vector<int64_t> vec{dims.dims, dims.dims + dims.dimCount};
          auto output = ctx.GetOutput(i, vec);
          CANN_MODEL_PREPARE_OUTPUTBUFFER(prepare,
                                          const_cast<void*>(output.GetTensorRawData()),
                                          aclmdlGetOutputSizeByIndex(prepare.modelDesc_, i));
        }
      }
      ORT_CATCH(const std::exception& e) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
      }

      aclrtStream stream = static_cast<aclrtStream>(ctx.GetGPUComputeStream());
      CANN_RETURN_IF_ERROR(aclmdlExecuteAsync(modelID, prepare.inputSet_, prepare.outputSet_, stream));

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

AllocatorPtr CANNExecutionProvider::CreateCannAllocator(OrtDevice::DeviceId device_id, size_t npu_mem_limit,
                                                        ArenaExtendStrategy arena_extend_strategy,
                                                        OrtArenaCfg* default_memory_arena_cfg) {
  AllocatorCreationInfo default_memory_info(
      [](OrtDevice::DeviceId id) {
        return std::make_unique<CANNAllocator>(id, CANN);
      },
      device_id,
      true,
      {default_memory_arena_cfg ? *default_memory_arena_cfg
                                : OrtArenaCfg(npu_mem_limit,
                                              static_cast<int>(arena_extend_strategy),
                                              -1,
                                              -1,
                                              -1,
                                              -1L)},
      true,
      false);

  return CreateAllocator(default_memory_info);
}

std::vector<AllocatorPtr> CANNExecutionProvider::CreatePreferredAllocators() {
  AllocatorCreationInfo pinned_memory_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<CANNPinnedAllocator>(device_id, CANN_PINNED);
      },
      DEFAULT_CPU_ALLOCATOR_DEVICE_ID);

  return std::vector<AllocatorPtr>{
      CreateCannAllocator(info_.device_id, info_.npu_mem_limit, info_.arena_extend_strategy,
                          info_.default_memory_arena_cfg),
      CreateAllocator(pinned_memory_info),
  };
}

void CANNExecutionProvider::RegisterStreamHandlers(IStreamCommandHandleRegistry& stream_handle_registry, AllocatorMap&) const {
  RegisterCannStreamHandles(stream_handle_registry, OrtDevice::NPU);
}

OrtDevice CANNExecutionProvider::GetOrtDeviceByMemType(OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeCPUInput) return OrtDevice();
  if (mem_type == OrtMemTypeCPUOutput) return OrtDevice(OrtDevice::CPU, OrtDevice::MemType::CANN_PINNED, 0);
  return default_device_;
}

}  // namespace onnxruntime
