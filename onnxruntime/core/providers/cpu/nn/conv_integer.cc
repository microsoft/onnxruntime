// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

namespace onnxruntime {

class ConvInteger : public OpKernel {
 public:
  explicit ConvInteger(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

  ConvAttributes conv_attrs_;
};

ONNX_OPERATOR_KERNEL_EX(
    ConvInteger,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    ConvInteger);

Status ConvInteger::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  uint8_t input_offset = 0;
  uint8_t filter_offset = 0;
  if (num_inputs >= 3) {
    const auto* X_Zero_Point = context->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(X_Zero_Point), "Must be a scalar or 1D tensor or size 1.");
    input_offset = *(X_Zero_Point->Data<uint8_t>());
  }
  if (num_inputs >= 4) {
    const auto* W_Zero_Point = context->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(W_Zero_Point), "Non per-tensor quantization is not supported now.");
    filter_offset = *(W_Zero_Point->Data<uint8_t>());
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  const size_t kernel_rank = kernel_shape.size();

  BufferUniquePtr col_buffer;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
  }

  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  const auto* Xdata = X->template Data<uint8_t>();
  const auto* Wdata = W->template Data<uint8_t>();
  auto* Ydata = Y->template MutableData<int32_t>();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (col_buffer_data != nullptr) {
        if (kernel_rank == 2) {
          math::Im2col<uint8_t, StorageOrder::NCHW>()(
              Xdata,
              C / conv_attrs_.group,
              input_shape[0],
              input_shape[1],
              kernel_shape[0],
              kernel_shape[1],
              dilations[0],
              dilations[1],
              pads[0],
              pads[1],
              pads[2],
              pads[3],
              strides[0],
              strides[1],
              col_buffer_data,
              input_offset);
        } else {
          math::Im2col<uint8_t, StorageOrder::NCHW>()(
              Xdata,
              input_shape.GetDims().data(),
              output_shape.GetDims().data(),
              kernel_dim,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_rank),
              col_buffer_data,
              false,
              input_offset);
        }
      }

      MLAS_GEMM_U8X8_PARAMETERS gemm_params;
      gemm_params.M = static_cast<size_t>(M / conv_attrs_.group);
      gemm_params.N = static_cast<size_t>(output_image_size);
      gemm_params.K = static_cast<size_t>(kernel_dim);
      gemm_params.A = Wdata + group_id * W_offset;
      gemm_params.lda = static_cast<size_t>(kernel_dim);
      gemm_params.ZeroPointA = filter_offset;
      gemm_params.B = (col_buffer_data == nullptr) ? Xdata : col_buffer_data,
      gemm_params.ldb = static_cast<size_t>(output_image_size);
      gemm_params.ZeroPointB = &input_offset;
      gemm_params.C = Ydata;
      gemm_params.ldc = static_cast<size_t>(output_image_size);
      MlasGemm(&gemm_params, thread_pool);

      Xdata += X_offset;
      Ydata += Y_offset;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
