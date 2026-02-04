// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

namespace onnxruntime {

using ConvPadVector = ConvAttributes::ConvPadVector;

class ConvInteger : public OpKernel {
 public:
  explicit ConvInteger(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
  }

  Status Compute(OpKernelContext* context) const override;

  ConvAttributes conv_attrs_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

ONNX_OPERATOR_KERNEL_EX(
    ConvInteger,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(),
                               DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(),
                               DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    ConvInteger);

Status ConvInteger::Compute(OpKernelContext* context) const {
  const auto input_defs = Node().InputDefs();
  size_t num_inputs = input_defs.size();
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  uint8_t input_offset = 0;
  uint8_t filter_offset = 0;
  if (num_inputs >= 3 && input_defs[2]->Exists()) {
    const auto* X_Zero_Point = context->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(X_Zero_Point), "Must be a scalar or 1D tensor or size 1.");
    input_offset = *static_cast<const uint8_t*>(X_Zero_Point->DataRaw());
  }
  if (num_inputs >= 4 && input_defs[3]->Exists()) {
    const auto* W_Zero_Point = context->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(W_Zero_Point), "Non per-tensor quantization is not supported now.");
    filter_offset = *static_cast<const uint8_t*>(W_Zero_Point->DataRaw());
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  TensorShapeVector Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
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
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(std::move(alloc)));
  }

  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  const auto* Xdata = static_cast<const uint8_t*>(X->DataRaw());
  const auto* Wdata = static_cast<const uint8_t*>(W->DataRaw());
  bool X_is_signed = X->IsDataType<int8_t>();
  auto* Ydata = Y->MutableData<int32_t>();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (col_buffer_data != nullptr) {
        if (kernel_rank == 1 && !X_is_signed) {
          // Fast-path 1D im2col for unsigned input, NCHW layout, group=1.
          // Falls back to the generic implementation for grouped convolutions.
          if (conv_attrs_.group == 1) {
            const int64_t C_per_group = C / conv_attrs_.group;
            const int64_t in_w = input_shape[0];
            const int64_t out_w = output_shape[0];
            const int64_t k_w = static_cast<int64_t>(kernel_shape[0]);
            const int64_t dilation_w = static_cast<int64_t>(dilations[0]);
            const int64_t stride_w = static_cast<int64_t>(strides[0]);
            const int64_t pad_left = static_cast<int64_t>(pads[0]);

            uint8_t* col = col_buffer_data;

            // For each channel and kernel position, write one row of length out_w.
            // Row index: c*k_w + kw
            for (int64_t c = 0; c < C_per_group; ++c) {
              const uint8_t* x_c = Xdata + c * in_w;

              for (int64_t kw = 0; kw < k_w; ++kw) {
                const int64_t base_x = kw * dilation_w - pad_left;
                uint8_t* dst_row = col + (c * k_w + kw) * out_w;

                for (int64_t ow = 0; ow < out_w; ++ow) {
                  const int64_t x = base_x + ow * stride_w;
                  dst_row[ow] = (static_cast<uint64_t>(x) < static_cast<uint64_t>(in_w)) ? x_c[x] : input_offset;
                }
              }
            }
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
        } else if (kernel_rank == 2) {
          if (X_is_signed) {
            math::Im2col<int8_t, StorageOrder::NCHW>()(
                reinterpret_cast<const int8_t*>(Xdata),
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
                reinterpret_cast<int8_t*>(col_buffer_data),
                static_cast<int8_t>(input_offset));
          } else {
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
          }
        } else {
          if (X_is_signed) {
            math::Im2col<int8_t, StorageOrder::NCHW>()(
                reinterpret_cast<const int8_t*>(Xdata),
                input_shape.GetDims().data(),
                output_shape.GetDims().data(),
                kernel_dim,
                kernel_shape.data(),
                strides.data(),
                dilations.data(),
                pads.data(),
                static_cast<int>(kernel_rank),
                reinterpret_cast<int8_t*>(col_buffer_data),
                false,
                static_cast<int8_t>(input_offset));
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
      }

      MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
      gemm_shape.M = static_cast<size_t>(M / conv_attrs_.group);
      gemm_shape.N = static_cast<size_t>(output_image_size);
      gemm_shape.K = static_cast<size_t>(kernel_dim);
      gemm_shape.AIsSigned = W->IsDataType<int8_t>();
      gemm_shape.BIsSigned = X_is_signed;

      MLAS_GEMM_QUANT_DATA_PARAMS gemm_params;
      gemm_params.A = Wdata + group_id * W_offset;
      gemm_params.lda = static_cast<size_t>(kernel_dim);
      gemm_params.ZeroPointA = filter_offset;
      gemm_params.B = (col_buffer_data == nullptr) ? Xdata : col_buffer_data;
      gemm_params.ldb = static_cast<size_t>(output_image_size);
      gemm_params.ZeroPointB = &input_offset;
      gemm_params.C = Ydata;
      gemm_params.ldc = static_cast<size_t>(output_image_size);

      MlasGemm(gemm_shape, gemm_params, thread_pool, &mlas_backend_kernel_selector_config_);

      Xdata += X_offset;
      Ydata += Y_offset;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
