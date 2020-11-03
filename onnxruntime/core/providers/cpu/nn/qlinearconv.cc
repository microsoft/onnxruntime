// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/util/gemmlowp_common.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

#if defined(MLAS_TARGET_AMD64_IX86)

class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info),
                                                   conv_attrs_(info),
                                                   is_W_signed_(false),
                                                   is_W_packed_(false) {
  }

  Status Compute(OpKernelContext* context) const override;
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;

 private:
  static void ReorderFilter(const uint8_t* input,
                            uint8_t* output,
                            size_t output_channels,
                            size_t input_channels,
                            size_t kernel_size) {
    for (size_t k = 0; k < kernel_size; k++) {
      for (size_t ic = 0; ic < input_channels; ic++) {
        for (size_t oc = 0; oc < output_channels; oc++) {
          size_t index = (oc * input_channels * kernel_size) + (ic * kernel_size) + k;
          *output++ = input[index];
        }
      }
    }
  }

  ConvAttributes conv_attrs_;
  TensorShape W_shape_;
#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
  BufferUniquePtr packed_W_buffer_;
  size_t packed_W_size_;
#endif
  BufferUniquePtr reordered_W_buffer_;
  bool is_W_signed_;
  bool is_W_packed_;
};

ONNX_CPU_OPERATOR_KERNEL(
    QLinearConv,
    10,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);

Status QLinearConv::PrePack(const Tensor& tensor, int input_idx, bool& is_packed) {
  is_packed = false;

  // Support packing the weight matrix.
  if (input_idx != 3) {
    return Status::OK();
  }

  const auto& shape = tensor.Shape().GetDims();
  size_t rank = shape.size();
  if (rank <= 2) {
    return Status::OK();
  }

  if (shape[0] % conv_attrs_.group != 0) {
    return Status::OK();
  }

  // Note: The tensor has already been allocated with this tensor shape, so all
  // shape indices are guaranteed to fit inside size_t.
  const size_t output_channels = static_cast<size_t>(shape[0]);
  const size_t group_input_channels = static_cast<size_t>(shape[1]);
  const size_t kernel_size =
    static_cast<size_t>(std::accumulate(shape.data() + 2, shape.data() + rank, 1LL, std::multiplies<int64_t>()));

  const size_t group_count = static_cast<size_t>(conv_attrs_.group);
  const size_t group_output_channels = output_channels / group_count;
  const size_t kernel_dim = group_input_channels * kernel_size;

  const auto* Wdata = static_cast<const uint8_t*>(tensor.DataRaw());
  W_shape_ = shape;
  is_W_signed_ = tensor.IsDataType<int8_t>();

  auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);

#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
  // Don't pack the filter buffer if the MlasConvDepthwise path is used.
  if (group_input_channels != 1 && group_output_channels != 1) {
    packed_W_size_ = MlasGemmPackBSize(group_output_channels, kernel_dim, true);

    if (packed_W_size_ != 0) {
      auto* packed_W = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(group_count) * packed_W_size_));
      packed_W_buffer_ = BufferUniquePtr(packed_W, BufferDeleter(alloc));

      // Allocate a temporary buffer to hold the reordered oihw->ohwi filter for
      // a single group.
      //
      // Note: The size of this buffer is less than or equal to the size of the original
      // weight tensor, so the allocation size is guaranteed to fit inside size_t.
      auto* group_reordered_W = static_cast<uint8_t*>(alloc->Alloc(group_output_channels * group_input_channels * kernel_size));
      BufferUniquePtr group_reordered_W_buffer(group_reordered_W, BufferDeleter(alloc));

      const size_t W_offset = group_output_channels * kernel_dim;

      for (int64_t group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        ReorderFilter(Wdata, group_reordered_W, group_output_channels, group_input_channels, kernel_size);
        MlasGemmPackB(group_output_channels, kernel_dim, group_reordered_W, group_output_channels, is_W_signed_, packed_W);
        packed_W += packed_W_size_;
        Wdata += W_offset;
      }

      is_W_packed_ = true;
      is_packed = true;
      return Status::OK();
    }
  }
#endif

  auto* reordered_W = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * output_channels * group_input_channels * kernel_size));
  reordered_W_buffer_ = BufferUniquePtr(reordered_W, BufferDeleter(alloc));

  ReorderFilter(Wdata, reordered_W, output_channels, group_input_channels, kernel_size);

  is_W_packed_ = true;
  is_packed = true;
  return Status::OK();
}

Status QLinearConv::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = is_W_packed_ ? nullptr : context->Input<Tensor>(3);
  const auto& W_shape = is_W_packed_ ? W_shape_ : W->Shape();
  const bool is_W_signed = (W != nullptr) ? W->IsDataType<int8_t>() : is_W_signed_;

  const int64_t N = X->Shape()[0];
  const int64_t M = W_shape[0];

  // validate offsets
  const Tensor* X_zero_point = context->Input<Tensor>(2);
  const Tensor* W_zero_point = context->Input<Tensor>(5);
  const Tensor* Y_zero_point = context->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_zero_point),
              "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_zero_point),
              "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");

  auto X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
  auto Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());

  uint8_t W_zero_point_value;
  const auto& W_zero_point_shape = W_zero_point->Shape();
  if (W_zero_point_shape.NumDimensions() == 0 ||
      (W_zero_point_shape.NumDimensions() == 1 && (W_zero_point_shape[0] == 1 || W_zero_point_shape[0] == M))) {
    const int64_t W_zero_point_size = W_zero_point_shape.Size();
    const auto* W_zero_point_data = static_cast<const uint8_t*>(W_zero_point->DataRaw());
    if (is_W_signed) {
      W_zero_point_value = 0;
      for (int64_t i = 0; i < W_zero_point_size; i++) {
        if (W_zero_point_data[i] != 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv : filter zero point must be zero");
        }
      }
    } else {
      W_zero_point_value = W_zero_point_data[0];
      for (int64_t i = 1; i < W_zero_point_size; i++) {
        if (W_zero_point_data[i] != W_zero_point_value) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv : filter zero point must be constant");
        }
      }
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv : filter zero point shape invalid");
  }

  // validate scale
  const Tensor* X_scale = context->Input<Tensor>(1);
  const Tensor* W_scale = context->Input<Tensor>(4);
  const Tensor* Y_scale = context->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
              "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
              "QLinearConv : result scale must be a scalar or 1D tensor of size 1");

  auto X_scale_value = *(X_scale->template Data<float>());
  auto Y_scale_value = *(Y_scale->template Data<float>());

  std::vector<float> output_scales;
  const auto& W_scale_shape = W_scale->Shape();
  if (W_scale_shape.NumDimensions() == 0 ||
      (W_scale_shape.NumDimensions() == 1 && (W_scale_shape[0] == 1 || W_scale_shape[0] == M))) {
    const int64_t W_scale_size = W_scale_shape.Size();
    const auto* W_scale_data = W_scale->template Data<float>();
    output_scales.resize(static_cast<size_t>(W_scale_size));
    for (int64_t i = 0; i < W_scale_size; i++) {
      output_scales[i] = (X_scale_value * W_scale_data[i] / Y_scale_value);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv : filter scale shape invalid");
  }

  const Tensor* B = context->Input<Tensor>(8);

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W_shape));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));

  const size_t kernel_rank = kernel_shape.size();

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
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

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  // Handle the case of a dynamic weight filter.
  BufferUniquePtr reordered_W_buffer;
  uint8_t* reordered_W = nullptr;
  bool use_reordered_W = true;
#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
  if (packed_W_buffer_) {
    use_reordered_W = false;
  }
#endif
  if (use_reordered_W) {
    if (reordered_W_buffer_) {
      reordered_W = static_cast<uint8_t*>(reordered_W_buffer_.get());
    } else {
      // Weight tensor was not constant or prepacking is disabled.
      reordered_W = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * W_shape.Size()));
      reordered_W_buffer = BufferUniquePtr(reordered_W, BufferDeleter(alloc));
      ReorderFilter(static_cast<const uint8_t*>(W->DataRaw()),
                    reordered_W,
                    static_cast<size_t>(M),
                    static_cast<size_t>(W_shape[1]),
                    static_cast<size_t>(kernel_size));
    }
  }

  int64_t group_count = conv_attrs_.group;
  int64_t group_input_channels = W_shape[1];
  int64_t group_output_channels = M / group_count;

  // Test for depthwise convolution.
  const bool is_depthwise_conv = (use_reordered_W && group_input_channels == 1 && group_output_channels == 1);
  if (is_depthwise_conv) {
    // Update the input and output channels to the number of groups in order to
    // reuse as much of the below standard convolution path.
    group_input_channels = group_count;
    group_output_channels = group_count;
    group_count = 1;
  }

  const int64_t X_offset = group_input_channels * input_image_size;
  const int64_t Y_offset = group_output_channels * output_image_size;
  const int64_t kernel_dim = group_input_channels * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  // Use an intermediate int32_t buffer for the GEMM computation before
  // requantizing to the output type.
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) * Y_offset);
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());

  const auto* Xdata = X->template Data<uint8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<uint8_t>();

  auto* transpose_input = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * X_offset));
  BufferUniquePtr transpose_input_buffer(transpose_input, BufferDeleter(alloc));

  auto* transpose_output = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * Y_offset));
  BufferUniquePtr transpose_output_buffer(transpose_output, BufferDeleter(alloc));

  BufferUniquePtr col_buffer;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
  }
  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  // Replicate the logic from MlasGemmU8X8Schedule to control the number of
  // worker threads used for the convolution.
  constexpr int32_t maximum_thread_count = 16;
  constexpr double thread_complexity = static_cast<double>(64 * 1024);

  const double complexity = static_cast<double>(output_image_size) *
                            static_cast<double>(group_output_channels) *
                            static_cast<double>(kernel_dim);

  int32_t thread_count = maximum_thread_count;
  if (complexity < thread_complexity * maximum_thread_count) {
    thread_count = static_cast<int32_t>(complexity / thread_complexity) + 1;
  }
  if (thread_count > output_image_size) {
    // Ensure that every thread produces at least one output.
    thread_count = static_cast<int32_t>(output_image_size);
  }

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  thread_count = std::min(thread_count, concurrency::ThreadPool::DegreeOfParallelism(thread_pool));

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    for (int64_t group_id = 0; group_id < group_count; ++group_id) {
      // Transpose the input from channels first (NCHW) to channels last (NHWC).
      MlasTranspose(Xdata,
                    transpose_input,
                    static_cast<size_t>(group_input_channels),
                    static_cast<size_t>(input_image_size));

      if (col_buffer_data != nullptr) {
        if (kernel_rank > 2) {
          math::Im2colNd<uint8_t, StorageOrder::NHWC>()(
              transpose_input,
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
              X_zero_point_value);
        }
      }

      auto conv_worker = [&](ptrdiff_t batch) {
        auto work = concurrency::ThreadPool::PartitionWork(batch, thread_count, static_cast<ptrdiff_t>(output_image_size));
        int64_t output_start = static_cast<int64_t>(work.start);
        int64_t output_count = static_cast<int64_t>(work.end - work.start);

        // Prepare the im2col transformation or use the input buffer directly for
        // pointwise convolutions.
        uint8_t* worker_gemm_input;
        if (col_buffer_data != nullptr) {
          worker_gemm_input = col_buffer_data + output_start * kernel_dim;
          if (kernel_rank == 2) {
            math::Im2col<uint8_t, StorageOrder::NHWC>()(
                transpose_input,
                group_input_channels,
                input_shape[0],
                input_shape[1],
                kernel_shape[0],
                kernel_shape[1],
                dilations[0],
                dilations[1],
                pads[0],
                pads[1],
                strides[0],
                strides[1],
                output_shape[1],
                output_start,
                output_count,
                worker_gemm_input,
                X_zero_point_value);
          } else if (kernel_rank == 1) {
            math::Im2col<uint8_t, StorageOrder::NHWC>()(
                transpose_input,
                group_input_channels,
                1,
                input_shape[0],
                1,
                kernel_shape[0],
                1,
                dilations[0],
                0,
                pads[0],
                1,
                strides[0],
                output_shape[0],
                output_start,
                output_count,
                worker_gemm_input,
                X_zero_point_value);
          }
        } else {
          worker_gemm_input = transpose_input + output_start * kernel_dim;
        }

        auto* worker_gemm_output = gemm_output + output_start * group_output_channels;
        auto* worker_transpose_output = transpose_output + output_start * group_output_channels;

        if (is_depthwise_conv) {
          if (is_W_signed) {
            MlasConvDepthwise(worker_gemm_input,
                              X_zero_point_value,
                              reinterpret_cast<int8_t*>(reordered_W),
                              static_cast<int8_t>(W_zero_point_value),
                              worker_gemm_output,
                              static_cast<size_t>(group_output_channels),
                              static_cast<size_t>(output_count),
                              static_cast<size_t>(kernel_size));
          } else {
            MlasConvDepthwise(worker_gemm_input,
                              X_zero_point_value,
                              reordered_W,
                              W_zero_point_value,
                              worker_gemm_output,
                              static_cast<size_t>(group_output_channels),
                              static_cast<size_t>(output_count),
                              static_cast<size_t>(kernel_size));
          }
        } else {
#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
          if (packed_W_buffer_) {
            MlasGemm(static_cast<size_t>(output_count),
                     static_cast<size_t>(group_output_channels),
                     static_cast<size_t>(kernel_dim),
                     worker_gemm_input,
                     static_cast<size_t>(kernel_dim),
                     X_zero_point_value,
                     static_cast<const int8_t*>(packed_W_buffer_.get()) + group_id * packed_W_size_,
                     W_zero_point_value,
                     is_W_signed,
                     worker_gemm_output,
                     static_cast<size_t>(group_output_channels),
                     nullptr);
          } else
#endif
          {
            MlasGemm(static_cast<size_t>(output_count),
                     static_cast<size_t>(group_output_channels),
                     static_cast<size_t>(kernel_dim),
                     worker_gemm_input,
                     static_cast<size_t>(kernel_dim),
                     X_zero_point_value,
                     reordered_W + group_id * group_output_channels,
                     static_cast<size_t>(M),
                     W_zero_point_value,
                     is_W_signed,
                     worker_gemm_output,
                     static_cast<size_t>(group_output_channels),
                     nullptr);
          }
        }

        if (output_scales.size() == 1) {
          MlasRequantizeOutputColumn(worker_gemm_output,
                                     worker_transpose_output,
                                     Bdata != nullptr ? Bdata + group_id * group_output_channels : nullptr,
                                     static_cast<size_t>(output_count),
                                     static_cast<size_t>(group_output_channels),
                                     output_scales[0],
                                     Y_zero_point_value);
        } else {
          MlasRequantizeOutputColumn(worker_gemm_output,
                                     worker_transpose_output,
                                     Bdata != nullptr ? Bdata + group_id * group_output_channels : nullptr,
                                     static_cast<size_t>(output_count),
                                     static_cast<size_t>(group_output_channels),
                                     output_scales.data() + group_id * group_output_channels,
                                     Y_zero_point_value);
        }
      };

      concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, thread_count, conv_worker);

      // Transpose the output from channels last (NHWC) to channels first (NCHW).
      MlasTranspose(transpose_output,
                    Ydata,
                    static_cast<size_t>(output_image_size),
                    static_cast<size_t>(group_output_channels));

      Xdata += X_offset;
      Ydata += Y_offset;
    }
  }

  return Status::OK();
}

#else

class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

 protected:
  ConvAttributes conv_attrs_;
};

ONNX_CPU_OPERATOR_KERNEL(
    QLinearConv,
    10,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);

Status QLinearConv::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(3);

  // validate offsets
  const Tensor* X_zero_point = context->Input<Tensor>(2);
  const Tensor* W_zero_point = context->Input<Tensor>(5);
  const Tensor* Y_zero_point = context->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_zero_point),
              "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_zero_point),
              "QLinearConv : filter zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_zero_point),
              "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");

  auto X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
  auto W_zero_point_value = *(W_zero_point->template Data<uint8_t>());
  auto Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());

  // validate scale
  const Tensor* X_scale = context->Input<Tensor>(1);
  const Tensor* W_scale = context->Input<Tensor>(4);
  const Tensor* Y_scale = context->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
              "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_scale),
              "QLinearConv : filter scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
              "QLinearConv : result scale must be a scalar or 1D tensor of size 1");

  auto X_scale_value = *(X_scale->template Data<float>());
  auto W_scale_value = *(W_scale->template Data<float>());
  auto Y_scale_value = *(Y_scale->template Data<float>());

  const Tensor* B = context->Input<Tensor>(8);

  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

  const size_t kernel_rank = kernel_shape.size();

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
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

  const int64_t group_input_channels = W->Shape()[1];
  const int64_t group_output_channels = M / conv_attrs_.group;

  const int64_t X_offset = group_input_channels * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t kernel_dim = group_input_channels * kernel_size;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr col_buffer;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
  }

  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  const float real_multiplier = (X_scale_value * W_scale_value) / Y_scale_value;

#ifdef MLAS_SUPPORTS_GEMM_U8X8_AND_REQUANTIZE_OUTPUT
  // Use an intermediate int32_t buffer for the GEMM computation before
  // requantizing to the output type.
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) * Y_offset);
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());
#else
  // Compute the fixed point multiplier and shift for requantizing with GEMMLOWP.
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);
#endif

  const auto* Xdata = X->template Data<uint8_t>();
  const auto* Wdata = W->template Data<uint8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<uint8_t>();

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    for (int64_t group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (col_buffer_data != nullptr) {
        if (kernel_rank == 2) {
          math::Im2col<uint8_t, StorageOrder::NCHW>()(
              Xdata,
              group_input_channels,
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
              X_zero_point_value);
        } else {
          math::Im2colNd<uint8_t, StorageOrder::NCHW>()(
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
              X_zero_point_value);
        }
      }

#ifdef MLAS_SUPPORTS_GEMM_U8X8_AND_REQUANTIZE_OUTPUT
      QGemm(static_cast<int>(group_output_channels),
            static_cast<int>(output_image_size),
            static_cast<int>(kernel_dim),
            Wdata + group_id * W_offset,
            static_cast<int>(kernel_dim),
            W_zero_point_value,
            col_buffer_data == nullptr ? Xdata : col_buffer_data,
            static_cast<int>(output_image_size),
            X_zero_point_value,
            false,
            gemm_output,
            static_cast<int>(output_image_size),
            context->GetOperatorThreadPool());

      MlasRequantizeOutput(gemm_output,
                           Ydata,
                           Bdata != nullptr ? Bdata + group_id * group_output_channels : nullptr,
                           static_cast<size_t>(group_output_channels),
                           static_cast<size_t>(output_image_size),
                           real_multiplier,
                           Y_zero_point_value);
#else
      GemmlowpMultiplyu8u8_u8(Wdata + group_id * W_offset,
                              col_buffer_data == nullptr ? Xdata : col_buffer_data,
                              Ydata,
                              W_zero_point_value,
                              X_zero_point_value,
                              Y_zero_point_value,
                              static_cast<int>(group_output_channels),
                              static_cast<int>(output_image_size),
                              static_cast<int>(kernel_dim),
                              integer_multiplier,
                              right_shift,
                              Bdata != nullptr ? Bdata + group_id * group_output_channels : nullptr);
#endif

      Xdata += X_offset;
      Ydata += Y_offset;
    }
  }

  return Status::OK();
}

#endif

}  // namespace onnxruntime
