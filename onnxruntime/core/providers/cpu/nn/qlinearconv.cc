// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/cpuid_info.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info),
                                                   conv_attrs_(info),
                                                   is_W_signed_(false),
                                                   is_W_packed_(false) {
    channels_last_ = (info.GetAttrOrDefault<int64_t>("channels_last", static_cast<int64_t>(0)) != 0);
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, bool& /*out*/ is_packed,
                 /*out*/ PrepackedWeight* prepacked_weight_for_caching,
                 AllocatorPtr alloc) override;

  Status UseCachedPrePackedWeight(const PrepackedWeight& cached_prepacked_weight,
                                  int input_idx,
                                  /*out*/ bool& read_from_cache) override;

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
  BufferUniquePtr packed_W_buffer_;
  size_t packed_W_size_;
  BufferUniquePtr reordered_W_buffer_;
  bool is_W_signed_;
  bool is_W_packed_;
  bool channels_last_;
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

#ifndef DISABLE_CONTRIB_OPS

namespace contrib {

// Register an alternate version of this kernel that supports the channels_last
// attribute in order to consume and produce NHWC tensors.
ONNX_OPERATOR_KERNEL_EX(
    QLinearConv,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);

}  // namespace contrib

#endif

Status QLinearConv::PrePack(const Tensor& tensor, int input_idx, bool& /*out*/ is_packed,
                            /*out*/ PrepackedWeight* prepacked_weight_for_caching,
                            AllocatorPtr alloc) {
  is_packed = false;

  // Support packing the weight matrix.
  if (input_idx != 3) {
    return Status::OK();
  }

  is_W_signed_ = tensor.IsDataType<int8_t>();

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

  const auto* Wdata = static_cast<const uint8_t*>(tensor.DataRaw());
  W_shape_ = shape;

  const size_t group_count = static_cast<size_t>(conv_attrs_.group);
  const size_t group_output_channels = output_channels / group_count;
  const size_t kernel_dim = group_input_channels * kernel_size;

  bool kernel_owns_prepacked_buffer = (prepacked_weight_for_caching == nullptr);

  // Don't pack the filter buffer if the MlasConvDepthwise path is used.
  if (group_input_channels != 1 && group_output_channels != 1) {
    packed_W_size_ = MlasGemmPackBSize(group_output_channels, kernel_dim, is_W_signed_);

    if (packed_W_size_ != 0) {
      auto* packed_W = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(group_count) * packed_W_size_));
      packed_W_buffer_ = BufferUniquePtr(packed_W, BufferDeleter(alloc));

      // Allocate a temporary buffer to hold the reordered oihw->hwio filter for
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

      if (!kernel_owns_prepacked_buffer) {
        prepacked_weight_for_caching->buffers_.push_back(std::move(packed_W_buffer_));
        prepacked_weight_for_caching->weights_sizes_.push_back(packed_W_size_);
        prepacked_weight_for_caching->is_filled_ = true;
        prepacked_weight_for_caching->flags_.push_back(is_W_signed_);
        packed_W_buffer_ = BufferUniquePtr(prepacked_weight_for_caching->buffers_[0].get(), BufferDeleter(nullptr));
      }

      is_W_packed_ = true;
      is_packed = true;
      return Status::OK();
    }
  }

  auto* reordered_W = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * output_channels * group_input_channels * kernel_size));
  reordered_W_buffer_ = BufferUniquePtr(reordered_W, BufferDeleter(alloc));

  ReorderFilter(Wdata, reordered_W, output_channels, group_input_channels, kernel_size);

  if (!kernel_owns_prepacked_buffer) {
    prepacked_weight_for_caching->buffers_.push_back(std::move(reordered_W_buffer_));
    reordered_W_buffer_ = BufferUniquePtr(prepacked_weight_for_caching->buffers_[1].get(), BufferDeleter(nullptr));
  }

  is_W_packed_ = true;
  is_packed = true;
  return Status::OK();
}

Status QLinearConv::UseCachedPrePackedWeight(const PrepackedWeight& cached_prepacked_weight,
                                             int input_idx,
                                             /*out*/ bool& read_from_cache) {
  if (input_idx != 3) {
    return Status::OK();
  }

  is_W_packed_ = true;
  read_from_cache = true;

  // These are cached pre-packed buffers and this kernel doesn't own these and hence the deleter is null
  packed_W_buffer_ = BufferUniquePtr(cached_prepacked_weight.buffers_[0].get(), BufferDeleter(nullptr));
  reordered_W_buffer_ = BufferUniquePtr(cached_prepacked_weight.buffers_[1].get(), BufferDeleter(nullptr));
  packed_W_size_ = cached_prepacked_weight.weights_sizes_[0];
  is_W_signed_ = cached_prepacked_weight.flags_[0];
  return Status::OK();
}

Status QLinearConv::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = is_W_packed_ ? nullptr : context->Input<Tensor>(3);
  const auto& W_shape = W ? W->Shape() : W_shape_;
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
    W_zero_point_value = W_zero_point_data[0];
    for (int64_t i = 1; i < W_zero_point_size; i++) {
      if (W_zero_point_data[i] != W_zero_point_value) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv : filter zero point must be constant");
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

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W_shape, channels_last_));

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

  const int64_t C = X->Shape()[channels_last_ ? 1 + kernel_rank : 1];
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;
  const size_t spatial_dim_end = spatial_dim_start + kernel_rank;

  std::vector<int64_t> Y_dims({N});
  if (!channels_last_) {
    Y_dims.push_back(M);
  }
  TensorShape input_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  if (channels_last_) {
    Y_dims.push_back(M);
  }
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(spatial_dim_start, spatial_dim_end);

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
  if (!packed_W_buffer_) {
    if (W == nullptr) {
      // Weight was constant and reordered.
      reordered_W = static_cast<uint8_t*>(reordered_W_buffer_.get());
    } else {
      // Weight tensor was not constant or prepacking is disabled.
      reordered_W = static_cast<uint8_t*>(alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * W_shape.Size()));
      reordered_W_buffer = BufferUniquePtr(reordered_W, BufferDeleter(alloc));
      ReorderFilter(
          static_cast<const uint8_t*>(W->DataRaw()),
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
  const bool is_depthwise_conv = (reordered_W != nullptr && group_input_channels == 1 && group_output_channels == 1);
  if (is_depthwise_conv) {
    // Update the input and output channels to the number of groups in order to
    // reuse as much of the below standard convolution path.
    group_input_channels = group_count;
    group_output_channels = group_count;
    group_count = 1;
  }

  const int64_t X_offset = C * input_image_size;
  const int64_t Y_offset = M * output_image_size;
  const int64_t kernel_dim = group_input_channels * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  // Use an intermediate int32_t buffer for the GEMM computation before
  // requantizing to the output type.
  auto* gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) * Y_offset);
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());

  const auto* Xdata = X->template Data<uint8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<uint8_t>();

  BufferUniquePtr transpose_input_buffer;
  BufferUniquePtr transpose_output_buffer;

  // Allocate temporary buffers for transposing to channels last format.
  if (!channels_last_) {
    auto* transpose_input = alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * X_offset);
    transpose_input_buffer = BufferUniquePtr(transpose_input, BufferDeleter(alloc));
    auto* transpose_output = alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * Y_offset);
    transpose_output_buffer = BufferUniquePtr(transpose_output, BufferDeleter(alloc));
  }

  BufferUniquePtr col_buffer;
  std::vector<uint8_t> padding_data;

  if (is_depthwise_conv) {
    // Allocate indirection buffer pointers and prepare a padding vector for
    // the im2col transform.
    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(const uint8_t*)) * kernel_size * output_image_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
    padding_data.resize(static_cast<size_t>(C), X_zero_point_value);
  } else if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    // Pointwise convolutions can use the original input tensor in place,
    // otherwise a temporary buffer is required for the im2col transform.
    int64_t group_col_buffer_size = (kernel_rank > 2) ? group_count * col_buffer_size : col_buffer_size;
    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * group_col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
  }

  // Replicate the logic from MlasGemmU8X8Schedule to control the number of
  // worker threads used for the convolution.
  int32_t maximum_thread_count;
  if (CPUIDInfo::GetCPUIDInfo().IsHybrid()) {
    maximum_thread_count = 64;
  } else {
    maximum_thread_count = 16;
  }
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
    const auto* input_data = Xdata;
    auto* output_data = Ydata;

    if (!channels_last_) {
      // Transpose the input from channels first (NCHW) to channels last (NHWC).
      MlasTranspose(
          Xdata,
          static_cast<uint8_t*>(transpose_input_buffer.get()),
          static_cast<size_t>(C),
          static_cast<size_t>(input_image_size));
      input_data = static_cast<uint8_t*>(transpose_input_buffer.get());
      output_data = static_cast<uint8_t*>(transpose_output_buffer.get());
    }

    // Threaded implementation of ND convolution is not yet supported, so
    // prepare all im2col transformations here.
    if (!is_depthwise_conv && col_buffer && kernel_rank > 2) {
      for (int64_t group_id = 0; group_id < group_count; ++group_id) {
        math::Im2col<uint8_t, StorageOrder::NHWC>()(
            input_data + group_id * group_input_channels,
            group_input_channels,
            C,
            input_shape.GetDims().data(),
            output_shape.GetDims().data(),
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<int64_t>(kernel_rank),
            static_cast<uint8_t*>(col_buffer.get()) + group_id * col_buffer_size,
            X_zero_point_value);
      }
    }

    auto conv_worker = [&](ptrdiff_t batch) {
      auto work = concurrency::ThreadPool::PartitionWork(batch, thread_count, static_cast<ptrdiff_t>(output_image_size));
      int64_t output_start = static_cast<int64_t>(work.start);
      int64_t output_count = static_cast<int64_t>(work.end - work.start);

      auto* worker_gemm_output = gemm_output + output_start * M;
      auto* worker_requantize_output = output_data + output_start * M;

      if (is_depthwise_conv) {
        auto* worker_col_buffer = static_cast<uint8_t const**>(col_buffer.get()) + output_start * kernel_size;
        math::Im2col<uint8_t, StorageOrder::NHWC>()(
            input_data,
            C,
            input_shape.GetDims().data(),
            output_shape.GetDims().data(),
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<ptrdiff_t>(kernel_rank),
            output_start,
            output_count,
            worker_col_buffer,
            padding_data.data());
        MlasConvDepthwise(
            worker_col_buffer,
            X_zero_point_value,
            reordered_W,
            W_zero_point_value,
            is_W_signed,
            worker_gemm_output,
            static_cast<size_t>(M),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      } else {
        for (int64_t group_id = 0; group_id < group_count; ++group_id) {
          // Prepare the im2col transformation or use the input buffer directly for
          // pointwise convolutions.
          const uint8_t* worker_gemm_input;
          if (col_buffer) {
            auto* worker_col_buffer = static_cast<uint8_t*>(col_buffer.get()) + output_start * kernel_dim;
            if (kernel_rank == 2) {
              math::Im2col<uint8_t, StorageOrder::NHWC>()(
                  input_data + group_id * group_input_channels,
                  group_input_channels,
                  C,
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
                  worker_col_buffer,
                  X_zero_point_value);
            } else if (kernel_rank == 1) {
              math::Im2col<uint8_t, StorageOrder::NHWC>()(
                  input_data + group_id * group_input_channels,
                  group_input_channels,
                  C,
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
                  worker_col_buffer,
                  X_zero_point_value);
            } else {
              // Use the im2col buffer prepared outside the thread, indexed by group.
              worker_col_buffer += group_id * col_buffer_size;
            }
            worker_gemm_input = worker_col_buffer;
          } else {
            worker_gemm_input = input_data + output_start * kernel_dim;
          }

          MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;
          gemm_shape.M = static_cast<size_t>(output_count);
          gemm_shape.N = static_cast<size_t>(group_output_channels);
          gemm_shape.K = static_cast<size_t>(kernel_dim);
          gemm_shape.BIsSigned = is_W_signed;

          MLAS_GEMM_U8X8_DATA_PARAMS gemm_params;
          gemm_params.A = worker_gemm_input;
          gemm_params.lda = static_cast<size_t>(kernel_dim);
          gemm_params.ZeroPointA = X_zero_point_value;
          if (packed_W_buffer_) {
            gemm_params.B = static_cast<const int8_t*>(packed_W_buffer_.get()) + group_id * packed_W_size_,
            gemm_params.BIsPacked = true;
          } else {
            gemm_params.B = reordered_W + group_id * group_output_channels,
            gemm_params.ldb = static_cast<size_t>(M);
          }
          gemm_params.ZeroPointB = &W_zero_point_value;
          gemm_params.C = worker_gemm_output + group_id * group_output_channels;
          gemm_params.ldc = static_cast<size_t>(M);
          MlasGemm(gemm_shape, gemm_params, nullptr);
        }
      }

      MlasRequantizeOutput(
          worker_gemm_output,
          worker_requantize_output,
          Bdata,
          static_cast<size_t>(output_count),
          static_cast<size_t>(M),
          output_scales.data(),
          output_scales.size() > 1,
          Y_zero_point_value);
    };

    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, thread_count, conv_worker);

    if (!channels_last_) {
      // Transpose the output from channels last (NHWC) to channels first (NCHW).
      MlasTranspose(
          output_data,
          Ydata,
          static_cast<size_t>(output_image_size),
          static_cast<size_t>(M));
    }

    Xdata += X_offset;
    Ydata += Y_offset;
  }

  return Status::OK();
}

}  // namespace onnxruntime
