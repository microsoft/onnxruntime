// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This file contains implementation of a fp16 convolution operator.
//

#include "core/mlas/inc/mlas.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

#include "core/common/safeint.h"
#include "core/framework/float16.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {

using ConvPadVector = ConvAttributes::ConvPadVector;

/**
 * @brief Convolution Operator for FP16 tensors
 *
 * With two optional fused operations:
 *
 * 1. Add
 * It takes an extra (optional) input Sum, a tensor same shape as the output.
 * Sum is added to the output tensor.
 *
 * 2. Activation
 * It takes an operator attribute 'activation', which supplies the activation info.
 *
 * Add is performed BEFORE activation.
 *
 * The implementation supports both NCHW and NHWC. It runs faster with NHWC.
 *
 * Currently this class implement 3 operators: onnx.Conv, ms.FusedConv and ms.NhwcFusedConv
 * In the constructor, if we see the operator name is NhwcFusedConv, we assume the
 * input layout to be NHWC, otherwise we assume layout is NCHW.
 *
 */
class FusedConvFp16 final : public OpKernel {
 public:
  FusedConvFp16(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
    channels_last_ = (info.GetKernelDef().OpName() == "NhwcFusedConv");
  }

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 protected:
  bool channels_last_{false};

 private:
  /**
   * @brief Reorder filter data to facilitate compute.
   *
   *        Based on Conv operator spec, filters are organized as (M x C/group x kH x kW),
   *        where C is the number of input channels, and kH and kW are the height and width
   *        of the kernel, and M is the number of feature maps. We need to change it into
   *        (kH x kW x C/group) x M, forming a matrix of M columns, where each kernel is a
   *        single column in channel last format.
   *
   * @param input
   * @param output
   * @param output_channels  number of feature maps
   * @param input_channels
   * @param kernel_size      kH x kW
   */
  static void ReorderFilter(const MLFloat16* input,
                            MLFloat16* output,
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

  MLAS_ACTIVATION activation_;
  ConvAttributes conv_attrs_;
  TensorShape W_shape_;
  BufferUniquePtr packed_W_buffer_;
  size_t packed_W_size_{0};
  bool is_W_packed_{false};
  BufferUniquePtr reordered_W_buffer_;
};

Status FusedConvFp16::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                              /*out*/ bool& is_packed,
                              /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;
  if (input_idx != 1) {
    // Only pack filter tensor (aka weights)
    return Status::OK();
  }

  const auto& shape = tensor.Shape().GetDims();
  size_t rank = shape.size();
  if (rank <= 2) {
    return Status::OK();
  }

  const int64_t M = shape[0];
  const int64_t C = shape[1];

  // Verify that the total number of output channels is a multiple of the group count.
  if (M % conv_attrs_.group != 0) {
    return Status::OK();
  }

  // Note: The tensor has already been allocated with this tensor shape, so all
  // shape indices are guaranteed to fit inside size_t.
  const size_t output_channels = static_cast<size_t>(M);
  const size_t group_input_channels = static_cast<size_t>(C);
  const size_t kernel_size =
      static_cast<size_t>(std::accumulate(shape.data() + 2, shape.data() + rank, 1LL, std::multiplies<int64_t>()));

  const auto* Wdata = static_cast<const MLFloat16*>(tensor.DataRaw());
  W_shape_ = shape;

  const size_t group_count = static_cast<size_t>(conv_attrs_.group);
  const size_t group_output_channels = output_channels / group_count;
  const size_t kernel_dim = group_input_channels * kernel_size;

  bool share_prepacked_weights = (prepacked_weights != nullptr);

  const bool is_depthwise_conv = (group_input_channels == 1 && group_output_channels == 1);
  // Don't pack the filter buffer if the MlasConvDepthwise path is used.
  if (!is_depthwise_conv) {
    packed_W_size_ = MlasHalfGemmPackBSize(group_output_channels, kernel_dim, false);
    if (packed_W_size_ != 0) {
      size_t packed_W_data_size = SafeInt<size_t>(group_count) * packed_W_size_;
      auto* packed_W = static_cast<MLFloat16*>(alloc->Alloc(packed_W_data_size));

      // Initialize memory to 0 as there could be some padding associated with pre-packed
      // buffer memory and we don not want it uninitialized and generate different hashes
      // if and when we try to cache this pre-packed buffer for sharing between sessions.
      memset((void*)packed_W, 0, packed_W_data_size);

      packed_W_buffer_ = BufferUniquePtr(packed_W, BufferDeleter(alloc));

      // Allocate a temporary buffer to hold the reordered oihw->hwio filter for
      // a single group.
      //
      // Note: The size of this buffer is less than or equal to the size of the original
      // weight tensor, so the allocation size is guaranteed to fit inside size_t.
      auto* group_reordered_W = static_cast<MLFloat16*>(
          alloc->Alloc(group_output_channels * kernel_dim * sizeof(MLFloat16)));
      BufferUniquePtr group_reordered_W_buffer(group_reordered_W, BufferDeleter(alloc));

      const size_t W_offset = group_output_channels * kernel_dim;

      for (int64_t group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        ReorderFilter(Wdata, group_reordered_W, group_output_channels, group_input_channels, kernel_size);
        MlasHalfGemmPackB(group_output_channels, kernel_dim, group_reordered_W, group_output_channels, packed_W);
        packed_W += packed_W_size_;
        Wdata += W_offset;
      }

      if (share_prepacked_weights) {
        prepacked_weights->buffers_.push_back(std::move(packed_W_buffer_));
        prepacked_weights->buffer_sizes_.push_back(packed_W_data_size);
      }

      is_W_packed_ = true;
      is_packed = true;
      return Status::OK();
    }
  }

  if (share_prepacked_weights) {
    prepacked_weights->buffers_.push_back(nullptr);  // packed_W_buffer_ is nullptr
    prepacked_weights->buffer_sizes_.push_back(0);
  }

  size_t reordered_w_data_size = SafeInt<size_t>(sizeof(MLFloat16)) * output_channels * kernel_dim;
  auto* reordered_W = static_cast<MLFloat16*>(alloc->Alloc(reordered_w_data_size));

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset((void*)reordered_W, 0, reordered_w_data_size);

  reordered_W_buffer_ = BufferUniquePtr(reordered_W, BufferDeleter(alloc));

  ReorderFilter(Wdata, reordered_W, output_channels, group_input_channels, kernel_size);

  if (share_prepacked_weights) {
    prepacked_weights->buffers_.push_back(std::move(reordered_W_buffer_));
    prepacked_weights->buffer_sizes_.push_back(reordered_w_data_size);
  }

  is_W_packed_ = true;
  is_packed = true;
  return Status::OK();
}

Status FusedConvFp16::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                int input_idx,
                                                /*out*/ bool& used_shared_buffers) {
  if (input_idx != 1) {
    // only the filter tensor is packed
    return Status::OK();
  }

  used_shared_buffers = true;

  if (prepacked_buffers.size() == 1) {  // This means that only packed_W_ exists
    packed_W_buffer_ = std::move(prepacked_buffers[0]);
  } else if (prepacked_buffers.size() == 2) {  // This means that only reordered_W_ exists
    // Enforce that the first "placeholder" buffer is nullptr
    ORT_ENFORCE(prepacked_buffers[0].get() == nullptr);
    reordered_W_buffer_ = std::move(prepacked_buffers[1]);
  }

  return Status::OK();
}

Status FusedConvFp16::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = is_W_packed_ ? nullptr : context->Input<Tensor>(1);
  const auto& W_shape = W ? W->Shape() : W_shape_;
  const Tensor* B = num_inputs >= 3 ? context->Input<Tensor>(2) : nullptr;

  // This tensor should be added to the result AFTER activation is applied
  const Tensor* Sum = num_inputs >= 4 ? context->Input<Tensor>(3) : nullptr;

  const int64_t N = X->Shape()[0];
  const int64_t M = W_shape[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W_shape, channels_last_));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));
  const size_t kernel_rank = kernel_shape.size();

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
  }

  const int64_t C = X->Shape()[channels_last_ ? 1 + kernel_rank : 1];
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;
  const size_t spatial_dim_end = spatial_dim_start + kernel_rank;

  TensorShapeVector Y_dims({N});
  if (!channels_last_) {
    Y_dims.push_back(M);
  }
  TensorShape input_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_end);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  if (channels_last_) {
    Y_dims.push_back(M);
  }
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(spatial_dim_start, spatial_dim_end);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  if (Sum && Sum->Shape() != Y->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Z shape does not match output shape.",
                           " Z: ", Sum->Shape().ToString().c_str(),
                           " Output: ", Y->Shape().ToString().c_str());
  }

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  // Handle the case of a dynamic weight filter.
  BufferUniquePtr reordered_W_buffer;
  MLFloat16* reordered_W = nullptr;
  if (!packed_W_buffer_) {
    if (reordered_W_buffer_) {
      // Weight was constant and reordered.
      reordered_W = static_cast<MLFloat16*>(reordered_W_buffer_.get());
    } else {
      // Weight tensor was not constant or prepacking is disabled.
      reordered_W = static_cast<MLFloat16*>(alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * W_shape.Size()));
      reordered_W_buffer = BufferUniquePtr(reordered_W, BufferDeleter(alloc));
      ReorderFilter(
          static_cast<const MLFloat16*>(W->DataRaw()),
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
  const bool is_depthwise_conv = (group_input_channels == 1 && group_output_channels == 1);
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

  const auto* Xdata = X->Data<MLFloat16>();
  const auto* Bdata = B != nullptr ? B->Data<MLFloat16>() : nullptr;
  auto* Ydata = Y->MutableData<MLFloat16>();
  const auto* sum_data = Sum != nullptr ? Sum->Data<MLFloat16>() : nullptr;

  BufferUniquePtr transpose_input_buffer;
  BufferUniquePtr transpose_output_buffer;

  // Allocate temporary buffers for transposing to channels last format.
  if (!channels_last_) {
    auto* transpose_input = alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * X_offset + MLAS_SYMM_QGEMM_BUF_OVERRUN);
    transpose_input_buffer = BufferUniquePtr(transpose_input, BufferDeleter(alloc));
    auto* transpose_output = alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * Y_offset);
    transpose_output_buffer = BufferUniquePtr(transpose_output, BufferDeleter(alloc));
  }

  BufferUniquePtr col_buffer;
  BufferUniquePtr indirection_buffer;
  size_t ind_buf_length = 0;
  std::vector<MLFloat16> padding_data;

  bool use_indirection_buffer = false;
  if (is_depthwise_conv) {
    use_indirection_buffer = true;
  } else if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    //    if (is_symmetric_conv_) {
    //      use_indirection_buffer = true;
    //    } else {
    // Pointwise convolutions can use the original input tensor in place,
    // otherwise a temporary buffer is required for the im2col transform.
    int64_t group_col_buffer_size = (kernel_rank > 2) ? group_count * col_buffer_size : col_buffer_size;
    group_col_buffer_size += MLAS_SYMM_QGEMM_BUF_OVERRUN;
    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * group_col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));
    memset(col_data, 0, SafeInt<size_t>(sizeof(MLFloat16)) * group_col_buffer_size);
    //    }
  }

  //  bool parallel_batch = is_symmetric_conv_ && channels_last_;

  if (use_indirection_buffer) {
    // Allocate indirection buffer pointers and prepare a padding vector for
    // the im2col transform.
    ind_buf_length = SafeInt<size_t>(sizeof(const MLFloat16*)) * kernel_size * output_image_size;
    //    if (parallel_batch)
    //      ind_buf_length *= SafeInt<size_t>(N);  // ind buffer per each image in the batch
    auto* indirection_data = alloc->Alloc(ind_buf_length);
    indirection_buffer = BufferUniquePtr(indirection_data, BufferDeleter(alloc));
    padding_data.resize(static_cast<size_t>(C), MLFloat16());
  }

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  /*************************************
   * Thread partition idea: we are essentially partition a GEMM A[M,K] x B[K,N].
   * Here B contains the conv filters, which are usually not big, so we assume
   * it can be in cache entirely. Then we simply partition A horizontally into
   * thin slices along M dimension. This would ensure that the slice of A fits
   * into the cache and reduce the chance of kernel waiting for memory.
   *
   * The thickness of A slice should be multiple of kernel stride M. Since
   * we have to choose from many different kernels, the logic of finding
   * the stride M is hacky.
   */

  // The following convoluted branches must match the kernel selection logic
  // in conv_worker.

  const int32_t stride_m = 6;
  const int64_t task_count = (output_image_size + stride_m - 1) / stride_m;

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    const auto* input_data = Xdata;
    auto* output_data = Ydata;
    const auto* add_src = sum_data;

    if (!channels_last_) {
      // Transpose the input from channels first (CHW) to channels last (HWC).
      MlasTranspose(
          Xdata,
          static_cast<MLFloat16*>(transpose_input_buffer.get()),
          static_cast<size_t>(C),
          static_cast<size_t>(input_image_size));
      input_data = static_cast<MLFloat16*>(transpose_input_buffer.get());
      output_data = static_cast<MLFloat16*>(transpose_output_buffer.get());
      add_src = nullptr;
    }

    // Threaded implementation of ND convolution is not yet supported, so
    // prepare all im2col transformations here.
    if (col_buffer && kernel_rank > 2) {
      for (int64_t group_id = 0; group_id < group_count; ++group_id) {
        math::Im2col<MLFloat16, StorageOrder::NHWC>()(
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
            static_cast<MLFloat16*>(col_buffer.get()) + group_id * col_buffer_size,
            MLFloat16());
      }
    }

    auto conv_worker = [&](ptrdiff_t batch) {
      int64_t output_start = (int64_t)batch * (int64_t)stride_m;
      int64_t output_count = std::min((int64_t)stride_m, output_image_size - output_start);

      MLFloat16 const** worker_indirection_buffer = nullptr;
      if (indirection_buffer) {
        worker_indirection_buffer = static_cast<MLFloat16 const**>(indirection_buffer.get()) + output_start * kernel_size;
        math::Im2col<MLFloat16, StorageOrder::NHWC>()(
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
            worker_indirection_buffer,
            padding_data.data());
      }

      auto* worker_output = output_data + output_start * M;
      const auto* worker_addsrc = add_src == nullptr ? nullptr : add_src + output_start * M;

      if (is_depthwise_conv) {
        MLAS_HALF_GEMM_ACTIVATION_PROCESSOR act(activation_, worker_addsrc);
        MlasConvDepthwise(
            worker_indirection_buffer,
            reordered_W,
            Bdata,
            worker_output,
            static_cast<size_t>(M),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size),
            (!channels_last_ && sum_data) ? nullptr : &act);
      } else {
        for (int64_t group_id = 0; group_id < group_count; ++group_id) {
          // Prepare the im2col transformation or use the input buffer directly for
          // pointwise convolutions.
          const auto* group_input_data = input_data + group_id * group_input_channels;
          const MLFloat16* AData;
          size_t lda;
          if (col_buffer) {
            auto* worker_col_buffer = static_cast<MLFloat16*>(col_buffer.get()) + output_start * kernel_dim;
            if (kernel_rank == 2) {
              math::Im2col<MLFloat16, StorageOrder::NHWC>()(
                  group_input_data,
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
                  MLFloat16());
            } else if (kernel_rank == 1) {
              math::Im2col<MLFloat16, StorageOrder::NHWC>()(
                  group_input_data,
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
                  MLFloat16());
            } else {
              // Use the im2col buffer prepared outside the thread, indexed by group.
              worker_col_buffer += group_id * col_buffer_size;
            }
            AData = reinterpret_cast<const MLFloat16*>(worker_col_buffer);
            lda = static_cast<size_t>(kernel_dim);
          } else {
            AData = reinterpret_cast<const MLFloat16*>(group_input_data + output_start * C);
            lda = static_cast<size_t>(C);
          }

          const auto* gemm_add = add_src == nullptr ? nullptr : worker_addsrc + group_id * group_output_channels;
          MLAS_HALF_GEMM_ACTIVATION_PROCESSOR act(activation_, gemm_add);
          MLAS_HALF_GEMM_DATA_PARAMS gemm_params;
          gemm_params.A = AData;
          gemm_params.lda = lda;
          if (packed_W_buffer_) {
            gemm_params.B = static_cast<const MLFloat16*>(packed_W_buffer_.get()) + group_id * packed_W_size_,
            gemm_params.ldb = 0;
          } else {
            gemm_params.B = reordered_W + group_id * group_output_channels,
            gemm_params.ldb = static_cast<size_t>(M);
          }
          gemm_params.C = worker_output + group_id * group_output_channels;
          gemm_params.ldc = static_cast<size_t>(M);
          gemm_params.Bias = Bdata;
          gemm_params.OutputProcessor = (!channels_last_ && sum_data) ? nullptr : &act;  // process fused activation and add

          MlasHalfGemmBatch(
              static_cast<size_t>(output_count),
              static_cast<size_t>(group_output_channels),
              static_cast<size_t>(kernel_dim),
              1, &gemm_params, nullptr);
        }
      }
    };

    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, onnxruntime::narrow<ptrdiff_t>(task_count), conv_worker);

    if (!channels_last_) {
      // Transpose the output from channels last (NHWC) to channels first (NCHW).
      MlasTranspose(
          output_data,
          Ydata,
          static_cast<size_t>(output_image_size),
          static_cast<size_t>(M));
      if (sum_data != nullptr) {
        MLAS_HALF_GEMM_ACTIVATION_PROCESSOR proc(activation_, sum_data);
        proc.Process(Ydata, 0, 0, static_cast<size_t>(M),
                     static_cast<size_t>(output_image_size),
                     static_cast<size_t>(output_image_size));
      }
    }

    Xdata += X_offset;
    Ydata += Y_offset;
    if (sum_data != nullptr) {
      sum_data += Y_offset;
    }
  }

  return Status::OK();
}

//
// Operator definitions
//

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Conv,
    11,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    FusedConvFp16);

#ifndef DISABLE_CONTRIB_OPS

namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    NhwcFusedConv,
    kMSDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    FusedConvFp16);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedConv,
    kMSDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    FusedConvFp16);

}  // namespace contrib
#endif

}  // namespace onnxruntime

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
