// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/providers/cpu/nn/pool.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/platform/threadpool.h"
#include "core/common/safeint.h"

namespace onnxruntime {

/**
 * @brief Pooling operator for type FP16,
 * Only max pool and average pool supported.
 *
 * Single threadded operation for now.
 *
 * TODO!! implemente thread partition similar with
 * fp16 conv operator
 */
class PoolFp16 : public OpKernel {
 public:
  explicit PoolFp16(const OpKernelInfo& info)
      : OpKernel(info),
        pool_attrs_(info, info.GetKernelDef().OpName(), info.node().SinceVersion()),
        is_max_pool_(info.GetKernelDef().OpName() == "MaxPool"),
        channels_last_(info.GetKernelDef().Domain() == kMSInternalNHWCDomain) {}

  Status Compute(OpKernelContext* context) const override;

 protected:
  PoolAttributes pool_attrs_;
  bool is_max_pool_;  // either max pool or average pool
  bool channels_last_;

 private:
  // Correct reference fallback for ceil_mode==1 && count_include_pad==1 AveragePool. The MLAS
  // fp16 im2col path divides by the full kernel_size and cannot drop the ceil_mode phantom tail
  // cells. Rather than duplicate the pooling loop, this casts the tensor MLFloat16->float,
  // delegates to the shared float ComputeAveragePoolReferenceCompute (cpu/nn/pool.cc), then casts
  // float->MLFloat16 with a single round at store. Accumulation happens in float inside the shared
  // helper, so the result is identical to accumulate-in-float + round-once. Handles both NCHW and
  // channels-last (NHWC) layouts (transposing to NCHW for the shared helper) and 1D/2D/3D. Zero
  // MLAS edits.
  Status ComputeAveragePoolFp16Reference(OpKernelContext* context,
                                         const Tensor* X,
                                         const TensorShapeVector& output_dims,
                                         const TensorShapeVector& pads,
                                         int64_t N,
                                         int64_t C) const;
};

Status PoolFp16::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();

  const size_t input_rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_rank >= 3, "Input dimension cannot be less than 3.");

  const int64_t N = input_shape[0];
  const int64_t C = channels_last_ ? input_shape[input_rank - 1] : input_shape[1];

  ORT_ENFORCE(input_shape.Size() > 0 || N == 0, "Invalid input shape. Only N can be zero. Got:", input_shape);

  const size_t spatial_dims = input_rank - 2;
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;

  // Compute the output size and effective padding for this pooling operation.
  TensorShapeVector output_dims({N});
  if (!channels_last_) {
    output_dims.push_back(C);
  }
  TensorShapeVector pads = pool_attrs_.pads;
  TensorShapeVector kernel_shape = pool_attrs_.kernel_shape;
  TensorShapeVector strides = pool_attrs_.strides;
  TensorShapeVector dilations = pool_attrs_.dilations;
  if (pool_attrs_.global_pooling) {
    const auto& input_dims = input_shape.GetDims();
    if (channels_last_) {
      kernel_shape.assign(input_dims.begin() + 1, input_dims.end() - 1);
    } else {
      kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    }
    pads.resize(kernel_shape.size() * 2, 0);
    strides.resize(kernel_shape.size(), 1);
    dilations.resize(kernel_shape.size(), 1);
  }
  if (kernel_shape.size() != spatial_dims) {
    std::ostringstream ss;
    ss << "Invalid kernel shape. Input shape ";
    ss << (channels_last_ ? "(NHWC):[" : "(NCHW):[");
    // NumDimensions() is the number of dimensions; TensorShape::Size() is the element count
    // (product of dims), so iterating to Size() would index beyond the dimension array.
    for (size_t i = 0; i < input_shape.NumDimensions(); i++) {
      ss << input_shape[i] << ", ";
    }
    ss << "] Kernel shape:[";
    for (size_t i = 0; i < kernel_shape.size(); i++) {
      ss << kernel_shape[i] << ", ";
    }
    ss << "]";

    ORT_THROW(ss.str());
  }

  int64_t kernel_size = 1;
  int64_t input_image_size = 1;
  int64_t output_image_size = 1;
  for (size_t dim = 0; dim < spatial_dims; ++dim) {
    int64_t kernel = kernel_shape[dim];
    int64_t input_dim = input_shape[dim + spatial_dim_start];

    kernel_size *= kernel;
    input_image_size *= input_dim;

    int64_t output_dim = 0;
    pool_attrs_.ComputeSizePadDilations(input_dim,
                                        strides[dim],
                                        kernel,
                                        &pads.at(dim),
                                        &pads.at(spatial_dims + dim),
                                        dilations[dim],
                                        &output_dim);
    output_dims.push_back(output_dim);

    output_image_size *= output_dim;
  }
  if (channels_last_) {
    output_dims.push_back(C);
  }

  // The MLAS fp16 NHWC avg-pool (MlasNhwcAvgPool -> mlas/lib/pooling_fp16.cpp) divides every
  // window by the full kernel_size and cannot drop the ceil_mode phantom tail cells, giving a
  // too-small average for ceil_mode==1 && count_include_pad. Route only that combo to a correct
  // float-accumulating reference loop; every other case keeps the fast MLAS im2col path. This
  // mirrors the float fix in pool.cc (Pool<float, AveragePool>::Compute guard) so the fp16 and
  // float AveragePool paths stay dtype-consistent.
  if (!is_max_pool_ && pool_attrs_.ceil_mode == 1 && pool_attrs_.count_include_pad &&
      !pool_attrs_.global_pooling) {
    return ComputeAveragePoolFp16Reference(context, X, output_dims, pads, N, C);
  }

  const bool need_padding = !is_max_pool_ && pool_attrs_.count_include_pad;
  std::vector<MLFloat16> padding_data;
  if (need_padding) {
    padding_data.resize(static_cast<size_t>(C), MLFloat16());
  }

  const auto* Xdata = X->Data<MLFloat16>();
  auto* Y = context->Output(0, output_dims);
  auto* Ydata = Y->MutableData<MLFloat16>();

  // Allocate temporary buffers for transposing to channels last format.
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr transpose_input_buffer;
  BufferUniquePtr transpose_output_buffer;
  if (!channels_last_) {
    auto* transpose_input = alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * C * input_image_size + MLAS_SYMM_QGEMM_BUF_OVERRUN);
    transpose_input_buffer = BufferUniquePtr(transpose_input, BufferDeleter(alloc));
    auto* transpose_output = alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * C * output_image_size);
    transpose_output_buffer = BufferUniquePtr(transpose_output, BufferDeleter(alloc));
  }

  // Allocate indirection buffer pointers and prepare a padding vector for the
  // im2col transform.
  auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(const MLFloat16*)) * kernel_size * output_image_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(std::move(alloc)));

  const int64_t output_stride = std::max((int64_t)2, (int64_t)8192 / (kernel_size * C));
  const int64_t task_count = (output_image_size + output_stride - 1) / output_stride;
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    const auto* input_data = Xdata;
    auto* output_data = Ydata;

    if (!channels_last_) {
      // Transpose the input from channels first (CHW) to channels last (HWC).
      MlasTranspose(
          Xdata,
          static_cast<MLFloat16*>(transpose_input_buffer.get()),
          static_cast<size_t>(C),
          static_cast<size_t>(input_image_size),
          thread_pool);
      input_data = static_cast<MLFloat16*>(transpose_input_buffer.get());
      output_data = static_cast<MLFloat16*>(transpose_output_buffer.get());
    }

    auto worker = [&](ptrdiff_t batch) {
      int64_t output_start = (int64_t)batch * (int64_t)output_stride;
      int64_t output_count = std::min((int64_t)output_stride, output_image_size - output_start);
      auto* outputptr = output_data + output_stride * C * batch;
      auto indirection_buffer = static_cast<MLFloat16 const**>(col_buffer.get()) + output_start * kernel_size;

      math::Im2col<MLFloat16, StorageOrder::NHWC>()(
          input_data,
          C,
          input_shape.GetDims().data() + spatial_dim_start,
          output_dims.data() + spatial_dim_start,
          kernel_shape.data(),
          strides.data(),
          dilations.data(),
          pads.data(),
          static_cast<ptrdiff_t>(spatial_dims),
          output_start,
          output_count,
          indirection_buffer,
          need_padding ? padding_data.data() : nullptr);

      if (is_max_pool_) {
        MlasNhwcMaxPool(
            indirection_buffer,
            outputptr,
            static_cast<size_t>(C),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      } else {
        MlasNhwcAvgPool(
            indirection_buffer,
            outputptr,
            static_cast<size_t>(C),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      }
    };
    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, onnxruntime::narrow<ptrdiff_t>(task_count), worker);

    if (!channels_last_) {
      // Transpose the output from channels last (NHWC) to channels first (NCHW).
      MlasTranspose(
          output_data,
          Ydata,
          static_cast<size_t>(output_image_size),
          static_cast<size_t>(C),
          thread_pool);
    }
    Xdata += input_image_size * C;
    Ydata += output_image_size * C;
  }

  return Status::OK();
}

// fp16 AveragePool reference fallback for ceil_mode + count_include_pad. Casts the input
// MLFloat16->float (transposing NHWC->NCHW when channels_last_), runs the shared float pooling
// core (ComputeAveragePoolReferenceCompute in cpu/nn/pool.cc), then casts float->MLFloat16 into
// the output (transposing NCHW->NHWC back when channels_last_). A single round happens at the
// float->fp16 store; all accumulation is in float inside the shared core, so this is numerically
// equivalent to accumulate-in-float + round-once and stays in lockstep with the float path's
// clamped-divisor semantics. Reuses the float loop per xadupre's review request.
Status PoolFp16::ComputeAveragePoolFp16Reference(OpKernelContext* context,
                                                 const Tensor* X,
                                                 const TensorShapeVector& output_dims,
                                                 const TensorShapeVector& pads,
                                                 int64_t N,
                                                 int64_t C) const {
  const TensorShape& input_shape = X->Shape();
  const size_t spatial_dims = output_dims.size() - 2;
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;

  // Build NCHW spatial extents plus flat spatial sizes for input and output.
  int64_t input_image_size = 1;
  int64_t output_image_size = 1;
  TensorShapeVector x_nchw_dims({N, C});
  TensorShapeVector out_nchw_dims({N, C});
  for (size_t d = 0; d < spatial_dims; ++d) {
    const int64_t in_d = input_shape[d + spatial_dim_start];
    const int64_t out_d = output_dims[d + spatial_dim_start];
    x_nchw_dims.push_back(in_d);
    out_nchw_dims.push_back(out_d);
    input_image_size *= in_d;
    output_image_size *= out_d;
  }
  const TensorShape x_nchw_shape(x_nchw_dims);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const size_t x_float_count = static_cast<size_t>(N) * static_cast<size_t>(C) *
                               static_cast<size_t>(input_image_size);
  const size_t y_float_count = static_cast<size_t>(N) * static_cast<size_t>(C) *
                               static_cast<size_t>(output_image_size);
  auto x_float = IAllocator::MakeUniquePtr<float>(alloc, x_float_count);
  auto y_float = IAllocator::MakeUniquePtr<float>(alloc, y_float_count);

  // Cast MLFloat16 -> float into an NCHW buffer (transpose from NHWC when channels_last_).
  const auto* Xdata = X->Data<MLFloat16>();
  float* x_ptr = x_float.get();
  if (channels_last_) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t s = 0; s < input_image_size; ++s) {
          x_ptr[(n * C + c) * input_image_size + s] =
              Xdata[(n * input_image_size + s) * C + c].ToFloat();
        }
      }
    }
  } else {
    for (size_t i = 0; i < x_float_count; ++i) {
      x_ptr[i] = Xdata[i].ToFloat();
    }
  }

  // Run the shared float pooling core. The guard in Compute() excludes global_pooling, so
  // pool_attrs_ carries fully-populated kernel_shape/strides/dilations that match this call.
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  ORT_RETURN_IF_ERROR(ComputeAveragePoolReferenceCompute(x_ptr, y_float.get(), x_nchw_shape,
                                                         out_nchw_dims, pads, pool_attrs_, tp));

  // Cast float -> MLFloat16 into the output (transpose back to NHWC when channels_last_).
  // MLFloat16(float) is the single, final rounding step.
  auto* Y = context->Output(0, output_dims);
  auto* Ydata = Y->MutableData<MLFloat16>();
  const float* y_ptr = y_float.get();
  if (channels_last_) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        for (int64_t s = 0; s < output_image_size; ++s) {
          Ydata[(n * output_image_size + s) * C + c] =
              MLFloat16(y_ptr[(n * C + c) * output_image_size + s]);
        }
      }
    }
  } else {
    for (size_t i = 0; i < y_float_count; ++i) {
      Ydata[i] = MLFloat16(y_ptr[i]);
    }
  }

  return Status::OK();
}
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MaxPool, 8, 11,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MaxPool,
    12, 21,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MaxPool,
    22,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    AveragePool, 11, 18,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    AveragePool,
    19, 21,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    AveragePool,
    22,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    GlobalAveragePool,
    1, 21,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    GlobalAveragePool,
    22,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxPool,
    kMSInternalNHWCDomain,
    12,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AveragePool,
    kMSInternalNHWCDomain,
    11,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    GlobalAveragePool,
    kMSInternalNHWCDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

}  // namespace contrib
#endif  // DISABLE_CONTRIB_OPS

}  // namespace onnxruntime
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
