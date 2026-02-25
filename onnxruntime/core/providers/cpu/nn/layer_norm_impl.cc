// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm_impl.h"
#include "layer_norm_helper.h"

#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/force_inline.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

namespace {

template <typename T,
          typename U,
          typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>, void>>
void ComputeJob(
    const T* X_data,
    const T* scale_data,
    const T* bias_data,
    const ptrdiff_t task_idx,
    const int64_t norm_size,
    const int64_t broadcast_param,
    const float* scale_float_ptr,
    const float* bias_float_ptr,
    float epsilon,
    bool simplified,
    T* Y_data,
    U* mean_data,
    U* inv_std_dev_data,
    AllocatorPtr alloc) {
  ORT_UNUSED_PARAMETER(scale_float_ptr);  // only used in MLFloat16 overload
  ORT_UNUSED_PARAMETER(bias_float_ptr);   // only used in MLFloat16 overload
  ORT_UNUSED_PARAMETER(alloc);

  const T* p_input = X_data + task_idx * norm_size;
  T* p_output = Y_data + task_idx * norm_size;

  T mean(0.0f);
  T mean_square(0.0f);

  for (int64_t h = 0; h < norm_size; h++) {
    p_output[h] = p_input[h];
    mean += p_input[h];
    mean_square += p_input[h] * p_input[h];
  }

  mean = mean / norm_size;
  if (simplified) {
    mean_square = sqrt(mean_square / norm_size + epsilon);
  } else {
    mean_square = sqrt(mean_square / norm_size - mean * mean + epsilon);
  }

  // Compute the offset of gamma and beta to support broadcasting.
  int64_t i = LAYER_NORM_SCALE_BIAS_OFFSET(broadcast_param, task_idx, norm_size);

  for (int64_t h = 0; h < norm_size; h++, i++) {
    if (simplified) {
      p_output[h] = p_output[h] / mean_square * scale_data[i];
    } else if (nullptr == bias_data) {
      p_output[h] = (p_output[h] - mean) / mean_square * scale_data[i];
    } else {
      p_output[h] = (p_output[h] - mean) / mean_square * scale_data[i] + bias_data[i];
    }
  }

  if (mean_data != nullptr) {
    // ONNX spec doesn't support 'double' for 'U' so when 'T' == double, 'U' == float and we need to narrow
    mean_data[task_idx] = gsl::narrow_cast<float>(mean);
  }

  if (inv_std_dev_data != nullptr) {
    inv_std_dev_data[task_idx] = gsl::narrow_cast<float>(1 / mean_square);
  }
}

// Helper to convert int64_t -> Eigen::Index safely
inline Eigen::Index ToEigenIndex(int64_t v) {
  return narrow<Eigen::Index>(v);
}

template <typename U>
void ComputeJob(
    const MLFloat16* X_data,
    const MLFloat16* scale_data,
    const MLFloat16* bias_data,
    const ptrdiff_t task_idx,
    const int64_t norm_size,
    const int64_t broadcast_param,
    const float* scale_float_ptr,
    const float* bias_float_ptr,
    float epsilon,
    bool simplified,
    MLFloat16* Y_data,
    U* mean_data,
    U* inv_std_dev_data,
    AllocatorPtr alloc) {
  ORT_UNUSED_PARAMETER(scale_data);  // only used in float/double overload
  ORT_UNUSED_PARAMETER(bias_data);   // only used in float/double overload
  ORT_UNUSED_PARAMETER(alloc);       // only required to create temporary float buffers

  // reinterpret input/output MLFloat16* as Eigen::half*
  const Eigen::half* p_input = reinterpret_cast<const Eigen::half*>(X_data + task_idx * norm_size);
  Eigen::half* p_output = reinterpret_cast<Eigen::half*>(Y_data + task_idx * norm_size);

  // Fix: cast norm_size to Eigen::Index
  Eigen::Map<const Eigen::Matrix<Eigen::half, Eigen::Dynamic, 1>> input_vec(
      p_input, ToEigenIndex(norm_size));
  Eigen::Map<Eigen::Matrix<Eigen::half, Eigen::Dynamic, 1>> output_vec(
      p_output, ToEigenIndex(norm_size));

  // Compute mean and mean_square in float for precision
  float mean = 0.0f;
  float mean_square = 0.0f;

  for (int64_t i = 0; i < norm_size; ++i) {
    float val = static_cast<float>(input_vec[ToEigenIndex(i)]);
    mean += val;
    mean_square += val * val;
  }

  mean /= gsl::narrow_cast<float>(norm_size);
  if (simplified) {
    mean_square = std::sqrt(mean_square / norm_size + epsilon);
  } else {
    mean_square = std::sqrt(mean_square / norm_size - mean * mean + epsilon);
  }

  // Offset calculation for broadcasting
  int64_t i = LAYER_NORM_SCALE_BIAS_OFFSET(broadcast_param, task_idx, norm_size);

  for (int64_t h = 0; h < norm_size; ++h, ++i) {
    float x = static_cast<float>(input_vec[ToEigenIndex(h)]);

    float y = 0.0f;
    if (simplified) {
      y = x / mean_square * scale_float_ptr[i];
    } else if (bias_float_ptr == nullptr) {
      y = (x - mean) / mean_square * scale_float_ptr[i];
    } else {
      y = (x - mean) / mean_square * scale_float_ptr[i] + bias_float_ptr[i];
    }

    output_vec[ToEigenIndex(h)] = gsl::narrow_cast<Eigen::half>(y);
  }

  if (mean_data != nullptr) {
    // ONNX spec doesn't support 'double' for 'U' so when 'T' == double, 'U' == float and we need to narrow
    mean_data[task_idx] = MLFloat16(mean);
  }

  if (inv_std_dev_data != nullptr) {
    inv_std_dev_data[task_idx] = MLFloat16(1.0f / mean_square);
  }
}
// Write a statistic value (mean or 1/denom) into the output buffer,
// converting from double to the target type U (including MLFloat16).
template <typename U>
ORT_FORCEINLINE void WriteStat(U* dst, ptrdiff_t index, double v) {
  if constexpr (std::is_same_v<U, MLFloat16>) {
    dst[index] = MLFloat16(static_cast<float>(v));
  } else {
    dst[index] = gsl::narrow_cast<U>(v);
  }
}
template <typename T>
struct NormalizationMath {
  static double LoadInput(const T* ptr, int64_t offset) {
    return static_cast<double>(ptr[offset]);
  }

  static double LoadScale(const T* scale_data,
                          const float* scale_float_ptr,
                          int64_t offset) {
    ORT_UNUSED_PARAMETER(scale_float_ptr);
    return static_cast<double>(scale_data[offset]);
  }

  static double LoadBias(const T* bias_data,
                         const float* bias_float_ptr,
                         int64_t offset) {
    ORT_UNUSED_PARAMETER(bias_float_ptr);
    if (!bias_data) {
      return 0.0;
    }
    return static_cast<double>(bias_data[offset]);
  }

  static void StoreOutput(T* dst, int64_t offset, double v) {
    dst[offset] = static_cast<T>(v);
  }
};

struct HalfMath {
  static double LoadInput(const MLFloat16* ptr, int64_t offset) {
    return static_cast<double>(static_cast<float>(ptr[offset]));
  }

  static double LoadScale(const MLFloat16* scale_data,
                          const float* scale_float_ptr,
                          int64_t offset) {
    if (scale_float_ptr) {
      return static_cast<double>(scale_float_ptr[offset]);
    }
    return static_cast<double>(static_cast<float>(scale_data[offset]));
  }

  static double LoadBias(const MLFloat16* bias_data,
                         const float* bias_float_ptr,
                         int64_t offset) {
    if (bias_float_ptr) {
      return static_cast<double>(bias_float_ptr[offset]);
    }
    if (bias_data) {
      return static_cast<double>(static_cast<float>(bias_data[offset]));
    }
    return 0.0;
  }

  static void StoreOutput(MLFloat16* dst, int64_t offset, double v) {
    dst[offset] = MLFloat16(static_cast<float>(v));
  }
};
// Shared generic implementation for LayerNorm with full NumPy-style broadcasting.
// DataT  - storage type (float/double/MLFloat16)
// MathPolicy - policy that handles load/store/cast for DataT
// U      - statistics output type (float, MLFloat16, etc.)
template <typename DataT, typename MathPolicy, typename U>
void ComputeJobGenericShared(
    const DataT* X_data,
    const DataT* scale_data,
    const DataT* bias_data,
    const ptrdiff_t task_idx,
    const LayerNormParams& params,
    const float* scale_float_ptr,
    const float* bias_float_ptr,
    float epsilon,
    bool simplified,
    DataT* Y_data,
    U* mean_data,
    U* inv_std_dev_data) {
  const int64_t norm_size = params.norm_size;
  const int64_t last_rank = params.last_rank;

  const DataT* p_input = X_data + task_idx * norm_size;
  DataT* p_output = Y_data + task_idx * norm_size;

  // Compute mean and denom (same for all types, via MathPolicy).
  double mean = 0.0;
  double mean_sq = 0.0;
  for (int64_t h = 0; h < norm_size; ++h) {
    const double xv = MathPolicy::LoadInput(p_input, h);
    mean += xv;
    mean_sq += xv * xv;
  }

  mean /= static_cast<double>(norm_size);
  const double denom = simplified
                           ? std::sqrt(mean_sq / norm_size + epsilon)
                           : std::sqrt(mean_sq / norm_size - mean * mean + epsilon);

  // Compute outer offsets for this logical row (same as before).
  int64_t off_sc_row = 0;
  int64_t off_bi_row = 0;

  const bool has_bias_any = (bias_data != nullptr) || (bias_float_ptr != nullptr);

  if (params.axis > 0) {
    const auto& outer_strides = params.x_outer_strides;

    for (int64_t d = 0; d < params.axis; ++d) {
      const size_t du = static_cast<size_t>(d);
      const int64_t dim = params.x_dims[du];
      const int64_t idx_d = (dim == 0)
                                ? 0
                                : (task_idx / outer_strides[du]) % dim;

      off_sc_row += idx_d * params.scale_strides[du];
      if (has_bias_any) {
        off_bi_row += idx_d * params.bias_strides[du];
      }
    }
  }

  // Prepare inner-dimension iteration (multi-dimensional idx for inner dims,
  //    plus optimized inner loop over the last dimension).
  ORT_ENFORCE(last_rank > 0);
  onnxruntime::InlinedVector<int64_t, 8> idx(static_cast<size_t>(last_rank), 0);

  const auto& x_inner_dims = params.x_inner_dims;
  const auto& scale_inner_inc = params.scale_inner_inc;
  const auto& bias_inner_inc = params.bias_inner_inc;

  const int64_t last_dim = x_inner_dims[static_cast<size_t>(last_rank - 1)];
  ORT_ENFORCE(last_dim > 0);
  ORT_ENFORCE(norm_size % last_dim == 0);
  const int64_t num_chunks = norm_size / last_dim;

  const int64_t sc_last_stride = !scale_inner_inc.empty() ? scale_inner_inc.back() : 0;
  const int64_t bi_last_stride =
      (has_bias_any && !bias_inner_inc.empty()) ? bias_inner_inc.back() : 0;

  //  Outer loop: iterate over "chunks" of the last dimension.
  for (int64_t c = 0; c < num_chunks; ++c) {
    int64_t off_sc = off_sc_row;
    int64_t off_bi = off_bi_row;

    // Base offsets for all inner dims except the last.
    for (int64_t d = 0; d < last_rank - 1; ++d) {
      const size_t du = static_cast<size_t>(d);
      off_sc += idx[du] * scale_inner_inc[du];
      if (has_bias_any) {
        off_bi += idx[du] * bias_inner_inc[du];
      }
    }

    const int64_t base_h = c * last_dim;

    //  Tight inner loop over the last dimension: compiler can vectorize this.
    for (int64_t i = 0; i < last_dim; ++i) {
      const int64_t h = base_h + i;

      const int64_t sc_offset = off_sc + i * sc_last_stride;
      const int64_t bi_offset = off_bi + i * bi_last_stride;

      const double x = MathPolicy::LoadInput(p_input, h);
      const double s = MathPolicy::LoadScale(scale_data, scale_float_ptr, sc_offset);
      const double b = MathPolicy::LoadBias(bias_data, bias_float_ptr, bi_offset);

      const double y = simplified
                           ? (x / denom) * s
                           : ((x - mean) / denom) * s + b;

      MathPolicy::StoreOutput(p_output, h, y);
    }

    //  Update multi-dimensional index 'idx' for the next chunk
    //    (iterate backwards from the second-to-last dimension).
    if (last_rank > 1) {
      for (int64_t d = last_rank - 2; d >= 0; --d) {
        const size_t du = static_cast<size_t>(d);
        if (++idx[du] < x_inner_dims[du]) {
          break;
        }
        idx[du] = 0;
      }
    }
  }

  //  Write statistics outputs.
  if (mean_data) {
    WriteStat<U>(mean_data, task_idx, mean);
  }
  if (inv_std_dev_data) {
    WriteStat<U>(inv_std_dev_data, task_idx, 1.0 / denom);
  }
}
template <typename T, typename U>
void ComputeJobGeneric(
    const T* X_data,
    const T* scale_data,
    const T* bias_data,
    const ptrdiff_t task_idx,
    const LayerNormParams& params,
    const float* scale_float_ptr,
    const float* bias_float_ptr,
    float epsilon,
    bool simplified,
    T* Y_data,
    U* mean_data,
    U* inv_std_dev_data) {
  ORT_UNUSED_PARAMETER(scale_float_ptr);
  ORT_UNUSED_PARAMETER(bias_float_ptr);

  using Policy = NormalizationMath<T>;
  ComputeJobGenericShared<T, Policy, U>(
      X_data, scale_data, bias_data,
      task_idx, params,
      nullptr,
      nullptr,
      epsilon, simplified,
      Y_data, mean_data, inv_std_dev_data);
}
template <typename U>
void ComputeJobGeneric(
    const MLFloat16* X_data,
    const MLFloat16* scale_data,
    const MLFloat16* bias_data,
    const ptrdiff_t task_idx,
    const LayerNormParams& params,
    const float* scale_float_ptr,
    const float* bias_float_ptr,
    float epsilon,
    bool simplified,
    MLFloat16* Y_data,
    U* mean_data,
    U* inv_std_dev_data) {
  using Policy = HalfMath;
  ComputeJobGenericShared<MLFloat16, Policy, U>(
      X_data, scale_data, bias_data,
      task_idx, params,
      scale_float_ptr, bias_float_ptr,
      epsilon, simplified,
      Y_data, mean_data, inv_std_dev_data);
}

void ConvertMLFloat16ToFloatIfNeeded(const Tensor& tensor, AllocatorPtr alloc, IAllocatorUniquePtr<float>& dest, bool& is_packed) {
  if (tensor.GetElementType() == utils::ToTensorProtoElementType<MLFloat16>()) {
    auto tensor_data_ptr = tensor.Data<MLFloat16>();
    auto tensor_size = static_cast<size_t>(tensor.Shape().Size());
    auto float_ptr = IAllocator::MakeUniquePtr<float>(alloc, tensor_size, true);

    MlasConvertHalfToFloatBuffer(tensor_data_ptr, float_ptr.get(), tensor_size);
    dest = std::move(float_ptr);
    is_packed = true;
  }
}

}  // namespace

LayerNormImpl::LayerNormImpl(const OpKernelInfo& op_kernel_info, bool simplified, bool contrib_op)
    : OpKernel(op_kernel_info),
      simplified_{simplified},
      contrib_op_{contrib_op},
      prepacked_scale_fp32_data_(nullptr),
      prepacked_bias_fp32_data_(nullptr) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
}

template <typename T, typename U>
Status LayerNormImpl::ComputeImpl(OpKernelContext* p_ctx, int64_t orig_axis, float epsilon, bool simplified) const {
  // Inputs
  const Tensor* X = p_ctx->Input<Tensor>(0);
  const Tensor* scale = prepacked_scale_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(1);
  const Tensor* bias = prepacked_bias_fp32_data_ ? nullptr : p_ctx->Input<Tensor>(2);
  const T* X_data = X->Data<T>();
  const T* scale_data = scale ? scale->Data<T>() : nullptr;
  const T* bias_data = (simplified || nullptr == bias) ? nullptr : bias->Data<T>();

  const TensorShape& x_shape = X->Shape();
  const TensorShape& scale_shape = scale ? scale->Shape() : prepacked_scale_fp32_shape_;
  const TensorShape& bias_shape = bias ? bias->Shape() : prepacked_bias_fp32_shape_;
  Tensor* Y = p_ctx->Output(0, x_shape);
  T* Y_data = Y->MutableData<T>();

  const int64_t axis = HandleNegativeAxis(orig_axis, x_shape.NumDimensions());

  std::vector<int64_t> mean_inv_std_dev_dim;
  mean_inv_std_dev_dim.reserve(x_shape.NumDimensions());
  for (int i = 0; i < static_cast<int>(x_shape.NumDimensions()); ++i) {
    if (i < axis) {
      mean_inv_std_dev_dim.emplace_back(x_shape.GetDims()[i]);
    } else {
      mean_inv_std_dev_dim.emplace_back(1);
    }
  }

  int output_index = 1;
  U* mean_data = nullptr;
  if (!simplified) {
    Tensor* mean = p_ctx->Output(output_index++, TensorShape(mean_inv_std_dev_dim));
    if (mean != nullptr) {
      mean_data = mean->MutableData<U>();
    }
  }

  U* inv_std_dev_data = nullptr;
  Tensor* inv_std_dev = p_ctx->Output(output_index, TensorShape(mean_inv_std_dev_dim));
  if (inv_std_dev != nullptr) {
    inv_std_dev_data = inv_std_dev->MutableData<U>();
  }

  onnxruntime::concurrency::ThreadPool* thread_pool = p_ctx->GetOperatorThreadPool();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(p_ctx->GetTempSpaceAllocator(&alloc));
  return ComputeWithoutContext<T, U>(X_data, x_shape, scale_data, scale_shape, bias_data, bias_shape, Y_data, mean_data,
                                     inv_std_dev_data, thread_pool, axis, epsilon, simplified, alloc);
}

Status LayerNormImpl::Compute(OpKernelContext* p_ctx) const {
  const auto elem_type = p_ctx->Input<Tensor>(0)->GetElementType();

  using SupportedTypeList = boost::mp11::mp_list<float, double, MLFloat16>;

  utils::MLTypeCallDispatcherFromTypeList<SupportedTypeList> t_disp(elem_type);
  return t_disp.InvokeRet<Status, SrcDispatcher>(this, p_ctx, axis_, epsilon_, simplified_, contrib_op_);
}

Status LayerNormImpl::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                              bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(prepacked_weights);

  is_packed = false;
  if (input_idx == 1) {  // scale
    prepacked_scale_fp32_shape_ = tensor.Shape();
    ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_scale_fp32_data_, is_packed);
  } else if (input_idx == 2) {  // bias
    prepacked_bias_fp32_shape_ = tensor.Shape();
    ConvertMLFloat16ToFloatIfNeeded(tensor, alloc, prepacked_bias_fp32_data_, is_packed);
  }

  return Status::OK();
}

template <typename T, typename U>
Status LayerNormImpl::ComputeWithoutContext(
    const T* X_data,
    const TensorShape& x_shape,
    const T* scale_data,
    const TensorShape& scale_shape,
    const T* bias_data,
    const TensorShape& bias_shape,
    T* Y_data,
    U* mean_data,
    U* inv_std_dev_data,
    onnxruntime::concurrency::ThreadPool* thread_pool,
    int64_t axis,
    float epsilon,
    bool simplified,
    AllocatorPtr alloc) const {
  LayerNormParams params;
  const bool has_bias =
      !simplified &&
      (bias_data != nullptr ||
       (std::is_same_v<T, MLFloat16> && prepacked_bias_fp32_data_ != nullptr));

  ORT_RETURN_IF_ERROR(
      LayerNormHelper::CheckInputs(x_shape, scale_shape, bias_shape, has_bias, axis, params));

  IAllocatorUniquePtr<float> scale_fp32;
  IAllocatorUniquePtr<float> bias_fp32;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    if (prepacked_scale_fp32_data_ == nullptr) {
      const size_t num_elems = static_cast<size_t>(params.scale_size);
      scale_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      MlasConvertHalfToFloatBuffer(scale_data, scale_fp32.get(), num_elems);
    }
    if (prepacked_bias_fp32_data_ == nullptr && bias_data) {
      const size_t num_elems = static_cast<size_t>(params.bias_size);
      bias_fp32 = IAllocator::MakeUniquePtr<float>(alloc, num_elems);
      MlasConvertHalfToFloatBuffer(bias_data, bias_fp32.get(), num_elems);
    }
  }

  // Resolve the float32 pointers for scale/bias (scf/bif) in the MLFloat16 case.
  // For non-MLFloat16 types, these remain null and the original T* buffers are used.
  const float* scf = nullptr;
  const float* bif = nullptr;

  if constexpr (std::is_same_v<T, MLFloat16>) {
    scf = prepacked_scale_fp32_data_ ? prepacked_scale_fp32_data_.get()
                                     : scale_fp32.get();

    if (has_bias) {
      bif = prepacked_bias_fp32_data_ ? prepacked_bias_fp32_data_.get()
                                      : (bias_fp32 ? bias_fp32.get() : nullptr);
    } else {
      bif = nullptr;
    }
  }
  // Launch one normalization job per logical row in X. For each row we either:
  //  - use the generic NumPy-style broadcasting path, or
  //  - use the existing fast-path based on broadcast_param.
  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool, static_cast<int32_t>(params.num_rows),
      [&](ptrdiff_t task_idx) {
        if (params.use_generic_broadcast) {
          ComputeJobGeneric(X_data, scale_data, bias_data, task_idx, params,
                            scf, bif,
                            epsilon, simplified, Y_data, mean_data, inv_std_dev_data);
        } else {
          ComputeJob(X_data, scale_data, bias_data, task_idx,
                     params.norm_size, params.broadcast_param,
                     scf, bif,
                     epsilon, simplified, Y_data, mean_data, inv_std_dev_data, alloc);
        }
      },
      0);

  return Status::OK();
}

}  // namespace onnxruntime
