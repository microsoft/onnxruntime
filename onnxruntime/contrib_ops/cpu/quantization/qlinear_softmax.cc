// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/qlinear_softmax.h"

#include <cstdint>
#include <type_traits>
#include <utility>

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/transpose.h"

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "gsl/gsl-lite.hpp"


namespace onnxruntime {
namespace contrib {

constexpr int OPSET13 = 13;

namespace {

// concept enabled in cpp20
template <typename T>
constexpr bool ValidType = std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>;

template <typename T, typename = typename std::enable_if_t<ValidType<T>> >
void QlinearBuildLookupTableUint32(uint32_t* table,
                                   const float x_scale,
                                   size_t reduce_len) {
  const double qscale =
      fmin(static_cast<double>(UINT32_MAX) / static_cast<double>(reduce_len), static_cast<double>(0x7fffff));
  for (int32_t i = 0; i < 256; i++) {
    double scaled_exp_xi = qscale * exp(static_cast<double>(i - 255) * static_cast<double>(x_scale));
    // we can't get the real max number of input tensor here, so we just assume 255.
    // in the process of computation, all numbers will have a shift to align 255
    if constexpr (std::is_same<T, int8_t>::value) {
      // 1 2 3 ......126 127 -128 -127 ..... -3 -2 -1
      uint8_t index = static_cast<uint8_t>(i - 128);
      table[index] = static_cast<uint32_t>(lrint(scaled_exp_xi));
    } else {
      table[i] = static_cast<uint32_t>(lrint(scaled_exp_xi));
    }
  }
}
}  // namespace

template <typename T>
void QLinearSoftmax<T>::BuildLookupTableIfFixed(const OpKernelInfo& info, uint32_t reduce_len) {
  const Tensor* tensor_x_scale = nullptr;

  bool get_x_scale = info.TryGetConstantInput(1, &tensor_x_scale);
  ORT_ENFORCE(tensor_x_scale == nullptr || IsScalarOr1ElementVector(tensor_x_scale),
              "QlinearBuildLookupTable : input X_scale must be a scalar or 1D tensor of size 1");
  bool is_fixed_parameters = get_x_scale;

  if (is_fixed_parameters) {
    fixed_lookup_table_.resize(256);
    const float X_scale = *(tensor_x_scale->Data<float>());
    QlinearBuildLookupTableUint32<T>(fixed_lookup_table_.data(), X_scale, reduce_len);
  }
}

template <typename T>
QLinearSoftmax<T>::QLinearSoftmax(const OpKernelInfo& info)
    : OpKernel(info) {
  const auto& node = info.node();
  int64_t opset = -1;
  Status status = info.GetAttr<int64_t>("opset", &opset);
  ORT_ENFORCE(status.IsOK(), "opset must be existed in attributes of QlinearSoftmax");
  opset_ = gsl::narrow_cast<int>(opset);

  int64_t axis = -1;
  status = info.GetAttr<int64_t>("axis", &axis);
  if (status.IsOK()) {
    axis_ = gsl::narrow_cast<int>(axis);
  } else {
    if (opset_ < OPSET13) {
      axis_ = 1;  // opset-12 and below, the default axis value is 1
    } else {
      axis_ = -1;  // opset-13, the default axis value is -1
    }
  }
  auto input_defs = node.InputDefs();
  const auto& x_shape = input_defs[0]->Shape();
  int rank = x_shape->dim_size();
  if (rank == 0) {
    return;
  }
  if (axis_ < 0) {
    axis_ = static_cast<int>(HandleNegativeAxis(axis_, int64_t(rank)));
  }
  uint32_t reduce_size = gsl::narrow_cast<uint32_t>(x_shape->dim(axis_).dim_value());
  if (opset_ < OPSET13) {
    for (int i = axis_ + 1; i < rank; i++) {
      reduce_size *= gsl::narrow_cast<uint32_t>(x_shape->dim(i).dim_value());
    }
  }
  ORT_ENFORCE(reduce_size > 0, "invalid reduce_size for softmax");
  this->BuildLookupTableIfFixed(info, reduce_size);
}

// compute method of Softmax
template <typename T>
Status QLinearSoftmax<T>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  auto* Y = ctx->Output(0, X_shape);

  // edge case. one or more dims with value of 0. nothing to do
  if (X_shape.Size() == 0) {
    return Status::OK();
  }
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  size_t D = X_shape[axis_];
  if (opset_ < OPSET13) {
    D = X_shape.SizeFromDimension(axis_);
  }
  const uint32_t* lookup_table = GetLookupTable(ctx, D);

  if (opset_ < OPSET13) {
    return ComputeImpl(ctx, *X, *Y, thread_pool, lookup_table);
  } else {
    return ComputeImplOpset13(ctx, *X, *Y, thread_pool, lookup_table);
  }
}
template <typename T>
common::Status QlinearSoftmaxCPU(size_t N,
                                 size_t D,
                                 const T* x_data,
                                 T* y_data,
                                 const uint32_t* lookup_table,
                                 uint32_t y_scale,
                                 T yzp,
                                 onnxruntime::concurrency::ThreadPool* thread_pool);

template <>
common::Status QlinearSoftmaxCPU<uint8_t>(size_t N,
                                          size_t D,
                                          const uint8_t* x_data,
                                          uint8_t* y_data,
                                          const uint32_t* lookup_table,
                                          uint32_t y_scale,
                                          uint8_t yzp,
                                          onnxruntime::concurrency::ThreadPool* thread_pool) {
  using onnxruntime::TensorOpCost;
  using onnxruntime::concurrency::ThreadPool;
  ThreadPool::TryParallelFor(
      thread_pool, N,
      TensorOpCost{static_cast<double>(D * 3),
                   static_cast<double>(D),
                   static_cast<double>(D * 3)},
      [x_data, y_data, D, y_scale, yzp, &lookup_table](std::ptrdiff_t first, std::ptrdiff_t last) {
        const auto c_y_scale = y_scale;
        const auto c_y_zp = yzp;
        const uint8_t* x_t = x_data + first * D;
        uint8_t* y_t = y_data + first * D;
        for (; first < last; first++) {
          // reduceMaxUint8
          uint8_t xmax = *std::max_element(x_t, x_t + D);
          // we want the xmas to align with 255 for higher precision.
          // as we build a lookup table with X-255. So we could use the adjustment here
          // to let all numbers have a shift in the lookup table.
          // 1 2 3 4 5 ...........................254 255
          // 1   3   5 ... 10
          // after the shift --->
          //                        235  237  239  .. 255
          const uint32_t* shifted_lookuptable = lookup_table + 255 - xmax;
          size_t elements_n = D;
          // reduceSumUin8ToUint32: need speedup
          uint32_t vsum = 0;
          const uint8_t* x_t_cur = x_t;
          do {
            const size_t vx = *x_t_cur++;
            vsum += shifted_lookuptable[vx];
          } while (--elements_n != 0);
          if (vsum == 0) {
            return;
          }
          elements_n = D;
          x_t_cur = x_t;
          // elementwise div
          const uint32_t vrounding = (vsum >> 1);
          do {
            const size_t vx = *x_t_cur++;
            const uint32_t vt = shifted_lookuptable[vx];
            // simulate round function, and re-quant to uint8
            const uint32_t vq = ((vt * c_y_scale) + vrounding) / vsum + c_y_zp;
            const uint8_t vy = vq > 255 ? static_cast<uint8_t>(255) : static_cast<uint8_t>(vq);
            *y_t++ = vy;
          } while (--elements_n != 0);
          x_t = x_t_cur;
        }
      });

  return Status::OK();
}

template <>
common::Status QlinearSoftmaxCPU<int8_t>(size_t N,
                                         size_t D,
                                         const int8_t* x_data,
                                         int8_t* y_data,
                                         const uint32_t* lookup_table,
                                         uint32_t y_scale,
                                         int8_t yzp,
                                         onnxruntime::concurrency::ThreadPool* thread_pool) {
  using onnxruntime::TensorOpCost;
  using onnxruntime::concurrency::ThreadPool;
  ThreadPool::TryParallelFor(
      thread_pool, N,
      TensorOpCost{static_cast<double>(D * 3),
                   static_cast<double>(D),
                   static_cast<double>(D * 3)},
      [x_data, y_data, D, y_scale, yzp, &lookup_table](std::ptrdiff_t first, std::ptrdiff_t last) {
        const auto c_y_scale = y_scale;
        const auto c_y_zp = yzp;

        const int8_t* x_t = x_data + first * D;
        int8_t* y_t = y_data + first * D;
        for (; first < last; first++) {
          // reduceMaxUint8
          int8_t xmax = *std::max_element(x_t, x_t + D);
          const size_t adjustment = 127 - xmax;
          const uint32_t* shifted_lookuptable = lookup_table;
          size_t elements_n = D;
          // reduceSumUin8ToUint32: need speedup
          uint32_t vsum = 0;
          const int8_t* x_t_cur = x_t;
          do {
            const size_t vx = uint8_t(adjustment + *x_t_cur++);
            vsum += shifted_lookuptable[vx];
          } while (--elements_n != 0);
          if (vsum == 0) {
            return;
          }
          elements_n = D;
          x_t_cur = x_t;
          // elementwise div
          const uint32_t vrounding = (vsum >> 1);
          do {
            const size_t vx = uint8_t(adjustment + *x_t_cur++);
            const uint32_t vt = shifted_lookuptable[vx];
            // simulate round function, and re-quant to int8
            const uint32_t vq = ((vt * c_y_scale) + vrounding) / vsum + c_y_zp;
            const int8_t vy = static_cast<int32_t>(vq) > 255 ? static_cast<int8_t>(255) : static_cast<int8_t>(vq);
            *y_t++ = vy;
          } while (--elements_n != 0);
          x_t = x_t_cur;
        }
      });

  return Status::OK();
}

template <typename T>
const uint32_t* QLinearSoftmax<T>::GetLookupTable(OpKernelContext* context, size_t reduce_len) const {
  const uint32_t* lookup_table = fixed_lookup_table_.data();
  if (fixed_lookup_table_.size() == 0) {
    tmp_lookup_table_.resize(256);
    lookup_table = tmp_lookup_table_.data();
    const float X_scale = *(context->Input<Tensor>(1)->Data<float>());
    QlinearBuildLookupTableUint32<T>(tmp_lookup_table_.data(), X_scale, reduce_len);
  }
  return lookup_table;
}

// opset-12 and below
template <typename T>
Status QLinearSoftmax<T>::ComputeImpl(OpKernelContext* context, const Tensor& input, Tensor& output,
                                      concurrency::ThreadPool* thread_pool,
                                      const uint32_t* lookup_table) const {
  const auto* Y_scale_tensor = context->Input<Tensor>(3);
  const auto* Y_zp_tensor = context->Input<Tensor>(4);
  const uint32_t Y_scale = gsl::narrow_cast<uint32_t>(1.0f / (*(Y_scale_tensor->Data<float>())));
  const T Y_zp = Y_zp_tensor ? *(Y_zp_tensor->Data<T>()) : 0;

  const auto& X_shape = input.Shape();
  const size_t N = X_shape.SizeToDimension(axis_);
  const size_t D = X_shape.SizeFromDimension(axis_);
  return QlinearSoftmaxCPU<T>(N, D, input.template Data<T>(), output.template MutableData<T>(),
                              lookup_table, Y_scale, Y_zp, thread_pool);
}

// opset-13 and above
template <typename T>
Status QLinearSoftmax<T>::ComputeImplOpset13(OpKernelContext* context,
                                             const Tensor& input, Tensor& output,
                                             concurrency::ThreadPool* thread_pool,
                                             const uint32_t* lookup_table) const {
  const auto* Y_scale_tensor = context->Input<Tensor>(3);
  const auto* Y_zp_tensor = context->Input<Tensor>(4);
  const uint32_t Y_scale = gsl::narrow_cast<uint32_t>(1.0f / (*(Y_scale_tensor->Data<float>())));
  const T Y_zp = Y_zp_tensor ? *(Y_zp_tensor->Data<T>()) : 0;

  const auto& X_shape = input.Shape();
  size_t rank = X_shape.NumDimensions();

  bool is_transpose_required = false;
  Tensor transposed_input;
  std::vector<int64_t> transposed_input_dims;
  Tensor intermediate_output;  // output that the softmax implementation will write into while using transposed input
  std::vector<size_t> permutation(rank);

  // The "semantic" meaning of axis has changed in opset-13.
  // Please compare: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
  // with https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Softmax-11 for detailed explanations
  // To account for the opset-13 behavior, our plan will be to transpose the "axis" dim to the innermost dim
  // and perform softmax and then reverse the transpose. We can skip the transposing aspect if the axis is already
  // the innermost dim
  if (size_t(axis_) != (rank - 1)) {
    is_transpose_required = true;
  }

  if (is_transpose_required) {
    AllocatorPtr alloc;
    auto status = context->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK())
      return status;

    std::iota(std::begin(permutation), std::end(permutation), 0);

    // swap the innermost dim with the dim corresponding to axis
    permutation[axis_] = rank - 1;
    permutation[rank - 1] = axis_;

    transposed_input_dims.reserve(rank);
    for (auto e : permutation) {
      transposed_input_dims.push_back(X_shape[e]);
    }

    // Allocate a temporary tensor to hold transposed input
    Tensor temp_input(input.DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, input, temp_input));
    transposed_input = std::move(temp_input);

    // Allocate memory for the intermediate output
    Tensor temp_output(output.DataType(), TensorShape(transposed_input_dims), alloc);
    intermediate_output = std::move(temp_output);
  }

  const size_t D = X_shape[axis_];
  const size_t N = X_shape.Size() / D;

  const T* x_data = is_transpose_required ? transposed_input.template Data<T>() : input.template Data<T>();
  T* y_data = is_transpose_required ? intermediate_output.template MutableData<T>() : output.template MutableData<T>();

  ORT_RETURN_IF_ERROR(QlinearSoftmaxCPU<T>(N, D, x_data, y_data, lookup_table, Y_scale, Y_zp, thread_pool));

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, intermediate_output, output));
  }
  return Status::OK();
}

#define REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                         \
      op_name, version, data_type,                                                           \
      KernelDefBuilder()                                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),                    \
      KERNEL_CLASS<data_type>);

REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearSoftmax, 1, uint8_t, QLinearSoftmax);
REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearSoftmax, 1, int8_t, QLinearSoftmax);

}  // namespace contrib
}  // namespace onnxruntime
