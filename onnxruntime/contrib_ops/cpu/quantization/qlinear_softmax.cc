// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_softmax.h"
#include <cstddef>
#include <cstdint>
#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/transpose.h"

#include "core/mlas/inc/mlas.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

void QlinearBuildLookupTableUint32(uint32_t* table,
                                   const float x_scale,
                                   size_t reduce_len) {
  const double qscale =
      fmin(((double)UINT32_MAX) / (double)reduce_len, double(0x7fffff));
  for (int32_t i = 0; i < 256; i++) {
    const double scaled_exp_xi =
        qscale * exp((double)(i - 255) * (double)x_scale);
    table[(uint32_t)i] = (uint32_t)lrint(scaled_exp_xi);
  }
}

template <typename T>
void QLinearSoftmax<T>::BuildLookupTableIfFixed(const OpKernelInfo& info, uint32_t channels) {
  const Tensor* tensor_x_scale = nullptr;
  const Tensor* tensor_x_zero_point = nullptr;
  const Tensor* tensor_y_scale = nullptr;
  const Tensor* tensor_y_zero_point = nullptr;

  bool get_x_scale = info.TryGetConstantInput(1, &tensor_x_scale);
  bool get_x_zero_point = !info.node().InputDefs()[2]->Exists() || info.TryGetConstantInput(2, &tensor_x_zero_point);
  bool get_y_scale = info.TryGetConstantInput(3, &tensor_y_scale);
  bool get_y_zero_point = !info.node().InputDefs()[4]->Exists() || info.TryGetConstantInput(4, &tensor_y_zero_point);
  bool is_fixed_parameters = get_x_scale && get_x_zero_point && get_y_scale && get_y_zero_point;
  ORT_ENFORCE(tensor_x_scale == nullptr || IsScalarOr1ElementVector(tensor_x_scale),
              "QlinearBuildLookupTable : input X_scale must be a scalar or 1D tensor of size 1");

  if (is_fixed_parameters) {
    fixed_lookup_table_.resize(256);
    const float X_scale = *(tensor_x_scale->Data<float>());
    QlinearBuildLookupTableUint32(fixed_lookup_table_.data(), X_scale, channels);
  }
}

template <typename T>
QLinearSoftmax<T>::QLinearSoftmax(const OpKernelInfo& info)
    : OpKernel(info) {
  const auto& node = info.node();
  opset_ = node.SinceVersion();
  int64_t axis = -1;
  Status status = info.GetAttr<int64_t>("axis", &axis);
  if (status.IsOK()) {
    axis_ = gsl::narrow_cast<int>(axis);
  } else {
    if (opset_ < 13) {
      axis_ = 1;  // opset-12 and below, the default axis value is 1
    } else {
      axis_ = -1;  // opset-13, the default axis value is -1
    }
  }
  auto input_defs = node.InputDefs();
  const auto& x_shape = input_defs[0]->Shape();
  size_t rank = x_shape->dim_size();
  if (rank == 0) {
    return;
  }
  if (axis_ < 0) {
    axis_ = static_cast<int>(HandleNegativeAxis(axis_, int64_t(rank)));
  }
  uint32_t channels = gsl::narrow_cast<uint32_t>(x_shape->dim(axis_).dim_value());
  if (opset_ < 13) {
    for (size_t i = axis_; i < rank; i++) {
      channels *= gsl::narrow_cast<uint32_t>(x_shape->dim(i).dim_value());
    }
  }
  ORT_ENFORCE(channels>0, "invalid channels for softmax");
  this->BuildLookupTableIfFixed(info, channels);
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
  if (opset_ < 13) {
    D = X_shape.SizeFromDimension(axis_);
  }
  const uint32_t* lookup_table = GetLookupTable(ctx, D);

  if (opset_ < 13) {
    return ComputeImpl(*X, *Y, thread_pool, lookup_table);
  } else {
    return ComputeImplOpset13(ctx, *X, *Y, thread_pool, lookup_table);
  }
}

common::Status QlinearSoftmaxCPU(size_t N,
                                 size_t D,
                                 const uint8_t* x_data,
                                 uint8_t* y_data,
                                 const uint32_t* lookup_table,
                                 onnxruntime::concurrency::ThreadPool* thread_pool) {
  using onnxruntime::TensorOpCost;
  using onnxruntime::concurrency::ThreadPool;
  ThreadPool::TryParallelFor(
      thread_pool, N, TensorOpCost{1.0, 1.0, 1.0},
      [x_data, y_data, D, &lookup_table](std::ptrdiff_t first, std::ptrdiff_t last) {
        const uint8_t* x_t = x_data + first * D;
        uint8_t* y_t = y_data + first * D;
        for (; first < last; first++) {
          // reduceMax
          uint8_t xmax = *std::max_element(x_t, x_t + D);
          // we want the xmas to align with 255 for higher accuracy.
          const uint32_t* tb = lookup_table + 255 - xmax;
          size_t size_n = D;
          // reduceSum
          uint32_t vsum = 0;
          const uint8_t* x_t_cur = x_t;
          do {
            const size_t vx = *x_t_cur++;
            vsum += tb[vx];
          } while (--size_n != 0);
          if (vsum == 0) {
            return;
          }
          size_n = D;
          x_t_cur = x_t;
          // elementwise div
          const uint32_t vrounding = (vsum >> 1);
          do {
            const size_t vx = *x_t_cur++;
            const uint32_t vt = tb[vx];
            // simulate round
            const uint32_t vq = ((vt << 8) + vrounding) / vsum;
            const uint8_t vy = vq > 255 ? UINT8_C(255) : (uint8_t)vq;
            *y_t++ = vy;
          } while (--size_n != 0);
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
    QlinearBuildLookupTableUint32(tmp_lookup_table_.data(), X_scale, reduce_len);
  }
  return lookup_table;
}

// opset-12 and below
template <typename T>
Status QLinearSoftmax<T>::ComputeImpl(const Tensor& input, Tensor& output,
                                      concurrency::ThreadPool* thread_pool,
                                      const uint32_t* lookup_table) const {
  const auto& X_shape = input.Shape();
  const size_t N = X_shape.SizeToDimension(axis_);
  const size_t D = X_shape.SizeFromDimension(axis_);
  return QlinearSoftmaxCPU(N, D, input.template Data<T>(), output.template MutableData<T>(),
                           lookup_table, thread_pool);
}

// opset-13 and above
template <typename T>
Status QLinearSoftmax<T>::ComputeImplOpset13(OpKernelContext* context,
                                             const Tensor& input, Tensor& output,
                                             concurrency::ThreadPool* thread_pool,
                                             const uint32_t* lookup_table) const {
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
  const size_t N = X_shape.Size()/D;

  const uint8_t* x_data = reinterpret_cast<const uint8_t*>(input.template Data<T>());
  uint8_t* y_data = reinterpret_cast<uint8_t*>(output.template MutableData<T>());
  return QlinearSoftmaxCPU(N, D, x_data, y_data, lookup_table, thread_pool);
}

#define REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                         \
      op_name, version, data_type,                                                           \
      KernelDefBuilder()                                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),                    \
      KERNEL_CLASS<data_type>);

REGISTER_QLINEAR_LOOKUPTABLE_TYPED_KERNEL(QLinearSoftmax, 1, uint8_t, QLinearSoftmax);

}  // namespace contrib
}  // namespace onnxruntime
