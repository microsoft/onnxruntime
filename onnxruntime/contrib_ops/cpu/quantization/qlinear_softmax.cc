// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/qlinear_softmax.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/transpose.h"

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include <gsl/gsl>

namespace onnxruntime {
namespace contrib {

constexpr int OPSET13 = 13;

namespace {

void QlinearBuildLookupTableUint32(gsl::span<QLinearSoftmax::EXP_OUT_DTYPE> table,
                                   const float x_scale,
                                   size_t reduce_len, bool is_signed) {
  // make sure sum(exp(x)) < max<T>()
  double bit_shift =
      log(std::numeric_limits<QLinearSoftmax::EXP_OUT_DTYPE>::max() / reduce_len);
  double reserve_bit = std::is_same_v<QLinearSoftmax::EXP_OUT_DTYPE, float> ? 5 : 3;
  bit_shift = std::max(0.0, bit_shift - reserve_bit) / x_scale;

  for (int32_t i = 0; i < 256; i++) {
    double scaled_exp_xi = exp((static_cast<double>(i) - 255 + bit_shift) * static_cast<double>(x_scale));
    // we can't get the real max value of input tensor here, so we just assume 255-bit_shift.
    // in the function of `MlasComputeQSoftmax`,
    // all numbers will have a shift (255-bit_shift-max_value) if its max value is not 255
    //
    // if is_signed index = [1 2 3 ......126 127 -128 -127 ..... -3 -2 -1]
    // else [0 1 2 3 4 ..... 256]
    uint8_t index = static_cast<uint8_t>(is_signed ? i - 128 : i);
    table[index] = static_cast<QLinearSoftmax::EXP_OUT_DTYPE>((scaled_exp_xi));
  }
}

void BuildLookupTableIfFixed(const OpKernelInfo& info,
                             std::vector<QLinearSoftmax::EXP_OUT_DTYPE>& fixed_lookup_table,
                             size_t reduce_len, bool is_signed) {
  const Tensor* tensor_x_scale = nullptr;

  bool get_x_scale = info.TryGetConstantInput(1, &tensor_x_scale);
  ORT_ENFORCE(tensor_x_scale == nullptr || IsScalarOr1ElementVector(tensor_x_scale),
              "QlinearBuildLookupTable : input X_scale must be a scalar or 1D tensor of size 1");
  bool is_fixed_parameters = get_x_scale && (tensor_x_scale != nullptr);

  if (is_fixed_parameters) {
    fixed_lookup_table.resize(256);
    const float X_scale = *(tensor_x_scale->Data<float>());
    QlinearBuildLookupTableUint32(fixed_lookup_table, X_scale, reduce_len, is_signed);
  }
}
}  // namespace

QLinearSoftmax::QLinearSoftmax(const OpKernelInfo& info)
    : OpKernel(info) {
  const auto& node = info.node();
  auto input_defs = node.InputDefs();
  auto input_type = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
  is_signed_ = (input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8);

  int64_t opset = -1;
  Status status = info.GetAttr<int64_t>("opset", &opset);
  ORT_ENFORCE(status.IsOK(), "opset must be existed in attributes of QlinearSoftmax");
  opset_ = gsl::narrow_cast<int>(opset);

  int64_t axis = -1;
  status = info.GetAttr<int64_t>("axis", &axis);
  if (status.IsOK()) {
    axis_ = gsl::narrow_cast<int>(axis);
  } else {
    // opset-12 and below, the default axis value is 1
    // opset-13, the default axis value is -1
    axis_ = opset_ < OPSET13 ? 1 : -1;
  }

  const auto* x_shape = input_defs[0]->Shape();
  if (x_shape != nullptr && x_shape->dim_size() > 0) {
    axis_ = static_cast<int>(HandleNegativeAxis(axis_, int64_t(x_shape->dim_size())));
    auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*x_shape);
    int64_t reduce_size = opset_ < OPSET13 ? input_shape.SizeFromDimension(axis_) : input_shape[axis_];
    // reduce_size could be negative if input-shape has a dynamic axis
    if (reduce_size > 0) {
      BuildLookupTableIfFixed(info, fixed_lookup_table_, onnxruntime::narrow<size_t>(reduce_size), is_signed_);
    }
  }
}

// compute method of Softmax
Status QLinearSoftmax::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  // edge case. one or more dims with value of 0. nothing to do
  if (X_shape.Size() == 0) {
    return Status::OK();
  }

  auto axis = static_cast<int>(HandleNegativeAxis(axis_, int64_t(X_shape.NumDimensions())));

  auto* Y = ctx->Output(0, X_shape);

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  const size_t D = onnxruntime::narrow<size_t>(opset_ < OPSET13 ? X_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis)) : X_shape[onnxruntime::narrow<size_t>(axis)]);
  EXP_OUT_DTYPE tmp_lookup_table[256];
  gsl::span<const EXP_OUT_DTYPE> lookup_table = GetLookupTable(ctx, tmp_lookup_table, D);

  if (opset_ < OPSET13) {
    return ComputeInternal(ctx, *X, *Y, lookup_table, axis, thread_pool);
  } else {
    return ComputeImplOpset13(ctx, *X, *Y, lookup_table, axis, thread_pool);
  }
}

gsl::span<const QLinearSoftmax::EXP_OUT_DTYPE> QLinearSoftmax::GetLookupTable(
    OpKernelContext* context,
    gsl::span<EXP_OUT_DTYPE> lookup_table_span,
    size_t reduce_len) const {
  gsl::span<const EXP_OUT_DTYPE> lookup_table = fixed_lookup_table_;
  if (fixed_lookup_table_.size() == 0) {
    lookup_table = lookup_table_span;
    const float X_scale = *(context->Input<Tensor>(1)->Data<float>());
    QlinearBuildLookupTableUint32(lookup_table_span, X_scale, reduce_len, is_signed_);
  }
  return lookup_table;
}

// opset-12 and below
Status QLinearSoftmax::ComputeInternal(OpKernelContext* context, const Tensor& input, Tensor& output,
                                       gsl::span<const EXP_OUT_DTYPE> lookup_table, int axis,
                                       concurrency::ThreadPool* thread_pool) const {
  const auto* X_scale_tensor = context->Input<Tensor>(1);
  const auto* Y_scale_tensor = context->Input<Tensor>(3);
  const auto* Y_zp_tensor = context->Input<Tensor>(4);
  const QLinearSoftmax::EXP_OUT_DTYPE Y_scale = std::floor(1.0F / (*(Y_scale_tensor->Data<float>())));
  const auto& X_shape = input.Shape();
  const size_t N = onnxruntime::narrow<size_t>(X_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis)));
  const size_t D = onnxruntime::narrow<size_t>(X_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis)));
  const int Y_zp = Y_zp_tensor ? (is_signed_ ? *(Y_zp_tensor->Data<int8_t>()) : *(Y_zp_tensor->Data<uint8_t>())) : 0;
  MlasComputeQSoftmax(input.DataRaw(), output.MutableDataRaw(), N, D, lookup_table.data(),
                      *X_scale_tensor->Data<float>(), Y_scale, Y_zp, is_signed_, thread_pool);
  return Status::OK();
}

// opset-13 and above
Status QLinearSoftmax::ComputeImplOpset13(OpKernelContext* context,
                                          const Tensor& input, Tensor& output,
                                          gsl::span<const EXP_OUT_DTYPE> lookup_table, int axis,
                                          concurrency::ThreadPool* thread_pool) const {
  const auto& X_shape = input.Shape();
  size_t rank = X_shape.NumDimensions();

  bool is_transpose_required = (size_t(axis) != (rank - 1));
  Tensor transposed_input;
  Tensor intermediate_output;  // output that the softmax implementation will write into while using transposed input
  std::vector<size_t> permutation(rank);

  if (is_transpose_required) {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    std::iota(std::begin(permutation), std::end(permutation), 0);

    // swap the innermost dim with the dim corresponding to axis
    permutation[axis] = rank - 1;
    permutation[rank - 1] = axis;
    std::vector<int64_t> transposed_input_dims(rank);
    std::transform(permutation.cbegin(), permutation.cend(),
                   transposed_input_dims.begin(), [&X_shape](size_t e) { return X_shape[e]; });

    // Allocate a temporary tensor to hold transposed input
    transposed_input = Tensor(input.DataType(), TensorShape(transposed_input_dims), alloc);
    // Perform the transpose
    ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(permutation, input, transposed_input));
    // Allocate memory for the intermediate output
    intermediate_output = Tensor(output.DataType(), TensorShape(transposed_input_dims), alloc);
  }

  common::Status status;

  const auto& input_tensor = is_transpose_required ? transposed_input : input;
  auto& output_tensor = is_transpose_required ? intermediate_output : output;

  ORT_RETURN_IF_ERROR(ComputeInternal(context, input_tensor, output_tensor, lookup_table, int(rank - 1), thread_pool));

  if (is_transpose_required) {
    // Perform the transpose to get the axes back to the original ordering
    status = (TransposeBase::DoTranspose(permutation, intermediate_output, output));
  }
  return status;
}

ONNX_CPU_OPERATOR_MS_KERNEL(
    QLinearSoftmax,
    1,
    KernelDefBuilder().TypeConstraint(
        "T",
        {DataTypeImpl::GetTensorType<uint8_t>(),
         DataTypeImpl::GetTensorType<int8_t>()}),
    QLinearSoftmax)

}  // namespace contrib
}  // namespace onnxruntime
