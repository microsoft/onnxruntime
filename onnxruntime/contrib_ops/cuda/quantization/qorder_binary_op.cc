#include "qorder_binary_op.cuh"
#include "qorder_binary_op.h"
#include "qorder_common.h"

#include "core/providers/cuda/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    QOrderedAdd,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("F", BuildKernelDefConstraints<float>())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 4),
    QOrderedAdd);

ONNX_OPERATOR_KERNEL_EX(
    QOrderedBiasGelu,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("F", BuildKernelDefConstraints<float>())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 4),
    QOrderedBiasGelu);

Status ShapeReordered32C(gsl::span<const int64_t> const& src, gsl::span<int64_t>& dst) {
  if (src.size() < 1 || src.back() % 32 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Shape not meet clean tile requirement!", src);
  }
  if (dst.size() <= src.size() || dst.size() <= 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "No enough space for tile 32C shape");
  }
  size_t len = std::max(src.size() + 1, size_t{3});
  int64_t cols = src.back();
  int64_t rows = (src.size() > 1) ? src[src.size() - 2] : int64_t{1};
  size_t head = (src.size() > 1) ? (src.size() - 2) : size_t{0};
  std::copy_n(src.begin(), head, dst.begin());
  dst[head] = cols / 32;
  dst[head + 1] = rows;
  dst[head + 2] = 32;
  dst = gsl::span<int64_t>(dst.data(), len);
  return Status::OK();
}

QOrderedAdd::QOrderedAdd(const OpKernelInfo& info) : CudaKernel(info) {
  order_A_ = GetCublasLtOrderAttr(info, "order_A");
  order_B_ = GetCublasLtOrderAttr(info, "order_B");
  order_Y_ = GetCublasLtOrderAttr(info, "order_Y");
  ORT_ENFORCE(order_A_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_A");
  ORT_ENFORCE(order_B_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_B");
  ORT_ENFORCE(order_Y_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_Y");
}

Status QOrderedAdd::ComputeInternal(OpKernelContext* context) const {
  BinaryElementwisePreparation prepare;
  int64_t tmp_shape[16];

  auto lhs_tensor = context->Input<Tensor>(0);
  gsl::span<int64_t> ordered_lhs_dims(tmp_shape, 16);
  ORT_RETURN_IF_ERROR(ShapeReordered32C(lhs_tensor->Shape().GetDims(), ordered_lhs_dims));
  TensorShape lhs_shape(ordered_lhs_dims);
  auto rhs_tensor = context->Input<Tensor>(2);
  gsl::span<int64_t> ordered_rhs_dims(tmp_shape, 16);
  ORT_RETURN_IF_ERROR(ShapeReordered32C(rhs_tensor->Shape().GetDims(), ordered_rhs_dims));
  TensorShape rhs_shape(ordered_rhs_dims);

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(Node().Name(), lhs_tensor->Shape(), rhs_tensor->Shape(), output_shape));
  auto output_tensor = context->Output(0, output_shape);
  ORT_RETURN_IF_ERROR(BinaryElementwiseBroadcastPrepare(lhs_tensor, rhs_tensor, output_tensor, &prepare, &lhs_shape, &rhs_shape));

  float scaleC = *(context->Input<Tensor>(4)->Data<float>());
  float scaleA = *(context->Input<Tensor>(1)->Data<float>()) / scaleC;
  float scaleB = *(context->Input<Tensor>(3)->Data<float>()) / scaleC;
  QOrdered_Impl_Add(
      Stream(),
      prepare.output_rank_or_simple_broadcast,
      &prepare.lhs_padded_strides,
      prepare.lhs_tensor->template Data<int8_t>(),
      scaleA,
      &prepare.rhs_padded_strides,
      prepare.rhs_tensor->template Data<int8_t>(),
      scaleB,
      &prepare.fdm_output_strides,
      prepare.fdm_H,
      prepare.fdm_C,
      prepare.output_tensor->template MutableData<int8_t>(),
      (size_t)prepare.output_tensor->Shape().Size());

  return Status::OK();
}

QOrderedBiasGelu::QOrderedBiasGelu(const OpKernelInfo& info) : CudaKernel(info) {
  order_A_ = GetCublasLtOrderAttr(info, "order_A");
  order_B_ = GetCublasLtOrderAttr(info, "order_B");
  order_Y_ = GetCublasLtOrderAttr(info, "order_Y");
  ORT_ENFORCE(order_A_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_A");
  ORT_ENFORCE(order_B_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_B");
  ORT_ENFORCE(order_Y_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_Y");
}

Status QOrderedBiasGelu::ComputeInternal(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape();
  auto input_rank = input_shape.NumDimensions();

  const auto* bias_tensor = context->Input<Tensor>(2);
  const auto& bias_shape = bias_tensor->Shape();
  auto bias_rank = bias_shape.NumDimensions();
  ORT_ENFORCE(bias_rank == 1);

  if (bias_rank != 1 || input_rank < 1 || input_shape[input_rank - 1] != bias_shape[0]) {
    // TODO: Make the error message more verbose
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input shapes don't meet the requirements");
  }

  // TODO: Bit-wise op ?
  if (bias_shape[0] % 32 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Bias of shape in multiples of 32 are only supported");
  }

  auto* output_tensor = context->Output(0, input_shape);

  float input_scale = *(context->Input<Tensor>(1)->Data<float>());
  float bias_scale = *(context->Input<Tensor>(3)->Data<float>());
  float output_scale = *(context->Input<Tensor>(4)->Data<float>());

  int64_t cols = input_shape[input_rank - 1];
  int64_t rows = (input_rank > 1 ? input_shape[input_rank - 2] : 1);
  int64_t batches = 1;

  if (input_rank > 2) {
    for (size_t i = 0; i < input_rank - 2; ++i) {
      batches *= input_shape[i];
    }
  }

  fast_divmod batch_size(static_cast<int>(rows) * static_cast<int>(cols));
  fast_divmod rows_times_thirty_two(static_cast<int>(rows) * 32);
  fast_divmod thirty_two(32);

  QOrdered_Col32OrderImpl_BiasGelu(
      Stream(),
      input_tensor->Data<int8_t>(),
      input_scale,
      bias_tensor->Data<int8_t>(),
      bias_scale,
      output_tensor->MutableData<int8_t>(),
      output_scale,
      batch_size,
      rows_times_thirty_two,
      thirty_two,
      input_shape.Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
