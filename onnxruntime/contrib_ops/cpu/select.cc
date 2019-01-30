#include "select.h"
#include "onnx/defs/schema.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
static Status ComputeSelect(OpKernelContext* ctx) {
  auto& condition = *ctx->Input<Tensor>(0);
  auto& X = *ctx->Input<Tensor>(1);
  auto& Y = *ctx->Input<Tensor>(2);

  auto shape = X.Shape();

  const bool* cond = condition.Data<bool>();
  const T* x = X.template Data<T>();
  const T* y = Y.template Data<T>();

  auto& Z = *ctx->Output(0, shape);
  T* z = Z.template MutableData<T>();

  if (shape.IsScalar()) {
      if (!Y.Shape().IsScalar() || !condition.Shape().IsScalar()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
            "When X is scalar, both Y and condition must be scalar!",
            " Shape of Y:", Y.Shape(), ", shape of condition:", condition.Shape());
      }
      *z = (*cond) ? *x : *y;
      return Status::OK();
  }

  if (Y.Shape() != shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
        "Y's shape:", Y.Shape(), " is not same as X's shape:", X.Shape());
  }
  
  if (condition.Shape() == shape) {
    int64_t sz = int64_t{shape.Size()};
    ConstEigenMatrixMap<T> mat_x(x, sz, 1);
    ConstEigenMatrixMap<T> mat_y(y, sz, 1);
    ConstEigenMatrixMap<bool> mat_c(cond, sz, 1);
    EigenMatrixMap<T> mat_z(z, sz, 1);
    mat_z = mat_c.select(mat_x, mat_y);
  }
  else if (condition.Shape().NumDimensions() == 1 && shape.NumDimensions() > 1
                && condition.Shape()[0] == shape[0]) {
      int64_t width = int64_t{shape.SizeFromDimension(1)};

      #pragma omp parallel for
      for (int64_t i = 0, rows = shape[0]; i < rows; ++i) {
        int64_t offset = i * width;
        ConstEigenMatrixMap<T> mat_x(x + offset, width, 1);
        ConstEigenMatrixMap<T> mat_y(y + offset, width, 1);
        EigenMatrixMap<T> mat_z(z + offset, width, 1);
        mat_z = cond[i] ? mat_x : mat_y;
      }
  }
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
        "conditions's shape:", Y.Shape(), " is not same as X's shape:", X.Shape(),
        " and aslo not 1D vector with shape:[", shape.GetDims()[0], "].");
  }

  return Status::OK();
}

Status Select::Compute(OpKernelContext* ctx) const {
  auto data_type = ctx->Input<Tensor>(1)->DataType();
  if (data_type == DataTypeImpl::GetType<float>()) {
      return ComputeSelect<float>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int32_t>()) {
      return ComputeSelect<int32_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int64_t>()) {
      return ComputeSelect<int64_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int16_t>()) {
      return ComputeSelect<int16_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<int8_t>()) {
      return ComputeSelect<int8_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<uint32_t>()) {
      return ComputeSelect<uint32_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<uint64_t>()) {
      return ComputeSelect<uint64_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<uint16_t>()) {
      return ComputeSelect<uint16_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<uint8_t>()) {
      return ComputeSelect<uint8_t>(ctx);
  }
  else if (data_type == DataTypeImpl::GetType<double>()) {
      return ComputeSelect<double>(ctx);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
                         "Unsupportted tensor data type:",
                         data_type);
}

/* Select operator */
ONNX_OPERATOR_KERNEL_EX(
    Select,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {
        DataTypeImpl::GetTensorType<float>(), 
        DataTypeImpl::GetTensorType<double>(),
        DataTypeImpl::GetTensorType<int8_t>(),
        DataTypeImpl::GetTensorType<int16_t>(), 
        DataTypeImpl::GetTensorType<int32_t>(), 
        DataTypeImpl::GetTensorType<int64_t>(),
        DataTypeImpl::GetTensorType<uint8_t>(),
        DataTypeImpl::GetTensorType<uint16_t>(), 
        DataTypeImpl::GetTensorType<uint32_t>(),
        DataTypeImpl::GetTensorType<uint64_t>()}),
    Select);


}  // namespace contrib
}  // namespace onnxruntime
