// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/pool.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/nn/max_pool_with_index.h"
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      op_name,                                                                          \
      kOnnxDomain,                                                                      \
      since_version,                                                                    \
      data_type,                                                                        \
      kCudaExecutionProvider,                                                           \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type>);

#define POOLING_KERNEL_VERSIONED(op_name, data_type, pool_type, since_version, end_version)                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                                                                          \
      op_name,                                                                                                                                      \
      kOnnxDomain,                                                                                                                                  \
      since_version,                                                                                                                                \
      end_version,                                                                                                                                  \
      data_type,                                                                                                                                    \
      kCudaExecutionProvider,                                                                                                                       \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      Pool<data_type, pool_type>);

POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, double, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 7, 9)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1)
POOLING_KERNEL(GlobalAveragePool, double, AveragePool, 1)
POOLING_KERNEL(GlobalAveragePool, MLFloat16, AveragePool, 1)
POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(MaxPool, double, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(MaxPool, MLFloat16, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED(MaxPool, double, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED(MaxPool, MLFloat16, MaxPool<8>, 8, 9)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1)
POOLING_KERNEL(GlobalMaxPool, double, MaxPool<1>, 1)
POOLING_KERNEL(GlobalMaxPool, MLFloat16, MaxPool<1>, 1)

class CudnnPoolingDescriptor final {
 public:
  CudnnPoolingDescriptor() : desc_(nullptr) {
  }

  ~CudnnPoolingDescriptor() {
    if (desc_ != nullptr) {
      cudnnDestroyPoolingDescriptor(desc_);
      desc_ = nullptr;
    }
  }

  Status Set(cudnnPoolingMode_t mode,
             const std::vector<int64_t>& kernel_shape,
             const std::vector<int64_t>& pads,
             const std::vector<int64_t>& strides) {
    if (!desc_)
      CUDNN_RETURN_IF_ERROR(cudnnCreatePoolingDescriptor(&desc_));

    int rank = gsl::narrow_cast<int>(kernel_shape.size());
    std::vector<int> window(rank);
    std::vector<int> padding(rank);
    std::vector<int> stride(rank);
    for (int i = 0; i < rank; i++) {
      window[i] = gsl::narrow_cast<int>(kernel_shape[i]);
    }
    for (int i = 0; i < rank; i++) {
      padding[i] = gsl::narrow_cast<int>(pads[i]);
    }
    for (int i = 0; i < rank; i++) {
      stride[i] = gsl::narrow_cast<int>(strides[i]);
    }
    CUDNN_RETURN_IF_ERROR(cudnnSetPoolingNdDescriptor(
        desc_,
        mode,
        CUDNN_PROPAGATE_NAN,
        rank,
        window.data(),
        padding.data(),
        stride.data()));

    return Status::OK();
  }

  operator cudnnPoolingDescriptor_t() const { return desc_; }

 private:
  cudnnPoolingDescriptor_t desc_;
};

template <typename T, typename PoolType>
Status Pool<T, PoolType>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  std::vector<int64_t> kernel_shape = kernel_shape_;
  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> strides = strides_;

  if (global_pooling_) {
    kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    pads.assign(kernel_shape.size(), 0);
    strides.assign(kernel_shape.size(), 1);
  }

  std::vector<int64_t> y_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads, dilations_, ceil_mode_);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  std::vector<int64_t> x_dims_cudnn = x_dims;
  std::vector<int64_t> y_dims_cudnn = y_dims;
  if (kernel_shape.size() < 2) {
    // cudnn only takes 4D or 5D input, so pad dimensions if needed
    x_dims_cudnn.push_back(1);
    y_dims_cudnn.push_back(1);
    pads.insert(pads.begin() + kernel_shape.size(), 0);
    pads.insert(pads.end(), 0);
    kernel_shape.push_back(1);
    strides.push_back(1);
  }

  const auto alpha = Consts<CudaT>::One;
  const auto beta = Consts<CudaT>::Zero;
  CudnnTensor x_tensor;
  CudnnTensor y_tensor;
  ORT_RETURN_IF_ERROR(x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));
  ORT_RETURN_IF_ERROR(y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>()));

  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  if (PoolType::type == onnxruntime::PoolType::kAveragePool) {
    mode = count_include_pad_ ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  CudnnPoolingDescriptor pooling_desc;
  ORT_RETURN_IF_ERROR(pooling_desc.Set(mode, kernel_shape, pads, strides));

  CUDNN_RETURN_IF_ERROR(cudnnPoolingForward(CudnnHandle(), pooling_desc, &alpha, x_tensor, x_data, &beta, y_tensor, y_data));

  return Status::OK();
}

template <typename T>
Status Pool<T, MaxPool<8>>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  std::vector<int64_t> kernel_shape = this->kernel_shape_;
  std::vector<int64_t> pads = this->pads_;
  std::vector<int64_t> strides = this->strides_;

  if (this->global_pooling_) {
    kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    pads.assign(kernel_shape.size(), 0);
    strides.assign(kernel_shape.size(), 1);
  }

  std::vector<int64_t> y_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads, this->dilations_, this->ceil_mode_);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  auto x_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
  auto y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  Tensor* I = context->Output(1, TensorShape(y_dims));
  if (nullptr != I) {
    auto i_data = I->template MutableData<int64_t>();
    MaxPoolWithIndex<CudaT>(
        x_shape,
        TensorShape(y_dims),
        kernel_shape,
        strides,
        pads,
        this->storage_order_,
        x_data,
        y_data,
        i_data);
  } else {
    Pool<T, MaxPool<1>>::ComputeInternal(context);
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
