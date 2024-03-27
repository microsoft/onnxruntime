// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/nn/pool.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/nn/max_pool_with_index.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version, op_domain, nhwc)              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                   \
      op_name, op_domain, since_version, data_type, kCudaExecutionProvider,                        \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type, nhwc>);

#define POOLING_KERNEL_VERSIONED(op_name, data_type, pool_type, since_version, end_version, op_domain, nhwc) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                                   \
      op_name, op_domain, since_version, end_version, data_type, kCudaExecutionProvider,                     \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),           \
      Pool<data_type, pool_type, nhwc>);

#define POOLING_KERNEL_WITH_INDICES(op_name, data_type, pool_type, since_version, op_domain, nhwc)    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(op_name, op_domain, since_version, data_type, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                         \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())    \
                                    .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),     \
                                Pool<data_type, pool_type, nhwc>);

#define POOLING_KERNEL_VERSIONED_WITH_INDICES(op_name, data_type, pool_type, since_version, end_version, op_domain, \
                                              nhwc)                                                                 \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(op_name, op_domain, since_version, end_version, data_type,                \
                                          kCudaExecutionProvider,                                                   \
                                          (*KernelDefBuilder::Create())                                             \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())        \
                                              .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),         \
                                          Pool<data_type, pool_type, nhwc>);

POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, double, AveragePool, 7, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 7, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, double, AveragePool, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 10, 10, kOnnxDomain, false)
// AveragePool and MaxPool op set 11 only update spec document on default value for dilations and strides.
POOLING_KERNEL(AveragePool, float, AveragePool, 11, kOnnxDomain, false)
POOLING_KERNEL(AveragePool, double, AveragePool, 11, kOnnxDomain, false)
POOLING_KERNEL(AveragePool, MLFloat16, AveragePool, 11, kOnnxDomain, false)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1, kOnnxDomain, false)
POOLING_KERNEL(GlobalAveragePool, double, AveragePool, 1, kOnnxDomain, false)
POOLING_KERNEL(GlobalAveragePool, MLFloat16, AveragePool, 1, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(MaxPool, double, MaxPool<1>, 1, 7, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED(MaxPool, MLFloat16, MaxPool<1>, 1, 7, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 8, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, double, MaxPool<8>, 8, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 8, 9, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, double, MaxPool<8>, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 10, 10, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 11, 11, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, double, MaxPool<8>, 11, 11, kOnnxDomain, false)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 11, 11, kOnnxDomain, false)
POOLING_KERNEL_WITH_INDICES(MaxPool, float, MaxPool<8>, 12, kOnnxDomain, false)
POOLING_KERNEL_WITH_INDICES(MaxPool, double, MaxPool<8>, 12, kOnnxDomain, false)
POOLING_KERNEL_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 12, kOnnxDomain, false)
POOLING_KERNEL_WITH_INDICES(MaxPool, int8_t, MaxPool<8>, 12, kOnnxDomain, false)
POOLING_KERNEL_WITH_INDICES(MaxPool, uint8_t, MaxPool<8>, 12, kOnnxDomain, false)

POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1, kOnnxDomain, false)
POOLING_KERNEL(GlobalMaxPool, double, MaxPool<1>, 1, kOnnxDomain, false)
POOLING_KERNEL(GlobalMaxPool, MLFloat16, MaxPool<1>, 1, kOnnxDomain, false)

// NHWC variants
#ifdef ENABLE_CUDA_NHWC_OPS
POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(MaxPool, MLFloat16, MaxPool<1>, 1, 7, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 8, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 8, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 10, 10, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 10, 10, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 11, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 11, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL_WITH_INDICES(MaxPool, float, MaxPool<8>, 12, kMSInternalNHWCDomain, true)
POOLING_KERNEL_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 12, kMSInternalNHWCDomain, true)
POOLING_KERNEL_WITH_INDICES(MaxPool, int8_t, MaxPool<8>, 12, kMSInternalNHWCDomain, true)
POOLING_KERNEL_WITH_INDICES(MaxPool, uint8_t, MaxPool<8>, 12, kMSInternalNHWCDomain, true)

POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1, kMSInternalNHWCDomain, true)
POOLING_KERNEL(GlobalMaxPool, MLFloat16, MaxPool<1>, 1, kMSInternalNHWCDomain, true)

POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 7, 9, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 10, 10, kMSInternalNHWCDomain, true)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 10, 10, kMSInternalNHWCDomain, true)
// AveragePool and MaxPool op set 11 only update spec document on default value for dilations
POOLING_KERNEL(AveragePool, float, AveragePool, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL(AveragePool, MLFloat16, AveragePool, 11, kMSInternalNHWCDomain, true)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1, kMSInternalNHWCDomain, true)
POOLING_KERNEL(GlobalAveragePool, MLFloat16, AveragePool, 1, kMSInternalNHWCDomain, true)
#endif

class CudnnPoolingDescriptor final {
 public:
  CudnnPoolingDescriptor() : desc_(nullptr) {}

  ~CudnnPoolingDescriptor() {
    if (desc_ != nullptr) {
      cudnnDestroyPoolingDescriptor(desc_);
      desc_ = nullptr;
    }
  }

  CudnnPoolingDescriptor(const CudnnPoolingDescriptor&) = delete;
  CudnnPoolingDescriptor& operator=(const CudnnPoolingDescriptor&) = delete;

  Status Set(cudnnPoolingMode_t mode, const gsl::span<const int64_t>& kernel_shape,
             const gsl::span<const int64_t>& pads, const gsl::span<const int64_t>& strides) {
    if (!desc_) CUDNN_RETURN_IF_ERROR(cudnnCreatePoolingDescriptor(&desc_));

    int rank = gsl::narrow_cast<int>(kernel_shape.size());
    InlinedVector<int> window(rank);
    InlinedVector<int> padding(rank);
    InlinedVector<int> stride(rank);
    for (int i = 0; i < rank; i++) {
      window[i] = gsl::narrow_cast<int>(kernel_shape[i]);
    }
    for (int i = 0; i < rank; i++) {
      padding[i] = gsl::narrow_cast<int>(pads[i]);
    }
    for (int i = 0; i < rank; i++) {
      stride[i] = gsl::narrow_cast<int>(strides[i]);
    }
    CUDNN_RETURN_IF_ERROR(SetPoolingNdDescriptorHelper(desc_, mode, CUDNN_PROPAGATE_NAN, rank, window.data(),
                                                       padding.data(), stride.data()));

    return Status::OK();
  }

  operator cudnnPoolingDescriptor_t() const { return desc_; }

 private:
  cudnnPoolingDescriptor_t desc_;
};

template <typename T, typename PoolType, bool Layout>
Status Pool<T, PoolType, Layout>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  auto kernel_shape = pool_attrs_.kernel_shape;
  auto strides = pool_attrs_.strides;
  TensorShapeVector pads = pool_attrs_.pads;

  if (pool_attrs_.global_pooling) {
    if constexpr (Layout == LAYOUT_NCHW) {
      kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    } else if constexpr (Layout == LAYOUT_NHWC) {
      kernel_shape.assign(x_dims.begin() + 1, x_dims.end() - 1);
    }
    pads.assign(2 * kernel_shape.size(), 0);
    strides.assign(kernel_shape.size(), 1);
  }
  auto out_channel = (Layout == LAYOUT_NHWC) ? x_shape[x_dims.size() - 1] : x_shape[1];

  auto y_dims = pool_attrs_.SetOutputSize(x_shape, out_channel, &pads, Layout == LAYOUT_NHWC);
  TensorShape y_shape(y_dims);
  Tensor* Y = context->Output(0, y_shape);
  // special case when there is a dim value of 0 in the shape.
  if (y_shape.Size() == 0) return Status::OK();

  auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  TensorShapeVector x_dims_cudnn(x_dims.begin(), x_dims.end());
  TensorShapeVector y_dims_cudnn(y_dims);
  if (kernel_shape.size() < 2) {
    // cuDNN only takes 4D or 5D input, so pad dimensions if needed
    if constexpr (Layout == LAYOUT_NHWC) {
      x_dims_cudnn.insert(x_dims_cudnn.end() - 1, 1);
      y_dims_cudnn.insert(y_dims_cudnn.end() - 1, 1);
      pads.insert(pads.begin() + pads.size() / 2, 0);
      pads.insert(pads.end(), 0);
      kernel_shape.insert(kernel_shape.end(), 1);
      strides.insert(strides.end(), 1);
    } else {  // Layout == LAYOUT_NCHW
      x_dims_cudnn.insert(x_dims_cudnn.end(), 1);
      y_dims_cudnn.insert(y_dims_cudnn.end(), 1);
      pads.insert(pads.begin() + pads.size() / 2, 0);
      pads.insert(pads.end(), 0);
      kernel_shape.insert(kernel_shape.end(), 1);
      strides.insert(strides.end(), 1);
    }
  }

  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  if constexpr (PoolType::type == onnxruntime::PoolType::kAveragePool) {
    mode = pool_attrs_.count_include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                         : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  CudnnPoolingDescriptor pooling_desc;
  ORT_RETURN_IF_ERROR(pooling_desc.Set(mode, kernel_shape, pads, strides));

  if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
    // Cast to float back and forth using temp buffer
    const auto alpha = Consts<float>::One;
    const auto beta = Consts<float>::Zero;
    CudnnTensor x_tensor;
    CudnnTensor y_tensor;
    ORT_RETURN_IF_ERROR(x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<float>(), Layout == LAYOUT_NHWC));
    ORT_RETURN_IF_ERROR(y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<float>(), Layout == LAYOUT_NHWC));

    const auto input_count = x_shape.Size();
    const auto output_count = y_shape.Size();

    IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count, context->GetComputeStream());
    auto temp_Y = GetScratchBuffer<float>(output_count, context->GetComputeStream());
    Impl_Cast<CudaT, float>(Stream(context), reinterpret_cast<const CudaT*>(x_data), temp_X.get(), input_count);
    CUDNN_RETURN_IF_ERROR(PoolingForwardHelper(GetCudnnHandle(context), pooling_desc, &alpha, x_tensor, temp_X.get(),
                                               &beta, y_tensor, temp_Y.get()));
    Impl_Cast<float, CudaT>(Stream(context), temp_Y.get(), y_data, output_count);
  } else {
    const auto alpha = Consts<CudaT>::One;
    const auto beta = Consts<CudaT>::Zero;
    CudnnTensor x_tensor;
    CudnnTensor y_tensor;
    ORT_RETURN_IF_ERROR(x_tensor.Set(x_dims_cudnn, CudnnTensor::GetDataType<CudaT>(), Layout == LAYOUT_NHWC));
    ORT_RETURN_IF_ERROR(y_tensor.Set(y_dims_cudnn, CudnnTensor::GetDataType<CudaT>(), Layout == LAYOUT_NHWC));

    CUDNN_RETURN_IF_ERROR(
        PoolingForwardHelper(GetCudnnHandle(context), pooling_desc, &alpha, x_tensor, x_data, &beta, y_tensor, y_data));
  }

  return Status::OK();
}

template <typename T, bool Layout>
Status Pool<T, MaxPool<8>, Layout>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto& x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  auto kernel_shape = this->pool_attrs_.kernel_shape;
  auto pads = this->pool_attrs_.pads;
  auto strides = this->pool_attrs_.strides;

  if (this->pool_attrs_.global_pooling) {
    if constexpr (Layout == LAYOUT_NCHW) {
      kernel_shape.assign(x_dims.begin() + 2, x_dims.end());
    } else if constexpr (Layout == LAYOUT_NHWC) {
      kernel_shape.assign(x_dims.begin() + 1, x_dims.end() - 1);
    }
    pads.assign(2 * kernel_shape.size(), 0);  // x{i}_begin + x{i}_end
    strides.assign(kernel_shape.size(), 1);
  }
  auto out_channel = Layout == LAYOUT_NHWC ? x_shape[x_shape.NumDimensions() - 1] : x_shape[1];
  auto y_dims = this->pool_attrs_.SetOutputSize(x_shape, out_channel, &pads, Layout == LAYOUT_NHWC);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  // special case when there is a dim value of 0 in the shape.
  if (Y->Shape().Size() == 0) return Status::OK();

  auto x_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto y_data = reinterpret_cast<CudaT*>(Y->MutableData<T>());

  // I is in NCHW format and the contained indices use NCHW math to compute the index
  auto i_dims = y_dims;
  if constexpr (Layout == LAYOUT_NHWC) {
    // y_dims in NHWDC format, i_dims has to be in NCHWD format.
    i_dims.insert(i_dims.begin() + 1, i_dims.back());  // N*C*HWDC
    i_dims.pop_back();                                 // NCHW
  }

  Tensor* I = context->Output(1, TensorShape(i_dims));
  if (nullptr != I || !this->pool_attrs_.default_dilations) {
    auto i_data = nullptr == I ? nullptr : I->MutableData<int64_t>();
    MaxPoolWithIndex<CudaT, Layout == LAYOUT_NHWC>(this->Stream(context), x_shape, TensorShape(y_dims), kernel_shape,
                                                   strides, pads, this->pool_attrs_.dilations,
                                                   this->pool_attrs_.storage_order, x_data, y_data, i_data);
  } else {
    ORT_RETURN_IF_ERROR((Pool<T, MaxPool<1>, Layout == LAYOUT_NHWC>::ComputeInternal(context)));
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
