// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/nn/pool.h"
#include "core/providers/rocm/nn/max_pool_with_index.h"
#include "core/providers/rocm/math/unary_elementwise_ops_impl.h"
#include "core/providers/rocm/reduction/reduction_ops.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace rocm {

#define POOLING_KERNEL_(op_name, data_type, pool, pool_type, since_version)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                   \
      op_name,                                                                                     \
      kOnnxDomain,                                                                                 \
      since_version,                                                                               \
      data_type,                                                                                   \
      kRocmExecutionProvider,                                                                      \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      pool<data_type, pool_type>);

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version) \
  POOLING_KERNEL_(op_name, data_type, Pool, pool_type, since_version)

#define GLOBAL_POOLING_KERNEL(op_name, data_type, pool_type, since_version) \
  POOLING_KERNEL_(op_name, data_type, GlobalPool, pool_type, since_version)

#define POOLING_KERNEL_VERSIONED(op_name, data_type, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                  \
      op_name,                                                                              \
      kOnnxDomain,                                                                          \
      since_version,                                                                        \
      end_version,                                                                          \
      data_type,                                                                            \
      kRocmExecutionProvider,                                                               \
      (*KernelDefBuilder::Create())                                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),                   \
      Pool<data_type, pool_type>);

#define POOLING_KERNEL_WITH_INDICES(op_name, data_type, pool_type, since_version) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                  \
      op_name,                                                                    \
      kOnnxDomain,                                                                \
      since_version,                                                              \
      data_type,                                                                  \
      kRocmExecutionProvider,                                                     \
      (*KernelDefBuilder::Create())                                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())          \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),           \
      Pool<data_type, pool_type>);

#define POOLING_KERNEL_VERSIONED_WITH_INDICES(op_name, data_type, pool_type, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                               \
      op_name,                                                                                           \
      kOnnxDomain,                                                                                       \
      since_version,                                                                                     \
      end_version,                                                                                       \
      data_type,                                                                                         \
      kRocmExecutionProvider,                                                                            \
      (*KernelDefBuilder::Create())                                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>())                                 \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),                                  \
      Pool<data_type, pool_type>);

POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, double, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 7, 9)
POOLING_KERNEL_VERSIONED(AveragePool, float, AveragePool, 10, 10)
POOLING_KERNEL_VERSIONED(AveragePool, double, AveragePool, 10, 10)
POOLING_KERNEL_VERSIONED(AveragePool, MLFloat16, AveragePool, 10, 10)
// AveragePool and MaxPool op set 11 only update spec document on default value for dilations and strides.
POOLING_KERNEL(AveragePool, float, AveragePool, 11)
POOLING_KERNEL(AveragePool, double, AveragePool, 11)
POOLING_KERNEL(AveragePool, MLFloat16, AveragePool, 11)
POOLING_KERNEL_VERSIONED(MaxPool, float, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(MaxPool, double, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED(MaxPool, MLFloat16, MaxPool<1>, 1, 7)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, double, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 8, 9)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, double, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 10, 10)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, float, MaxPool<8>, 11, 11)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, double, MaxPool<8>, 11, 11)
POOLING_KERNEL_VERSIONED_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 11, 11)
POOLING_KERNEL_WITH_INDICES(MaxPool, float, MaxPool<8>, 12)
POOLING_KERNEL_WITH_INDICES(MaxPool, double, MaxPool<8>, 12)
POOLING_KERNEL_WITH_INDICES(MaxPool, MLFloat16, MaxPool<8>, 12)
POOLING_KERNEL_WITH_INDICES(MaxPool, int8_t, MaxPool<8>, 12)
POOLING_KERNEL_WITH_INDICES(MaxPool, uint8_t, MaxPool<8>, 12)

GLOBAL_POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1)
GLOBAL_POOLING_KERNEL(GlobalAveragePool, double, AveragePool, 1)
GLOBAL_POOLING_KERNEL(GlobalAveragePool, MLFloat16, AveragePool, 1)
GLOBAL_POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1)
GLOBAL_POOLING_KERNEL(GlobalMaxPool, double, MaxPool<1>, 1)
GLOBAL_POOLING_KERNEL(GlobalMaxPool, MLFloat16, MaxPool<1>, 1)

class MiopenPoolingDescriptor final {
 public:
  MiopenPoolingDescriptor() : desc_(nullptr) {
  }

  ~MiopenPoolingDescriptor() {
    if (desc_ != nullptr) {
      miopenDestroyPoolingDescriptor(desc_);
      desc_ = nullptr;
    }
  }

  MiopenPoolingDescriptor(const MiopenPoolingDescriptor&) = delete;
  MiopenPoolingDescriptor& operator=(const MiopenPoolingDescriptor&) = delete;

  Status Set(miopenPoolingMode_t mode,
             const gsl::span<const int64_t>& kernel_shape,
             const gsl::span<const int64_t>& pads,
             const gsl::span<const int64_t>& strides) {
    if (!desc_)
      MIOPEN_RETURN_IF_ERROR(miopenCreatePoolingDescriptor(&desc_));

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
    MIOPEN_RETURN_IF_ERROR(SetPoolingNdDescriptorHelper(
        desc_,
        mode,
        MIOPEN_PROPAGATE_NAN,
        rank,
        window.data(),
        padding.data(),
        stride.data()));

    return Status::OK();
  }

  operator miopenPoolingDescriptor_t() const { return desc_; }

 private:
  miopenPoolingDescriptor_t desc_;
};

template <typename T, typename PoolType>
Status Pool<T, PoolType>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToHipType<T>::MappedType HipT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();
  const auto x_dims = x_shape.GetDims();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  auto kernel_shape = pool_attrs_.kernel_shape;
  auto pads = pool_attrs_.pads;
  auto strides = pool_attrs_.strides;
  ORT_ENFORCE(!this->pool_attrs_.global_pooling);

  auto y_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  TensorShape y_shape(y_dims);
  Tensor* Y = context->Output(0, y_shape);
  // special case when there is a dim value of 0 in the shape.
  if (y_shape.Size() == 0)
    return Status::OK();

  auto x_data = reinterpret_cast<const HipT*>(X->Data<T>());
  auto y_data = reinterpret_cast<HipT*>(Y->MutableData<T>());

  TensorShapeVector x_dims_miopen(x_dims.begin(), x_dims.end());
  TensorShapeVector y_dims_miopen(y_dims);
  if (kernel_shape.size() < 2) {
    // miopen only takes 4D or 5D input, so pad dimensions if needed
    x_dims_miopen.push_back(1);
    y_dims_miopen.push_back(1);
    pads.insert(pads.begin() + kernel_shape.size(), 0);
    pads.insert(pads.end(), 0);
    kernel_shape.push_back(1);
    strides.push_back(1);
  }

  miopenPoolingMode_t mode = miopenPoolingMax;
  if constexpr (PoolType::type == onnxruntime::PoolType::kAveragePool) {
    mode = pool_attrs_.count_include_pad ? miopenPoolingAverageInclusive
                                         : miopenPoolingAverage;
  }
  MiopenPoolingDescriptor pooling_desc;
  ORT_RETURN_IF_ERROR(pooling_desc.Set(mode, kernel_shape, pads, strides));

  if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
    // Cast to float back and forth using temp buffer
    const auto alpha = Consts<float>::One;
    const auto beta = Consts<float>::Zero;
    MiopenTensor x_tensor;
    MiopenTensor y_tensor;
    ORT_RETURN_IF_ERROR(x_tensor.Set(x_dims_miopen, MiopenTensor::GetDataType<float>()));
    ORT_RETURN_IF_ERROR(y_tensor.Set(y_dims_miopen, MiopenTensor::GetDataType<float>()));

    const auto input_count = x_shape.Size();
    const auto output_count = y_shape.Size();

    IAllocatorUniquePtr<float> temp_X = GetScratchBuffer<float>(input_count, context->GetComputeStream());
    auto temp_Y = GetScratchBuffer<float>(output_count, context->GetComputeStream());
    Impl_Cast<HipT, float>(Stream(context), reinterpret_cast<const HipT*>(x_data), temp_X.get(), input_count);
    MIOPEN_RETURN_IF_ERROR(PoolingForwardHelper(GetMiopenHandle(context), pooling_desc, &alpha, x_tensor, temp_X.get(), &beta, y_tensor, temp_Y.get()));
    Impl_Cast<float, HipT>(Stream(context), temp_Y.get(), y_data, output_count);
  } else {
    const auto alpha = Consts<HipT>::One;
    const auto beta = Consts<HipT>::Zero;
    MiopenTensor x_tensor;
    MiopenTensor y_tensor;
    ORT_RETURN_IF_ERROR(x_tensor.Set(x_dims_miopen, MiopenTensor::GetDataType<HipT>()));
    ORT_RETURN_IF_ERROR(y_tensor.Set(y_dims_miopen, MiopenTensor::GetDataType<HipT>()));

    MIOPEN_RETURN_IF_ERROR(PoolingForwardHelper(GetMiopenHandle(context), pooling_desc, &alpha, x_tensor, x_data, &beta, y_tensor, y_data));
  }

  return Status::OK();
}

template <typename T>
Status Pool<T, MaxPool<8>>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToHipType<T>::MappedType HipT;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  auto kernel_shape = this->pool_attrs_.kernel_shape;
  auto pads = this->pool_attrs_.pads;
  auto strides = this->pool_attrs_.strides;
  ORT_ENFORCE(!this->pool_attrs_.global_pooling);

  auto y_dims = this->pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(y_dims));

  // special case when there is a dim value of 0 in the shape.
  if (Y->Shape().Size() == 0)
    return Status::OK();

  auto x_data = reinterpret_cast<const HipT*>(X->Data<T>());
  auto y_data = reinterpret_cast<HipT*>(Y->MutableData<T>());

  Tensor* I = context->Output(1, TensorShape(y_dims));
  if (nullptr != I || !this->pool_attrs_.default_dilations) {
    auto i_data = nullptr == I ? nullptr : I->MutableData<int64_t>();
    MaxPoolWithIndex<HipT, false>(
        this->Stream(context),
        x_shape,
        TensorShape(y_dims),
        kernel_shape,
        strides,
        pads,
        this->pool_attrs_.dilations,
        this->pool_attrs_.storage_order,
        x_data,
        y_data,
        i_data);
  } else {
    ORT_RETURN_IF_ERROR((Pool<T, MaxPool<1>>::ComputeInternal(context)));
  }
  return Status::OK();
}

template <typename T, typename PoolType>
Status GlobalPool<T, PoolType>::ComputeInternal(OpKernelContext* context) const {
  using HipT = typename ToHipType<T>::MappedType;
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  if (x_shape.NumDimensions() < 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input dimension cannot be less than 3.");
  }

  ORT_ENFORCE(this->pool_attrs_.global_pooling);

  miopenReduceTensorOp_t reduce_op;
  if constexpr (PoolType::type == onnxruntime::PoolType::kAveragePool) {
    reduce_op = MIOPEN_REDUCE_TENSOR_AVG;
  } else if (PoolType::type == onnxruntime::PoolType::kMaxPool) {
    reduce_op = MIOPEN_REDUCE_TENSOR_MAX;
  } else {
    ORT_NOT_IMPLEMENTED();
  }

  miopenDataType_t miopen_type_X = MiopenTensor::GetDataType<HipT>();

  MiopenReduceDescriptor reduce_desc;
  if constexpr (std::is_same<T, MLFloat16>::value) {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(
        reduce_op, MiopenTensor::GetDataType<float>(), MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES));
  } else {
    ORT_RETURN_IF_ERROR(reduce_desc.Set(reduce_op, miopen_type_X, MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES));
  }

  auto x_dims = x_shape.AsShapeVector();
  TensorShapeVector y_dims;
  y_dims.resize(x_dims.size(), 1);
  y_dims[0] = x_dims[0];
  y_dims[1] = x_dims[1];

  Tensor* Y = context->Output(0, y_dims);

  MiopenTensor input_tensor;
  MiopenTensor output_tensor;
  ORT_RETURN_IF_ERROR(input_tensor.Set(x_dims, miopen_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(y_dims, miopen_type_X));

  auto miopen_handle = this->GetMiopenHandle(context);
  size_t workspace_bytes{};
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionWorkspaceSize(
      miopen_handle, reduce_desc, input_tensor, output_tensor, &workspace_bytes));
  auto workspace_buffer = RocmKernel::GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());

  size_t indices_bytes{};
  MIOPEN_RETURN_IF_ERROR(miopenGetReductionIndicesSize(
      miopen_handle, reduce_desc, input_tensor, output_tensor, &indices_bytes));
  auto indices_buffer = RocmKernel::GetScratchBuffer<void>(indices_bytes, context->GetComputeStream());

  const auto one = ReduceConsts<HipT>::One;
  const auto zero = ReduceConsts<HipT>::Zero;

  MIOPEN_RETURN_IF_ERROR(miopenReduceTensor(
      miopen_handle, reduce_desc, indices_buffer.get(), indices_bytes, workspace_buffer.get(), workspace_bytes,
      &one, input_tensor, reinterpret_cast<const HipT*>(X->DataRaw()),
      &zero, output_tensor, reinterpret_cast<HipT*>(Y->MutableDataRaw())));

  return Status::OK();
}

}  // namespace rocm
}  // namespace onnxruntime
