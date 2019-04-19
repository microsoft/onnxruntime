// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include <sstream>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/GPU/Half.h"
#include "core/common/common.h"

#if defined(USE_MLAS) && defined(_M_AMD64)
#include "core/mlas/inc/mlas.h"
#endif

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

template <typename SrcType,
          typename DstType>
inline void CastData(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto shape_size = shape.Size();
  auto in_vector = ConstEigenVectorMap<SrcType>(in->template Data<SrcType>(), shape_size);
  auto output_vector = EigenVectorMap<DstType>(out->template MutableData<DstType>(), shape_size);
  output_vector = in_vector.template cast<DstType>();
}

template <>
inline void CastData<float, MLFloat16>(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto out_data = out->template MutableData<MLFloat16>();
  auto shape_size = shape.Size();
  auto in_vector = ConstEigenVectorMap<float>(in->template Data<float>(), shape_size);
  auto output_vector = EigenVectorMap<Eigen::half>(static_cast<Eigen::half*>(static_cast<void*>(out_data)), shape_size);
  output_vector = in_vector.template cast<Eigen::half>();
}

template <>
inline void CastData<MLFloat16, float>(const Tensor* in, Tensor* out, const TensorShape& shape) {
  auto out_data = out->template MutableData<float>();
  auto in_data = in->template Data<MLFloat16>();
  auto shape_size = shape.Size();
#if defined(USE_MLAS) && defined(_M_AMD64)
  MlasConvertHalfToFloatBuffer(&in_data[0].val, out_data, shape_size);
#else
  auto in_vector = ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(in_data)), shape_size);
  auto output_vector = EigenVectorMap<float>(out_data, shape_size);
  output_vector = in_vector.template cast<float>();
#endif
}

template <typename SrcType,
          typename DstType>
inline void CastFloat16Data(const Tensor* in, Tensor* out, const TensorShape& shape, const AllocatorPtr& allocator) {
  ORT_ENFORCE(allocator != nullptr);
  const int64_t len = shape.Size();
  ORT_ENFORCE(len > 0);
  void* buffer = allocator->AllocArray(sizeof(float), len);
  ORT_ENFORCE(buffer);
  Tensor tmp_tensor(DataTypeImpl::GetType<float>(), shape, buffer, allocator->Info());
  if (std::is_same<SrcType, MLFloat16>::value) {
    CastData<MLFloat16, float>(in, &tmp_tensor, shape);  // first cast to float
    CastData<float, DstType>(&tmp_tensor, out, shape);   // then cast to the destination type.
  } else if (std::is_same<DstType, MLFloat16>::value) {
    CastData<SrcType, float>(in, &tmp_tensor, shape);
    CastData<float, MLFloat16>(&tmp_tensor, out, shape);
  }
  allocator->Free(buffer);
}

template <typename SrcType>
inline void CastToStringData(const Tensor* in, Tensor* out, const TensorShape& shape) {
  const int64_t len = shape.Size();
  ORT_ENFORCE(len > 0);
  for (int i = 0; i < len; ++i) {
    if (std::is_floating_point<SrcType>::value && std::isnan(in->Data<SrcType>()[i])) {
      out->MutableData<std::string>()[i] = "NaN";
    } else if (std::is_floating_point<SrcType>::value && std::isinf(in->Data<SrcType>()[i])) {
      if (in->Data<SrcType>()[i] < std::numeric_limits<SrcType>::lowest()) {
        out->MutableData<std::string>()[i] = "-INF";
      } else {
        out->MutableData<std::string>()[i] = "INF";
      }
    } else {
      std::ostringstream convert;
      convert << in->Data<SrcType>()[i];
      out->MutableData<std::string>()[i] = convert.str();
    }
  }
}

template <typename DstType>
inline void CastFromStringData(const Tensor* in, Tensor* out, const TensorShape& shape) {
  if (std::is_same<DstType, std::string>::value) return;
  const int64_t len = shape.Size();
  ORT_ENFORCE(len > 0);
  if (std::is_same<DstType, float>::value) {
    float* mutable_data = out->MutableData<float>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stof(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, double>::value) {
    double* mutable_data = out->MutableData<double>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stod(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, int8_t>::value) {
    int8_t* mutable_data = out->MutableData<int8_t>();
    for (int i = 0; i < len; ++i) {
      int temp_i = std::stoi(in->Data<std::string>()[i]);
      mutable_data[i] = static_cast<int8_t>(temp_i);
    }
  } else if (std::is_same<DstType, uint8_t>::value) {
    uint8_t* mutable_data = out->MutableData<uint8_t>();
    for (int i = 0; i < len; ++i) {
      unsigned long temp_ui = std::stoul(in->Data<std::string>()[i]);
      mutable_data[i] = static_cast<uint8_t>(temp_ui);
    }
  } else if (std::is_same<DstType, int16_t>::value) {
    int16_t* mutable_data = out->MutableData<int16_t>();
    for (int i = 0; i < len; ++i) {
      int temp_i = std::stoi(in->Data<std::string>()[i]);
      mutable_data[i] = static_cast<int16_t>(temp_i);
    }
  } else if (std::is_same<DstType, uint16_t>::value) {
    uint16_t* mutable_data = out->MutableData<uint16_t>();
    for (int i = 0; i < len; ++i) {
      unsigned long temp_ui = std::stoul(in->Data<std::string>()[i]);
      mutable_data[i] = static_cast<uint16_t>(temp_ui);
    }
  } else if (std::is_same<DstType, int32_t>::value) {
    int32_t* mutable_data = out->MutableData<int32_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stol(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, uint32_t>::value) {
    uint32_t* mutable_data = out->MutableData<uint32_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stoul(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, int64_t>::value) {
    int64_t* mutable_data = out->MutableData<int64_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stoll(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, uint64_t>::value) {
    uint64_t* mutable_data = out->MutableData<uint64_t>();
    for (int i = 0; i < len; ++i) {
      mutable_data[i] = std::stoull(in->Data<std::string>()[i]);
    }
  } else {
    ORT_THROW("Unsupported type in cast op: from String to ", typeid(DstType).name());
  }
}  // namespace onnxruntime

template <typename T>
class Cast final : public OpKernel {
 public:
  Cast(const OpKernelInfo& info) : OpKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename SrcType,
            typename DstType>
  void CastData(const Tensor* in, Tensor* out, const TensorShape& shape) const {
    ::onnxruntime::CastData<SrcType, DstType>(in, out, shape);
  }

  template <typename SrcType,
            typename DstType>
  Status CastFloat16Data(const Tensor* in, Tensor* out, const TensorShape& shape, OpKernelContext* context) const {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    ::onnxruntime::CastFloat16Data<SrcType, DstType>(in, out, shape, allocator);
    return Status::OK();
  }

  template <typename SrcType>
  Status CastToStringData(const Tensor* in, Tensor* out, const TensorShape& shape) const {
    ::onnxruntime::CastToStringData<SrcType>(in, out, shape);
    return Status::OK();
  }

  template <typename DstType>
  Status CastFromStringData(const Tensor* in, Tensor* out, const TensorShape& shape) const {
    ::onnxruntime::CastFromStringData<DstType>(in, out, shape);
    return Status::OK();
  }

  ONNX_NAMESPACE::TensorProto_DataType to_;
};


const std::vector<MLDataType> castOpTypeConstraints{
    DataTypeImpl::GetTensorType<bool>(),
    DataTypeImpl::GetTensorType<float>(),
    DataTypeImpl::GetTensorType<double>(),
    DataTypeImpl::GetTensorType<uint8_t>(),
    DataTypeImpl::GetTensorType<uint16_t>(),
    DataTypeImpl::GetTensorType<uint32_t>(),
    DataTypeImpl::GetTensorType<uint64_t>(),
    DataTypeImpl::GetTensorType<int8_t>(),
    DataTypeImpl::GetTensorType<int16_t>(),
    DataTypeImpl::GetTensorType<int32_t>(),
    DataTypeImpl::GetTensorType<int64_t>(),
    DataTypeImpl::GetTensorType<MLFloat16>(),
    DataTypeImpl::GetTensorType<std::string>()};

#define ADD_FROM_CAST_OP(in_type)                                                                                                  \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                                                                        \
      Cast,                                                                                                                        \
      6,                                                                                                                           \
      9,                                                                                                                           \
      in_type,                                                                                                                     \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", castOpTypeConstraints), \
      Cast<in_type>);                                                                                                              \
                                                                                                                                   \
  template <>                                                                                                                      \
  Status Cast<in_type>::Compute(OpKernelContext* context) const {                                                                  \
    const Tensor* X = context->Input<Tensor>(0);                                                                                   \
    const TensorShape& shape = X->Shape();                                                                                         \
    Tensor* Y = context->Output(0, TensorShape(shape));                                                                            \
                                                                                                                                   \
    switch (to_) {                                                                                                                 \
      case TensorProto_DataType_BOOL:                                                                                              \
        CastData<in_type, bool>(X, Y, shape);                                                                                      \
        break;                                                                                                                     \
      case TensorProto_DataType_INT16:                                                                                             \
        CastData<in_type, int16_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_INT32:                                                                                             \
        CastData<in_type, int32_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_INT64:                                                                                             \
        CastData<in_type, int64_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT8:                                                                                             \
        CastData<in_type, uint8_t>(X, Y, shape);                                                                                   \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT16:                                                                                            \
        CastData<in_type, uint16_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT32:                                                                                            \
        CastData<in_type, uint32_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_UINT64:                                                                                            \
        CastData<in_type, uint64_t>(X, Y, shape);                                                                                  \
        break;                                                                                                                     \
      case TensorProto_DataType_FLOAT:                                                                                             \
        CastData<in_type, float>(X, Y, shape);                                                                                     \
        break;                                                                                                                     \
      case TensorProto_DataType_DOUBLE:                                                                                            \
        CastData<in_type, double>(X, Y, shape);                                                                                    \
        break;                                                                                                                     \
      case TensorProto_DataType_INT8:                                                                                              \
        CastData<in_type, int8_t>(X, Y, shape);                                                                                    \
        break;                                                                                                                     \
      case TensorProto_DataType_FLOAT16:                                                                                           \
        if (std::is_same<in_type, float>::value) {                                                                                 \
          CastData<float, MLFloat16>(X, Y, shape);                                                                                 \
        } else {                                                                                                                   \
          auto st = CastFloat16Data<in_type, MLFloat16>(X, Y, shape, context);                                                     \
          if (!st.IsOK()) return st;                                                                                               \
        }                                                                                                                          \
        break;                                                                                                                     \
      case TensorProto_DataType_STRING:                                                                                            \
        CastToStringData<in_type>(X, Y, shape);                                                                                    \
        break;                                                                                                                     \
      case TensorProto_DataType_UNDEFINED:                                                                                         \
        ORT_THROW("Cast op must have 'to' argument of type DataType"); /*break;*/                                                  \
      default:                                                                                                                     \
        ORT_THROW("Unexpected 'to' argument value: ", to_);                                                                        \
    }                                                                                                                              \
    return Status::OK();                                                                                                           \
  }

ADD_FROM_CAST_OP(uint8_t);
ADD_FROM_CAST_OP(uint16_t);
ADD_FROM_CAST_OP(uint32_t);
ADD_FROM_CAST_OP(uint64_t);
ADD_FROM_CAST_OP(int8_t);
ADD_FROM_CAST_OP(int16_t);
ADD_FROM_CAST_OP(int32_t);
ADD_FROM_CAST_OP(int64_t);
ADD_FROM_CAST_OP(bool);
ADD_FROM_CAST_OP(float);
ADD_FROM_CAST_OP(double);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Cast,
    6,
    9,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>()).TypeConstraint("T2", castOpTypeConstraints),
    Cast<MLFloat16>);

template <>
Status Cast<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, TensorShape(shape));
  Status st;
  switch (to_) {
    case TensorProto_DataType_BOOL:
      st = CastFloat16Data<MLFloat16, bool>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT16:
      st = CastFloat16Data<MLFloat16, int16_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT32:
      st = CastFloat16Data<MLFloat16, int32_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT64:
      st = CastFloat16Data<MLFloat16, int64_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT8:
      st = CastFloat16Data<MLFloat16, uint8_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT16:
      st = CastFloat16Data<MLFloat16, uint16_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT32:
      st = CastFloat16Data<MLFloat16, uint32_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_UINT64:
      st = CastFloat16Data<MLFloat16, uint64_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_FLOAT:
      CastData<MLFloat16, float>(X, Y, shape);
      break;
    case TensorProto_DataType_FLOAT16: {
      auto X_type = X->DataType();
      const void* source = X->DataRaw(X_type);
      void* target = Y->MutableDataRaw(X_type);
      // if source and target pointers are not equal, we need to copy the data.
      if (target != source) {
        memcpy(target, source, shape.Size() * X_type->Size());
      }
      st = Status::OK();
      break;
    }
    case TensorProto_DataType_DOUBLE:
      st = CastFloat16Data<MLFloat16, double>(X, Y, shape, context);
      break;
    case TensorProto_DataType_INT8:
      st = CastFloat16Data<MLFloat16, int8_t>(X, Y, shape, context);
      break;
    case TensorProto_DataType_STRING:
      ORT_THROW("Casting to and from strings is not supported yet."); /*break;*/
    case TensorProto_DataType_UNDEFINED:
      ORT_THROW("Cast op must have 'to' argument of type DataType"); /*break;*/
    default:
      ORT_THROW("Unexpected 'to' argument value: ", to_);
  }
  return st;
}

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Cast,
    9,
    string,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>()).TypeConstraint("T2", castOpTypeConstraints),
    Cast<std::string>);

template <>
Status Cast<std::string>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL,
                                  "Input is missing. The operator Cast expects one and only one input");
  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, TensorShape(shape));
  Status st;
  switch (to_) {
    case TensorProto_DataType_INT16:
      st = CastFromStringData<int16_t>(X, Y, shape);
      break;
    case TensorProto_DataType_INT32:
      st = CastFromStringData<int32_t>(X, Y, shape);
      break;
    case TensorProto_DataType_INT64:
      st = CastFromStringData<int64_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT8:
      st = CastFromStringData<uint8_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT16:
      st = CastFromStringData<uint16_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT32:
      st = CastFromStringData<uint32_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UINT64:
      st = CastFromStringData<uint64_t>(X, Y, shape);
      break;
    case TensorProto_DataType_FLOAT:
      st = CastFromStringData<float>(X, Y, shape);
      break;
    case TensorProto_DataType_DOUBLE:
      st = CastFromStringData<double>(X, Y, shape);
      break;
    case TensorProto_DataType_INT8:
      st = CastFromStringData<int8_t>(X, Y, shape);
      break;
    case TensorProto_DataType_UNDEFINED:
      ORT_THROW("Cast op must have 'to' argument of type DataType");
    default:
      ORT_THROW("Unexpected 'to' argument value: ", to_);
  }
  return st;
}
}  //namespace onnxruntime
