// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/arch/CUDA/Half.h"

#if defined(USE_MLAS) && defined(_M_AMD64)
#include "core/mlas/inc/mlas.h"
#endif

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
  Tensor tmp_tensor(DataTypeImpl::GetType<float>(), shape, buffer, allocator->Info(), nullptr);
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
    for (int i = 0; i < len; ++i) {
      out->MutableData<float>()[i] = std::stof(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, double>::value) {
    for (int i = 0; i < len; ++i) {
      out->MutableData<double>()[i] = std::stod(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, int8_t>::value) {
    for (int i = 0; i < len; ++i) {
      int temp_i = std::stoi(in->Data<std::string>()[i]);
      out->MutableData<int8_t>()[i] = static_cast<int8_t>(temp_i);
    }
  } else if (std::is_same<DstType, uint8_t>::value) {
    for (int i = 0; i < len; ++i) {
      unsigned long temp_ui = std::stoul(in->Data<std::string>()[i]);
      out->MutableData<uint8_t>()[i] = static_cast<uint8_t>(temp_ui);
    }
  } else if (std::is_same<DstType, int16_t>::value) {
    for (int i = 0; i < len; ++i) {
      int temp_i = std::stoi(in->Data<std::string>()[i]);
      out->MutableData<int16_t>()[i] = static_cast<int16_t>(temp_i);
    }
  } else if (std::is_same<DstType, uint16_t>::value) {
    for (int i = 0; i < len; ++i) {
      unsigned long temp_ui = std::stoul(in->Data<std::string>()[i]);
      out->MutableData<uint16_t>()[i] = static_cast<uint16_t>(temp_ui);
    }
  } else if (std::is_same<DstType, int32_t>::value) {
    for (int i = 0; i < len; ++i) {
      out->MutableData<int32_t>()[i] = std::stol(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, uint32_t>::value) {
    for (int i = 0; i < len; ++i) {
      out->MutableData<uint32_t>()[i] = std::stoul(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, int64_t>::value) {
    for (int i = 0; i < len; ++i) {
      out->MutableData<int64_t>()[i] = std::stoll(in->Data<std::string>()[i]);
    }
  } else if (std::is_same<DstType, uint64_t>::value) {
    for (int i = 0; i < len; ++i) {
      out->MutableData<uint64_t>()[i] = std::stoull(in->Data<std::string>()[i]);
    }
  } else {
    return;
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

}  //namespace onnxruntime
