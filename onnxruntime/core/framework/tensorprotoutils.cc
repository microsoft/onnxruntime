// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"

#include <memory>
#include "core/graph/onnx_protobuf.h"
#include "core/common/logging/logging.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorutils.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value_patterns_planner.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace utils {
std::vector<int64_t> GetTensorShapeFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<int64_t> tensor_shape_vec(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return tensor_shape_vec;
}

std::vector<int64_t> GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto) {
  const auto& dims = tensor_shape_proto.dim();
  std::vector<int64_t> tensor_shape_vec(dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i].has_dim_param()
                              ? -1 /* symbolic dimensions are represented as -1 in onnxruntime*/
                              : dims[i].dim_value();
  }
  return tensor_shape_vec;
}

template <typename T>
common::Status GetTensorByTypeFromTensorProto(const TensorProto& tensor_proto,
                                              const TensorShape& tensor_shape,
                                              std::unique_ptr<Tensor>* p_tensor,
                                              AllocatorPtr alloc,
                                              void* preallocated,
                                              size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  //tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
  if (tensor_size < 0) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid shape ", tensor_shape);
  }
  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<256>(static_cast<size_t>(tensor_size), sizeof(T), &size_to_allocate)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
  }

  if (preallocated && preallocated_size != size_to_allocate)
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "The buffer planner is not consistent with tensor buffer size, expected ", size_to_allocate, ", got ", preallocated_size);
  //TODO(): size_to_allocate could be zero. We shouldn't pass zero to alloc->Alloc()
  T* p_data = static_cast<T*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  ONNXRUNTIME_RETURN_IF_ERROR(::onnxruntime::utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  return common::Status::OK();
}

template <>
common::Status GetTensorByTypeFromTensorProto<std::string>(const TensorProto& tensor_proto,
                                                           const TensorShape& tensor_shape,
                                                           std::unique_ptr<Tensor>* p_tensor,
                                                           AllocatorPtr alloc,
                                                           void* preallocated,
                                                           size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<256>(static_cast<size_t>(tensor_size), sizeof(std::string), &size_to_allocate)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
  }

  if (preallocated && preallocated_size != size_to_allocate)
    return Status(ONNXRUNTIME, FAIL, "The buffer planner is not consistent with tensor buffer size");

  std::string* p_data = static_cast<std::string*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<std::string>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  /*
  In the case of string tensors, the strings need to be constructed in the pre-allocated memory (placement
  new) before calling Unpack (which copies the strings from the proto). Placement new happens inside the
  Tensor's constructor. Hence the order of invocation of Tensor construction and Unpack needs to be reversed
  in comparison to other types. This has the disadvantage of alloc/deallocing a Tensor if Unpack fails;
  however restricting it to string types only alleviates this concern for other types at least. Hence the template
  specialization for string.
  */
  ONNXRUNTIME_RETURN_IF_ERROR(::onnxruntime::utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));

  return common::Status::OK();
}

template <>
common::Status GetTensorByTypeFromTensorProto<MLFloat16>(const TensorProto& tensor_proto,
                                                         const TensorShape& tensor_shape,
                                                         std::unique_ptr<Tensor>* p_tensor,
                                                         AllocatorPtr alloc,
                                                         void* preallocated,
                                                         size_t preallocated_size) {
  int64_t tensor_size = tensor_shape.Size();
  if (tensor_size < 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
  }
  static_assert(sizeof(MLFloat16) == sizeof(uint16_t), "MLFloat16 must has 16 bit size");
  size_t size_to_allocate;
  if (!IAllocator::CalcMemSizeForArrayWithAlignment<256>(static_cast<size_t>(tensor_size), sizeof(MLFloat16), &size_to_allocate)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
  }

  if (preallocated && preallocated_size != size_to_allocate)
    return Status(ONNXRUNTIME, FAIL, "The buffer planner is not consistent with tensor buffer size");

  MLFloat16* p_data = static_cast<MLFloat16*>(preallocated ? preallocated : alloc->Alloc(size_to_allocate));
  ONNXRUNTIME_RETURN_IF_ERROR(::onnxruntime::utils::TensorUtils::UnpackTensor(tensor_proto, p_data, tensor_size));
  *p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<MLFloat16>(),
                                       tensor_shape,
                                       static_cast<void*>(p_data),
                                       alloc->Info(),
                                       preallocated ? nullptr : alloc);  // no deleter for preallocated

  return common::Status::OK();
}

Status TensorProtoToMLValue(const ONNX_NAMESPACE::TensorProto& input, AllocatorPtr allocator, void* preallocated,
                            size_t preallocated_size, MLValue& value) {
  std::unique_ptr<Tensor> p_tensor;
  ONNXRUNTIME_RETURN_IF_ERROR(GetTensorFromTensorProto(input, &p_tensor, allocator, preallocated, preallocated_size));
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

#define CASE_PROTO(X, Y)                                               \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X: \
    return GetTensorByTypeFromTensorProto<Y>(tensor_proto, tensor_shape, p_tensor, allocator, preallocated, preallocated_size);

common::Status GetTensorFromTensorProto(const TensorProto& tensor_proto,
                                        std::unique_ptr<Tensor>* p_tensor,
                                        AllocatorPtr allocator,
                                        void* preallocated,
                                        size_t preallocated_size) {
  std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
  TensorShape tensor_shape{tensor_shape_vec};
  switch (tensor_proto.data_type()) {
    CASE_PROTO(FLOAT, float);
    CASE_PROTO(DOUBLE, double);
    CASE_PROTO(BOOL, bool);
    CASE_PROTO(INT8, int8_t);
    CASE_PROTO(INT16, int16_t);
    CASE_PROTO(INT32, int32_t);
    CASE_PROTO(INT64, int64_t);
    CASE_PROTO(UINT8, uint8_t);
    CASE_PROTO(UINT16, uint16_t);
    CASE_PROTO(UINT32, uint32_t);
    CASE_PROTO(UINT64, uint64_t);
    CASE_PROTO(STRING, std::string);
    CASE_PROTO(FLOAT16, MLFloat16);
    default: {
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }
  }
}

TensorProto::DataType GetTensorProtoType(const Tensor& tensor) {
  auto tensor_type = tensor.DataType();
  TensorProto::DataType dtype = TensorProto_DataType_UNDEFINED;

  if (tensor_type == DataTypeImpl::GetType<float>())
    dtype = TensorProto_DataType_FLOAT;
  else if (tensor_type == DataTypeImpl::GetType<double>())
    dtype = TensorProto_DataType_DOUBLE;
  else if (tensor_type == DataTypeImpl::GetType<int8_t>())
    dtype = TensorProto_DataType_INT8;
  else if (tensor_type == DataTypeImpl::GetType<int16_t>())
    dtype = TensorProto_DataType_INT16;
  else if (tensor_type == DataTypeImpl::GetType<int32_t>())
    dtype = TensorProto_DataType_INT32;
  else if (tensor_type == DataTypeImpl::GetType<int64_t>())
    dtype = TensorProto_DataType_INT64;
  else if (tensor_type == DataTypeImpl::GetType<uint8_t>())
    dtype = TensorProto_DataType_UINT8;
  else if (tensor_type == DataTypeImpl::GetType<uint16_t>())
    dtype = TensorProto_DataType_UINT16;
  else if (tensor_type == DataTypeImpl::GetType<uint32_t>())
    dtype = TensorProto_DataType_UINT32;
  else if (tensor_type == DataTypeImpl::GetType<uint64_t>())
    dtype = TensorProto_DataType_UINT64;
  else if (tensor_type == DataTypeImpl::GetType<bool>())
    dtype = TensorProto_DataType_BOOL;

  return dtype;
}

}  // namespace utils
}  // namespace onnxruntime
