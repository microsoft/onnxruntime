// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <unistd.h>

#include "core/providers/cann/cann_utils.h"

namespace onnxruntime {
namespace cann {

template <typename T>
aclDataType getACLType() {
  return ACL_DT_UNDEFINED;
}

#define GET_ACL_TYPE(onnx_type, cann_type) \
  template <>                              \
  aclDataType getACLType<onnx_type>() {    \
    return cann_type;                      \
  }

GET_ACL_TYPE(int8_t, ACL_INT8);
GET_ACL_TYPE(int16_t, ACL_INT16);
GET_ACL_TYPE(int32_t, ACL_INT32);
GET_ACL_TYPE(int64_t, ACL_INT64);
GET_ACL_TYPE(uint8_t, ACL_UINT8);
GET_ACL_TYPE(uint16_t, ACL_UINT16);
GET_ACL_TYPE(uint32_t, ACL_UINT32);
GET_ACL_TYPE(uint64_t, ACL_UINT64);
GET_ACL_TYPE(float, ACL_FLOAT);
GET_ACL_TYPE(MLFloat16, ACL_FLOAT16);
GET_ACL_TYPE(BFloat16, ACL_BF16);
GET_ACL_TYPE(double, ACL_DOUBLE);
GET_ACL_TYPE(bool, ACL_BOOL);

template <typename T>
Status Fill(Tensor* y, void* addr, aclrtStream stream) {
  int64_t one = 1;
  int64_t dims = static_cast<int64_t>(y->Shape().NumDimensions());

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, ACL_INT64, 1, &dims, format);
    CANN_CONST_INPUTDESC(prepare, 0, const_cast<int64_t*>(y->Shape().GetDims().data()), dims * sizeof(int64_t));
    CANN_PREPARE_INPUTDESC(prepare, aclType, 1, &one, format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, dims, y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<int64_t*>(y->Shape().GetDims().data()), dims * sizeof(int64_t));
    CANN_PREPARE_INPUTBUFFER(prepare, addr, sizeof(T));
    CANN_PREPARE_OUTPUTBUFFER(prepare, y->MutableDataRaw(), y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("Fill",
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              stream));

  return Status::OK();
}

template Status Fill<uint8_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<uint16_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<uint32_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<uint64_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int8_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int16_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int32_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int64_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<MLFloat16>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<float>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<double>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<bool>(Tensor* y, void* addr, aclrtStream stream);

template <typename T>
Status Broadcast(const Tensor* x, Tensor* y, void* addr, aclrtStream stream) {
  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "shape", static_cast<int64_t>(y->Shape().NumDimensions()),
                                           y->Shape().GetDims().data()));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, x->Shape().NumDimensions(), x->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, y->Shape().NumDimensions(), y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(x->DataRaw()), x->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, addr, y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("BroadcastToD",
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              stream));

  return Status::OK();
}

template Status Broadcast<uint8_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<uint16_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<uint32_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<uint64_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int8_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int16_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int32_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int64_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<MLFloat16>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<BFloat16>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<float>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<double>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<bool>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);

Status aclrtblasGemmEx(aclTransType transA,
                       aclTransType transB,
                       aclTransType transC,
                       int m,
                       int n,
                       int k,
                       const void* alpha,
                       const void* matrixA,
                       int lda,
                       aclDataType dataTypeA,
                       const void* matrixB,
                       int ldb,
                       aclDataType dataTypeB,
                       const void* beta,
                       void* matrixC,
                       int ldc,
                       aclDataType dataTypeC,
                       aclComputeType type,
                       aclrtStream stream) {
  ORT_UNUSED_PARAMETER(transC);
  ORT_UNUSED_PARAMETER(lda);
  ORT_UNUSED_PARAMETER(ldb);
  ORT_UNUSED_PARAMETER(ldc);
  ORT_UNUSED_PARAMETER(type);

  TensorShape A;
  TensorShape B;
  TensorShape C{m, n};
  TensorShape shape{1};

  A = transA ? TensorShape{k, m} : TensorShape{m, k};
  B = transB ? TensorShape{n, k} : TensorShape{k, n};

  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_a", transA));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_b", transB));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, dataTypeA, A.NumDimensions(), A.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeB, B.NumDimensions(), B.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeC, C.NumDimensions(), C.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeC, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeC, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, dataTypeC, C.NumDimensions(), C.GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(matrixA), A.Size() * aclDataTypeSize(dataTypeA));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(matrixB), B.Size() * aclDataTypeSize(dataTypeB));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(matrixC), C.Size() * aclDataTypeSize(dataTypeC));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(alpha), aclDataTypeSize(dataTypeC));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(beta), aclDataTypeSize(dataTypeC));
    CANN_PREPARE_OUTPUTBUFFER(prepare, const_cast<void*>(matrixC), C.Size() * aclDataTypeSize(dataTypeC));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("GEMM",
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              stream));

  return Status::OK();
}

bool FileExist(const std::string& file_name) {
  return (access(file_name.c_str(), F_OK) != -1);
}

void GenerateHashValue(const std::string string, HashValue& hash_value) {
  uint32_t hash[4] = {0, 0, 0, 0};
  MurmurHash3::x86_128(string.data(), gsl::narrow_cast<int32_t>(string.size()), hash[0], &hash);
  hash_value = hash[0] | (uint64_t(hash[1]) << 32);
}

}  // namespace cann
}  // namespace onnxruntime
