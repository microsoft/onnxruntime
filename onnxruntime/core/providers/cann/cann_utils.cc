// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

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

}  // namespace cann
}  // namespace onnxruntime
