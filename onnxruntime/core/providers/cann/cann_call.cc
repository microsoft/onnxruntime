// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <string.h>
#include "core/providers/shared_library/provider_api.h"
#include "cann_call.h"

namespace onnxruntime {

template <typename ERRTYPE>
const char* CannErrString(ERRTYPE x) {
  ORT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* CannErrString<aclError>(aclError e) {
  aclrtSynchronizeDevice();

  switch (e) {
    CASE_ENUM_TO_STR(ACL_SUCCESS);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_PARAM);
    CASE_ENUM_TO_STR(ACL_ERROR_UNINITIALIZE);
    CASE_ENUM_TO_STR(ACL_ERROR_REPEAT_INITIALIZE);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_FILE);
    CASE_ENUM_TO_STR(ACL_ERROR_WRITE_FILE);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_FILE_SIZE);
    CASE_ENUM_TO_STR(ACL_ERROR_PARSE_FILE);
    CASE_ENUM_TO_STR(ACL_ERROR_FILE_MISSING_ATTR);
    CASE_ENUM_TO_STR(ACL_ERROR_FILE_ATTR_INVALID);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_DUMP_CONFIG);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_PROFILING_CONFIG);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_MODEL_ID);
    CASE_ENUM_TO_STR(ACL_ERROR_DESERIALIZE_MODEL);
    CASE_ENUM_TO_STR(ACL_ERROR_PARSE_MODEL);
    CASE_ENUM_TO_STR(ACL_ERROR_READ_MODEL_FAILURE);
    CASE_ENUM_TO_STR(ACL_ERROR_MODEL_SIZE_INVALID);
    CASE_ENUM_TO_STR(ACL_ERROR_MODEL_MISSING_ATTR);
    CASE_ENUM_TO_STR(ACL_ERROR_MODEL_INPUT_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_MODEL_OUTPUT_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_MODEL_NOT_DYNAMIC);
    CASE_ENUM_TO_STR(ACL_ERROR_OP_TYPE_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_OP_INPUT_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_OP_OUTPUT_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_OP_ATTR_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_OP_NOT_FOUND);
    CASE_ENUM_TO_STR(ACL_ERROR_OP_LOAD_FAILED);
    CASE_ENUM_TO_STR(ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    CASE_ENUM_TO_STR(ACL_ERROR_FORMAT_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED);
    CASE_ENUM_TO_STR(ACL_ERROR_KERNEL_NOT_FOUND);
    CASE_ENUM_TO_STR(ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED);
    CASE_ENUM_TO_STR(ACL_ERROR_KERNEL_ALREADY_REGISTERED);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_QUEUE_ID);
    CASE_ENUM_TO_STR(ACL_ERROR_REPEAT_SUBSCRIBE);
    CASE_ENUM_TO_STR(ACL_ERROR_STREAM_NOT_SUBSCRIBE);
    CASE_ENUM_TO_STR(ACL_ERROR_THREAD_NOT_SUBSCRIBE);
    CASE_ENUM_TO_STR(ACL_ERROR_WAIT_CALLBACK_TIMEOUT);
    CASE_ENUM_TO_STR(ACL_ERROR_REPEAT_FINALIZE);
    CASE_ENUM_TO_STR(ACL_ERROR_NOT_STATIC_AIPP);
    CASE_ENUM_TO_STR(ACL_ERROR_COMPILING_STUB_MODE);
    CASE_ENUM_TO_STR(ACL_ERROR_GROUP_NOT_SET);
    CASE_ENUM_TO_STR(ACL_ERROR_GROUP_NOT_CREATE);
    CASE_ENUM_TO_STR(ACL_ERROR_PROF_ALREADY_RUN);
    CASE_ENUM_TO_STR(ACL_ERROR_PROF_NOT_RUN);
    CASE_ENUM_TO_STR(ACL_ERROR_DUMP_ALREADY_RUN);
    CASE_ENUM_TO_STR(ACL_ERROR_DUMP_NOT_RUN);
    CASE_ENUM_TO_STR(ACL_ERROR_PROF_REPEAT_SUBSCRIBE);
    CASE_ENUM_TO_STR(ACL_ERROR_PROF_API_CONFLICT);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_MAX_OPQUEUE_NUM_CONFIG);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_OPP_PATH);
    CASE_ENUM_TO_STR(ACL_ERROR_OP_UNSUPPORTED_DYNAMIC);
    CASE_ENUM_TO_STR(ACL_ERROR_RELATIVE_RESOURCE_NOT_CLEARED);
    CASE_ENUM_TO_STR(ACL_ERROR_UNSUPPORTED_JPEG);
    CASE_ENUM_TO_STR(ACL_ERROR_BAD_ALLOC);
    CASE_ENUM_TO_STR(ACL_ERROR_API_NOT_SUPPORT);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_DEVICE);
    CASE_ENUM_TO_STR(ACL_ERROR_MEMORY_ADDRESS_UNALIGNED);
    CASE_ENUM_TO_STR(ACL_ERROR_RESOURCE_NOT_MATCH);
    CASE_ENUM_TO_STR(ACL_ERROR_INVALID_RESOURCE_HANDLE);
    CASE_ENUM_TO_STR(ACL_ERROR_FEATURE_UNSUPPORTED);
    CASE_ENUM_TO_STR(ACL_ERROR_PROF_MODULES_UNSUPPORTED);
    CASE_ENUM_TO_STR(ACL_ERROR_STORAGE_OVER_LIMIT);
    CASE_ENUM_TO_STR(ACL_ERROR_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(ACL_ERROR_FAILURE);
    CASE_ENUM_TO_STR(ACL_ERROR_GE_FAILURE);
    CASE_ENUM_TO_STR(ACL_ERROR_RT_FAILURE);
    CASE_ENUM_TO_STR(ACL_ERROR_DRV_FAILURE);
    CASE_ENUM_TO_STR(ACL_ERROR_PROFILING_FAILURE);

    default:
      return "(look for ACL_ERROR_xxx in acl.h)";
  }
}

template <>
const char* CannErrString<ge::graphStatus>(ge::graphStatus e) {
  using namespace ge;

  aclrtSynchronizeDevice();

  switch (e) {
    CASE_ENUM_TO_STR(GRAPH_FAILED);
    CASE_ENUM_TO_STR(GRAPH_SUCCESS);
    CASE_ENUM_TO_STR(GRAPH_NOT_CHANGED);
    CASE_ENUM_TO_STR(GRAPH_PARAM_INVALID);
    CASE_ENUM_TO_STR(GRAPH_NODE_WITHOUT_CONST_INPUT);
    CASE_ENUM_TO_STR(GRAPH_NODE_NEED_REPASS);
    CASE_ENUM_TO_STR(GRAPH_INVALID_IR_DEF);
    CASE_ENUM_TO_STR(OP_WITHOUT_IR_DATATYPE_INFER_RULE);

    default:
      return "(look for graphStatus in ge_error_codes.h)";
  }
}

template <typename ERRTYPE, bool THRW>
bool CannCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg) {
  if (retCode != successCode) {
    try {
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        snprintf(hostname, HOST_NAME_MAX, "%s", "?");
      int currentCannDevice;
      (void)aclrtGetDevice(&currentCannDevice);
      (void)aclGetRecentErrMsg();
      static char str[1024];
      snprintf(str, sizeof(str), "%s failure %d: %s ; NPU=%d ; hostname=%s ; expr=%s; %s",
               libName, static_cast<int>(retCode), CannErrString(retCode), currentCannDevice,
               hostname,
               exprString, msg);
      if (THRW) {
        ORT_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
      }
    } catch (const std::exception& e) {
      if (THRW) {
        ORT_THROW(e.what());
      } else {
        LOGS_DEFAULT(ERROR) << e.what();
      }
    }
    return false;
  }
  return true;
}

template bool CannCall<aclError, false>(aclError retCode, const char* exprString, const char* libName,
                                        aclError successCode, const char* msg);
template bool CannCall<aclError, true>(aclError retCode, const char* exprString, const char* libName,
                                       aclError successCode, const char* msg);

template bool CannCall<ge::graphStatus, false>(ge::graphStatus retCode, const char* exprString, const char* libName,
                                               ge::graphStatus successCode, const char* msg);
template bool CannCall<ge::graphStatus, true>(ge::graphStatus retCode, const char* exprString, const char* libName,
                                              ge::graphStatus successCode, const char* msg);

}  // namespace onnxruntime
