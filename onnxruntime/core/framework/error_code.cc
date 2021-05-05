// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/framework/error_code_helper.h"
#include <cassert>
using onnxruntime::common::Status;

struct OrtStatus {
  OrtErrorCode code;
  char msg[1];  // a null-terminated string
};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 28196)
#pragma warning(disable : 6387)
#endif
//Even we say it may not return NULL, indeed it may.
_Check_return_ _Ret_notnull_ OrtStatus* ORT_API_CALL OrtApis::CreateStatus(OrtErrorCode code,
                                                                           _In_z_ const char* msg) NO_EXCEPTION {
  assert(!(code == 0 && msg != nullptr));
  SafeInt<size_t> clen(nullptr == msg ? 0 : strnlen(msg, 65535));
  OrtStatus* p = reinterpret_cast<OrtStatus*>(::malloc(sizeof(OrtStatus) + clen));
  if (p == nullptr) return nullptr;  // OOM. What we can do here? abort()?
  p->code = code;
  memcpy(p->msg, msg, clen);
  p->msg[clen] = '\0';
  return p;
}

namespace onnxruntime {
_Ret_notnull_ OrtStatus* ToOrtStatus(const Status& st) {
  if (st.IsOK())
    return nullptr;
  SafeInt<size_t> clen(st.ErrorMessage().length());
  OrtStatus* p = reinterpret_cast<OrtStatus*>(::malloc(sizeof(OrtStatus) + clen));
  if (p == nullptr)
    return nullptr;
  p->code = static_cast<OrtErrorCode>(st.Code());
  memcpy(p->msg, st.ErrorMessage().c_str(), clen);
  p->msg[clen] = '\0';
  return p;
}
}  // namespace onnxruntime
#ifdef _MSC_VER
#pragma warning(pop)
#endif
ORT_API(OrtErrorCode, OrtApis::GetErrorCode, _In_ const OrtStatus* status) {
  return status->code;
}

ORT_API(const char*, OrtApis::GetErrorMessage, _In_ const OrtStatus* status) {
  return status->msg;
}

ORT_API(void, OrtApis::ReleaseStatus, _Frees_ptr_opt_ OrtStatus* value) { ::free(value); }
