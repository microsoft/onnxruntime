// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/common/status.h"
#include "core/framework/error_code_helper.h"
#include <cassert>
using onnxruntime::common::Status;

struct OrtStatus {
  OrtErrorCode code;
  char msg[1];  // a null-terminated string
};

//Even we say it may not return NULL, indeed it may.
ORT_EXPORT _Check_return_ _Ret_notnull_ OrtStatus* ORT_API_CALL OrtCreateStatus(OrtErrorCode code, _In_ const char* msg) NO_EXCEPTION {
  assert(!(code == 0 && msg != nullptr));
  size_t clen = strlen(msg);
  OrtStatus* p = reinterpret_cast<OrtStatus*>(::malloc(sizeof(OrtStatus) + clen));
  if (p == nullptr) return nullptr;  // OOM. What we can do here? abort()?
  p->code = code;
  memcpy(p->msg, msg, clen);
  p->msg[clen] = '\0';
  return p;
}

namespace onnxruntime {
OrtStatus* ToOrtStatus(const Status& st) {
  if (st.IsOK())
    return nullptr;
  size_t clen = st.ErrorMessage().length();
  OrtStatus* p = reinterpret_cast<OrtStatus*>(::malloc(sizeof(OrtStatus) + clen));
  p->code = static_cast<OrtErrorCode>(st.Code());
  memcpy(p->msg, st.ErrorMessage().c_str(), clen);
  p->msg[clen] = '\0';
  return p;
}
}  // namespace onnxruntime
ORT_API(OrtErrorCode, OrtGetErrorCode, _In_ const OrtStatus* status) {
  return status->code;
}

ORT_API(const char*, OrtGetErrorMessage, _In_ const OrtStatus* status) {
  return status->msg;
}

ORT_API(void, OrtReleaseStatus, _Frees_ptr_opt_ OrtStatus* value) { ::free(value); }
