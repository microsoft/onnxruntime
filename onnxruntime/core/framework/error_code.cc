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

ORT_API(OrtStatus*, OrtCreateStatus, OrtErrorCode code, _In_ const char* msg) {
#ifndef NDEBUG
  assert(code != ORT_OK);
  //Don't check if msg is NULL at here, because it will generate a useless compiler warning
  //assert(msg != nullptr);
  size_t clen = strlen(msg);
#else
  //@snnn: after the discussion with Ryan and Scott, we decide to treat the last arugment(msg)
  // of this function as optional even we mark it as required.
  size_t clen = msg == nullptr ? 0 : strlen(msg);
#endif
  OrtStatus* p = reinterpret_cast<OrtStatus*>(::malloc(sizeof(OrtStatus) + clen));
  if (p == nullptr) return nullptr;  // OOM
  p->code = code;
  if (clen != 0) memcpy(p->msg, msg, clen);
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
