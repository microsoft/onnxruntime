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
  assert(!(code == 0 && msg != nullptr));
  size_t clen = strlen(msg);
  OrtStatus* p = reinterpret_cast<OrtStatus*>(new char[sizeof(OrtStatus) + clen]);
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
  OrtStatus* p = reinterpret_cast<OrtStatus*>(new char[sizeof(OrtStatus) + clen]);
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

ORT_API(void, OrtReleaseStatus, _Frees_ptr_opt_ OrtStatus* value) { delete[] reinterpret_cast<char*>(value); }
