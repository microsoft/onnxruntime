// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "custom_op_library.h"

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* /* options */, const OrtApiBase* api) {
  const OrtApi* ort_api = api->GetApi(ORT_API_VERSION);
  return ort_api->CreateStatus(ORT_FAIL, "Failure from custom op library's RegisterCustomOps()");
}
