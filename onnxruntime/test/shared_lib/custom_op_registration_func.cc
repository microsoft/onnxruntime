// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "custom_op_utils.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static Ort::CustomOpDomain GetCustomOpDomain() {
  static const MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider};
  Ort::CustomOpDomain domain{"ort_unit_test"};
  domain.Add(&custom_op);

  return domain;
}

// exported registration function for testing OrtApi RegisterCustomOps
extern "C" EXPORT OrtStatus* ORT_API_CALL RegisterUnitTestCustomOps(OrtSessionOptions* options,
                                                                    const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  // need this to be static so it's initialized once and remains valid beyond the scope of this function
  static Ort::CustomOpDomain domain = GetCustomOpDomain();

  Ort::UnownedSessionOptions session_options(options);
  session_options.Add(domain);

  return nullptr;
}
