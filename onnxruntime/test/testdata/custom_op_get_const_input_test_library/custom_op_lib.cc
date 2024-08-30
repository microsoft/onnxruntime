// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <mutex>
#include <vector>

#include "custom_op_lib.h"
#include "custom_op.h"

static const char* c_OpDomain = "test.customop";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  // Allow use of Ort::GetApi() in C++ API implementations.
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  Ort::UnownedSessionOptions session_options(options);

  static TestCustomOp c_CustomOp;

  OrtStatus* result = nullptr;

#ifndef ORT_NO_EXCEPTIONS
  try {
#endif
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_CustomOp);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));

#ifndef ORT_NO_EXCEPTIONS
  } catch (const std::exception& e) {
    Ort::Status status{e};
    result = status.release();
  }
#endif

  return result;
}
