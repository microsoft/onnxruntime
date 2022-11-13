// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <mutex>
#include <vector>

#include "custom_op_lib.h"
#include "openvino_wrapper.h"
#include "core/common/common.h"

static const char* c_OpDomain = "test.customop.ov";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {

  // Allow use of Ort::GetApi() and Ort::OpWrapper::GetApi() in C++ ORT api implementations.
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
  Ort::UnownedSessionOptions session_options(options);

  static CustomOpOpenVINO c_CustomOpOpenVINO(session_options.GetConst());

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_CustomOpOpenVINO);

    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));

  } ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }

  return result;
}
