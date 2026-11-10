// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/model_package_api.h"
#include "core/session/ort_apis.h"

#if defined(ORT_ENABLE_STATEFUL)
#include "stateful/bridge/stateful_bridge.h"
#endif

ORT_API(const OrtStatefulApi*, OrtApis::GetStatefulApi) {
#if defined(ORT_ENABLE_STATEFUL)
  return OrtStatefulAPI::GetStatefulApiImpl();
#else
  return nullptr;
#endif
}

ORT_API_STATUS_IMPL(OrtModelPackageAPI::CreateSessionChain, const OrtEnv* env,
                    const OrtModelPackageContext* context, const OrtModelPackageChainConfig* config,
                    OrtSessionChain** out) {
  API_IMPL_BEGIN
#if defined(ORT_ENABLE_STATEFUL)
  return CreateSessionChainImpl(env, context, config, out);
#else
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(config);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Stateful execution is not enabled in this build");
#endif
  API_IMPL_END
}
