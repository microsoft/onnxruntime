// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace OrtStatefulAPI {
ORT_API(const OrtStatefulApi*, GetStatefulApiImpl);
}

namespace OrtModelPackageAPI {
ORT_API_STATUS_IMPL(CreateSessionChainImpl, const OrtEnv* env, const OrtModelPackageContext* context,
                    const OrtModelPackageChainConfig* config, OrtSessionChain** out);
}
