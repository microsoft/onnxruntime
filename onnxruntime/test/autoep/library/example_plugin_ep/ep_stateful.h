// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../plugin_ep_utils.h"

OrtStatus* RegisterExampleStatefulProvider(OrtEp* ep, ApiPtrs apis, const OrtLogger& logger,
                                           const std::string& mode);
OrtStatus* CreateLegacyExampleEp23(ApiPtrs apis, const char* name, OrtEp** out);
void ReleaseLegacyExampleEp23(OrtEp* ep) noexcept;
