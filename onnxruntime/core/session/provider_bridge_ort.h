// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

struct ProviderInfo_OpenVINO;
ProviderInfo_OpenVINO* TryGetProviderInfo_OpenVINO();

bool InitProvidersSharedLibrary();
void UnloadSharedProviders();

}  // namespace onnxruntime
