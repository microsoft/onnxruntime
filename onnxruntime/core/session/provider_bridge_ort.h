// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct ProviderInfo_OpenVINO;

namespace onnxruntime {

ProviderInfo_OpenVINO* TryGetProviderInfo_OpenVINO();

bool InitProvidersSharedLibrary();
void UnloadSharedProviders();

}  // namespace onnxruntime
