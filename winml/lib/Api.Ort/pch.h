// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "winrt_headers.h"

#include "core/providers/winml/winml_provider_factory.h"
#include "adapter/winml_adapter_c_api.h"

using UniqueOrtModel = std::unique_ptr<OrtModel, void (*)(OrtModel*)>;
using UniqueOrtSessionOptions = std::unique_ptr<OrtSessionOptions, void (*)(OrtSessionOptions*)>;
using UniqueOrtSession = std::unique_ptr<OrtSession, void (*)(OrtSession*)>;
using UniqueOrtExecutionProvider = std::unique_ptr<OrtExecutionProvider, void (*)(OrtExecutionProvider*)>;
using UniqueOrtValue = std::unique_ptr<OrtValue, void (*)(OrtValue*)>;
using UniqueOrtMemoryInfo = std::unique_ptr<OrtMemoryInfo, void (*)(OrtMemoryInfo*)>;
using UniqueOrtTypeInfo = std::unique_ptr<OrtTypeInfo, void (*)(OrtTypeInfo*)>;
using UniqueOrtTensorTypeAndShapeInfo = std::unique_ptr<OrtTensorTypeAndShapeInfo, void (*)(OrtTensorTypeAndShapeInfo*)>;
using UniqueOrtAllocator = std::unique_ptr<OrtAllocator, OrtStatus* (*)(OrtAllocator*)>;
using UniqueOrtRunOptions = std::unique_ptr<OrtRunOptions, void (*)(OrtRunOptions*)>;
