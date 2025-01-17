// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "adapter/winml_adapter_c_api.h"

// clang-format off
// preserve visually scannable types

using UniqueOrtModel                  = std::unique_ptr<OrtModel,                  decltype(WinmlAdapterApi::ReleaseModel)>;
using UniqueOrtThreadPool             = std::unique_ptr<OrtThreadPool,             decltype(WinmlAdapterApi::ReleaseThreadPool)>;
using UniqueOrtAllocator              = std::unique_ptr<OrtAllocator,              decltype(OrtApi::ReleaseAllocator)>;
using UniqueOrtSessionOptions         = std::unique_ptr<OrtSessionOptions,         decltype(OrtApi::ReleaseSessionOptions)>;
using UniqueOrtSession                = std::unique_ptr<OrtSession,                decltype(OrtApi::ReleaseSession)>;
using UniqueOrtValue                  = std::unique_ptr<OrtValue,                  decltype(OrtApi::ReleaseValue)>;
using UniqueOrtMemoryInfo             = std::unique_ptr<OrtMemoryInfo,             decltype(OrtApi::ReleaseMemoryInfo)>;
using UniqueOrtTypeInfo               = std::unique_ptr<OrtTypeInfo,               decltype(OrtApi::ReleaseTypeInfo)>;
using UniqueOrtTensorTypeAndShapeInfo = std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(OrtApi::ReleaseTensorTypeAndShapeInfo)>;
using UniqueOrtRunOptions             = std::unique_ptr<OrtRunOptions,             decltype(OrtApi::ReleaseRunOptions)>;
using UniqueOrtEnv                    = std::unique_ptr<OrtEnv,                    decltype(OrtApi::ReleaseEnv)>;

// clang-format on
