// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "fileHelpers.h"
#include "winrt_headers.h"

namespace APITest {
static void LoadModel(const std::wstring& modelPath, winml::LearningModel& learningModel) {
  std::wstring fullPath = FileHelpers::GetModulePath() + modelPath;
  learningModel = winml::LearningModel::LoadFromFilePath(fullPath);
};

static inline uint64_t GetAdapterIdQuadPart(winml::LearningModelDevice& device) {
  LARGE_INTEGER id;
  id.LowPart = device.AdapterId().LowPart;
  id.HighPart = device.AdapterId().HighPart;
  return id.QuadPart;
};

static inline _LUID GetAdapterIdAsLUID(winml::LearningModelDevice& device) {
  _LUID id;
  id.LowPart = device.AdapterId().LowPart;
  id.HighPart = device.AdapterId().HighPart;
  return id;
}
};  // namespace APITest
