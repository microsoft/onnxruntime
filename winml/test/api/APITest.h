// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "fileHelpers.h"
namespace APITest {
static void LoadModel(const std::wstring& modelPath,
                      winrt::Windows::AI::MachineLearning::LearningModel& learningModel) {
  std::wstring fullPath = FileHelpers::GetModulePath() + modelPath;
  learningModel = winrt::Windows::AI::MachineLearning::LearningModel::LoadFromFilePath(fullPath);
};

static uint64_t GetAdapterIdQuadPart(winrt::Windows::AI::MachineLearning::LearningModelDevice& device) {
  LARGE_INTEGER id;
  id.LowPart = device.AdapterId().LowPart;
  id.HighPart = device.AdapterId().HighPart;
  return id.QuadPart;
};

static _LUID GetAdapterIdAsLUID(winrt::Windows::AI::MachineLearning::LearningModelDevice& device) {
  _LUID id;
  id.LowPart = device.AdapterId().LowPart;
  id.HighPart = device.AdapterId().HighPart;
  return id;
}
};  // namespace APITest
