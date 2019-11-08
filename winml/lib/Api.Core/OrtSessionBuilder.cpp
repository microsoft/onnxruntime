// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "CpuOrtSessionBuilder.h"
#include "DmlOrtSessionBuilder.h"

#include "LearningModelDevice.h"

using namespace Windows::AI::MachineLearning;

std::unique_ptr<IOrtSessionBuilder> Windows::AI::MachineLearning::CreateOrtSessionBuilder(
    ID3D12Device* device, ID3D12CommandQueue* queue) {

  auto session_builder =
      (device == nullptr)
          ? std::unique_ptr<IOrtSessionBuilder>(new CpuOrtSessionBuilder())
          : std::unique_ptr<IOrtSessionBuilder>(new DmlOrtSessionBuilder(device, queue));

  return session_builder;
}