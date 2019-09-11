#include "pch.h"

#include "CpuOrtSessionBuilder.h"
#include "DmlOrtSessionBuilder.h"

#include "LearningModelDevice.h"

using namespace Windows::AI::MachineLearning;

std::unique_ptr<IOrtSessionBuilder> Windows::AI::MachineLearning::CreateOrtSessionBuilder(
    winml::LearningModelDevice const& device) {
  auto device_impl = device.as<winmlp::LearningModelDevice>();

  auto session_builder =
      device_impl->IsCpuDevice()
          ? std::unique_ptr<IOrtSessionBuilder>(new CpuOrtSessionBuilder())
          : std::unique_ptr<IOrtSessionBuilder>(new DmlOrtSessionBuilder(device));

  return session_builder;
}