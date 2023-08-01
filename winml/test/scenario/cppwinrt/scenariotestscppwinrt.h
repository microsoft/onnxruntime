// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct ScenarioTestsApi {
  SetupClass ScenarioCppWinrtTestsClassSetup;
  VoidTest Sample1;
  VoidTest Scenario1LoadBindEvalDefault;
  VoidTest Scenario2LoadModelFromStream;
  VoidTest Scenario5AsyncEval;
  VoidTest Scenario7EvalWithNoBind;
  VoidTest Scenario8SetDeviceSampleDefault;
  VoidTest Scenario8SetDeviceSampleCPU;
  VoidTest Scenario17DevDiagnostics;
  VoidTest Scenario22ImageBindingAsCPUTensor;
  VoidTest Scenario23NominalPixelRange;
  VoidTest QuantizedModels;
  VoidTest EncryptedStream;
  VoidTest Scenario3SoftwareBitmapInputBinding;
  VoidTest Scenario6BindWithProperties;
  VoidTest Scenario8SetDeviceSampleDefaultDirectX;
  VoidTest Scenario8SetDeviceSampleMinPower;
  VoidTest Scenario8SetDeviceSampleMaxPerf;
  VoidTest Scenario8SetDeviceSampleMyCameraDevice;
  VoidTest Scenario8SetDeviceSampleCustomCommandQueue;
  VoidTest Scenario9LoadBindEvalInputTensorGPU;
  VoidTest Scenario13SingleModelOnCPUandGPU;
  VoidTest Scenario11FreeDimensionsTensor;
  VoidTest Scenario11FreeDimensionsImage;
  VoidTest Scenario14RunModelSwapchain;
  VoidTest Scenario20aLoadBindEvalCustomOperatorCPU;
  VoidTest Scenario20bLoadBindEvalReplacementCustomOperatorCPU;
  VoidTest Scenario21RunModel2ChainZ;
  VoidTest Scenario22ImageBindingAsGPUTensor;
  VoidTest MsftQuantizedModels;
  VoidTest SyncVsAsync;
  VoidTest CustomCommandQueueWithFence;
  VoidTest ReuseVideoFrame;
  VoidTest DeviceLostRecovery;
  VoidTest Scenario8SetDeviceSampleD3D11Device;
  VoidTest D2DInterop;
  VoidTest BindMultipleCPUBuffersInputsOnCpu;
  VoidTest BindMultipleCPUBuffersInputsOnGpu;
  VoidTest BindMultipleCPUBuffersOutputsOnCpu;
  VoidTest BindMultipleCPUBuffersOutputsOnGpu;
};
const ScenarioTestsApi& getapi();

WINML_TEST_CLASS_BEGIN(ScenarioCppWinrtTests)
WINML_TEST_CLASS_SETUP_CLASS(ScenarioCppWinrtTestsClassSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(ScenarioCppWinrtTests, Sample1)
WINML_TEST(ScenarioCppWinrtTests, Scenario1LoadBindEvalDefault)
WINML_TEST(ScenarioCppWinrtTests, Scenario2LoadModelFromStream)
WINML_TEST(ScenarioCppWinrtTests, Scenario5AsyncEval)
WINML_TEST(ScenarioCppWinrtTests, Scenario7EvalWithNoBind)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleDefault)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleCPU)
WINML_TEST(ScenarioCppWinrtTests, Scenario17DevDiagnostics)
WINML_TEST(ScenarioCppWinrtTests, Scenario22ImageBindingAsCPUTensor)
WINML_TEST(ScenarioCppWinrtTests, Scenario23NominalPixelRange)
WINML_TEST(ScenarioCppWinrtTests, QuantizedModels)
WINML_TEST(ScenarioCppWinrtTests, EncryptedStream)
WINML_TEST(ScenarioCppWinrtTests, Scenario3SoftwareBitmapInputBinding)
WINML_TEST(ScenarioCppWinrtTests, Scenario6BindWithProperties)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleDefaultDirectX)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleMinPower)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleMaxPerf)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleCustomCommandQueue)
WINML_TEST(ScenarioCppWinrtTests, Scenario9LoadBindEvalInputTensorGPU)
WINML_TEST(ScenarioCppWinrtTests, Scenario13SingleModelOnCPUandGPU)
WINML_TEST(ScenarioCppWinrtTests, Scenario11FreeDimensionsTensor)
WINML_TEST(ScenarioCppWinrtTests, Scenario11FreeDimensionsImage)
WINML_TEST(ScenarioCppWinrtTests, Scenario14RunModelSwapchain)
WINML_TEST(ScenarioCppWinrtTests, Scenario20aLoadBindEvalCustomOperatorCPU)
WINML_TEST(ScenarioCppWinrtTests, Scenario20bLoadBindEvalReplacementCustomOperatorCPU)
WINML_TEST(ScenarioCppWinrtTests, Scenario21RunModel2ChainZ)
WINML_TEST(ScenarioCppWinrtTests, Scenario22ImageBindingAsGPUTensor)
WINML_TEST(ScenarioCppWinrtTests, MsftQuantizedModels)
WINML_TEST(ScenarioCppWinrtTests, SyncVsAsync)
WINML_TEST(ScenarioCppWinrtTests, CustomCommandQueueWithFence)
WINML_TEST(ScenarioCppWinrtTests, ReuseVideoFrame)
WINML_TEST(ScenarioCppWinrtTests, DeviceLostRecovery)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleD3D11Device)
WINML_TEST(ScenarioCppWinrtTests, D2DInterop)
WINML_TEST(ScenarioCppWinrtTests, Scenario8SetDeviceSampleMyCameraDevice)
WINML_TEST(ScenarioCppWinrtTests, BindMultipleCPUBuffersInputsOnCpu)
WINML_TEST(ScenarioCppWinrtTests, BindMultipleCPUBuffersInputsOnGpu)
WINML_TEST(ScenarioCppWinrtTests, BindMultipleCPUBuffersOutputsOnCpu)
WINML_TEST(ScenarioCppWinrtTests, BindMultipleCPUBuffersOutputsOnGpu)
WINML_TEST_CLASS_END()
