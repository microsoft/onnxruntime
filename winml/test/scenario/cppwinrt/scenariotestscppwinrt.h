// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct ScenarioTestsApi
{
    SetupClass ScenarioCppWinrtTestsClassSetup;
    SetupTest ScenarioCppWinrtTestsGpuMethodSetup;
    SetupTest ScenarioCppWinrtTestsSkipEdgeCoreMethodSetup;
    SetupTest ScenarioCppWinrtTestsGpuSkipEdgeCoreMethodSetup;
    VoidTest Sample1;
    VoidTest Scenario1LoadBindEvalDefault;
    VoidTest Scenario2LoadModelFromStream;
    VoidTest Scenario5AsyncEval;
    VoidTest Scenario7EvalWithNoBind;
    VoidTest Scenario8SetDeviceSampleDefault;
    VoidTest Scenario8SetDeviceSampleCPU;
    VoidTest Scenario17DevDiagnostics;
    VoidTest DISABLED_Scenario22ImageBindingAsCPUTensor;
    VoidTest QuantizedModels;
    VoidTest EncryptedStream;
    VoidTest Scenario3SoftwareBitmapInputBinding;
    VoidTest Scenario6BindWithProperties;
    VoidTest Scenario8SetDeviceSampleDefaultDirectX;
    VoidTest Scenario8SetDeviceSampleMinPower;
    VoidTest Scenario8SetDeviceSampleMaxPerf;
    VoidTest Scenario8SetDeviceSampleMyCameraDevice;
    VoidTest Scenario8SetDeviceSampleCustomCommandQueue;
    VoidTest DISABLED_Scenario9LoadBindEvalInputTensorGPU;
    VoidTest Scenario13SingleModelOnCPUandGPU;
    VoidTest Scenario11FreeDimensionsTensor;
    VoidTest Scenario11FreeDimensionsImage;
    VoidTest Scenario14RunModelSwapchain;
    VoidTest Scenario20aLoadBindEvalCustomOperatorCPU;
    VoidTest Scenario20bLoadBindEvalReplacementCustomOperatorCPU;
    VoidTest DISABLED_Scenario21RunModel2ChainZ;
    VoidTest DISABLED_Scenario22ImageBindingAsGPUTensor;
    VoidTest MsftQuantizedModels;
    VoidTest DISABLED_SyncVsAsync;
    VoidTest DISABLED_CustomCommandQueueWithFence;
    VoidTest DISABLED_ReuseVideoFrame;
    VoidTest DeviceLostRecovery;
    VoidTest Scenario8SetDeviceSampleD3D11Device;
    VoidTest D2DInterop;
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
WINML_TEST(ScenarioCppWinrtTests, DISABLED_Scenario22ImageBindingAsCPUTensor)
WINML_TEST(ScenarioCppWinrtTests, QuantizedModels)
WINML_TEST(ScenarioCppWinrtTests, EncryptedStream)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN(ScenarioCppWinrtTestsGpu)
WINML_TEST_CLASS_SETUP_CLASS(ScenarioCppWinrtTestsClassSetup)
WINML_TEST_CLASS_SETUP_METHOD(ScenarioCppWinrtTestsGpuMethodSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario3SoftwareBitmapInputBinding)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario6BindWithProperties)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario8SetDeviceSampleDefaultDirectX)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario8SetDeviceSampleMinPower)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario8SetDeviceSampleMaxPerf)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario8SetDeviceSampleCustomCommandQueue)
WINML_TEST(ScenarioCppWinrtTestsGpu, DISABLED_Scenario9LoadBindEvalInputTensorGPU)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario13SingleModelOnCPUandGPU)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario11FreeDimensionsTensor)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario11FreeDimensionsImage)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario14RunModelSwapchain)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario20aLoadBindEvalCustomOperatorCPU)
WINML_TEST(ScenarioCppWinrtTestsGpu, Scenario20bLoadBindEvalReplacementCustomOperatorCPU)
WINML_TEST(ScenarioCppWinrtTestsGpu, DISABLED_Scenario21RunModel2ChainZ)
WINML_TEST(ScenarioCppWinrtTestsGpu, DISABLED_Scenario22ImageBindingAsGPUTensor)
WINML_TEST(ScenarioCppWinrtTestsGpu, MsftQuantizedModels)
WINML_TEST(ScenarioCppWinrtTestsGpu, DISABLED_SyncVsAsync)
WINML_TEST(ScenarioCppWinrtTestsGpu, DISABLED_CustomCommandQueueWithFence)
WINML_TEST(ScenarioCppWinrtTestsGpu, DISABLED_ReuseVideoFrame)
WINML_TEST(ScenarioCppWinrtTestsGpu, DeviceLostRecovery)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN(ScenarioCppWinrtTestsSkipEdgeCore)
WINML_TEST_CLASS_SETUP_CLASS(ScenarioCppWinrtTestsClassSetup)
WINML_TEST_CLASS_SETUP_METHOD(ScenarioCppWinrtTestsSkipEdgeCoreMethodSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(ScenarioCppWinrtTestsSkipEdgeCore, Scenario8SetDeviceSampleMyCameraDevice)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN(ScenarioCppWinrtTestsGpuSkipEdgeCore)
WINML_TEST_CLASS_SETUP_CLASS(ScenarioCppWinrtTestsClassSetup)
WINML_TEST_CLASS_SETUP_METHOD(ScenarioCppWinrtTestsGpuSkipEdgeCoreMethodSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(ScenarioCppWinrtTestsGpuSkipEdgeCore, Scenario8SetDeviceSampleD3D11Device)
WINML_TEST(ScenarioCppWinrtTestsGpuSkipEdgeCore, D2DInterop)
WINML_TEST_CLASS_END()