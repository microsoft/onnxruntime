// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct ScenarioTestApi
{
    SetupTest ScenarioCppWinrtTestSetup;
    SetupTest ScenarioCppWinrtGpuTestSetup;
    SetupTest ScenarioCppWinrtGpuSkipEdgeCoreTestSetup;
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
const ScenarioTestApi& getapi();

WINML_TEST_CLASS_BEGIN_WITH_SETUP(ScenarioCppWinrtTest, ScenarioCppWinrtTestSetup)
WINML_TEST(ScenarioCppWinrtTest, Sample1)
WINML_TEST(ScenarioCppWinrtTest, Scenario1LoadBindEvalDefault)
WINML_TEST(ScenarioCppWinrtTest, Scenario2LoadModelFromStream)
WINML_TEST(ScenarioCppWinrtTest, Scenario5AsyncEval)
WINML_TEST(ScenarioCppWinrtTest, Scenario7EvalWithNoBind)
WINML_TEST(ScenarioCppWinrtTest, Scenario8SetDeviceSampleDefault)
WINML_TEST(ScenarioCppWinrtTest, Scenario8SetDeviceSampleCPU)
WINML_TEST(ScenarioCppWinrtTest, Scenario17DevDiagnostics)
WINML_TEST(ScenarioCppWinrtTest, DISABLED_Scenario22ImageBindingAsCPUTensor)
WINML_TEST(ScenarioCppWinrtTest, QuantizedModels)
WINML_TEST(ScenarioCppWinrtTest, EncryptedStream)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN_WITH_SETUP(ScenarioCppWinrtGpuTest, ScenarioCppWinrtGpuTestSetup)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario3SoftwareBitmapInputBinding)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario6BindWithProperties)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario8SetDeviceSampleDefaultDirectX)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario8SetDeviceSampleMinPower)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario8SetDeviceSampleMaxPerf)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario8SetDeviceSampleCustomCommandQueue)
WINML_TEST(ScenarioCppWinrtGpuTest, DISABLED_Scenario9LoadBindEvalInputTensorGPU)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario13SingleModelOnCPUandGPU)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario11FreeDimensionsTensor)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario11FreeDimensionsImage)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario14RunModelSwapchain)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario20aLoadBindEvalCustomOperatorCPU)
WINML_TEST(ScenarioCppWinrtGpuTest, Scenario20bLoadBindEvalReplacementCustomOperatorCPU)
WINML_TEST(ScenarioCppWinrtGpuTest, DISABLED_Scenario21RunModel2ChainZ)
WINML_TEST(ScenarioCppWinrtGpuTest, DISABLED_Scenario22ImageBindingAsGPUTensor)
WINML_TEST(ScenarioCppWinrtGpuTest, MsftQuantizedModels)
WINML_TEST(ScenarioCppWinrtGpuTest, DISABLED_SyncVsAsync)
WINML_TEST(ScenarioCppWinrtGpuTest, DISABLED_CustomCommandQueueWithFence)
WINML_TEST(ScenarioCppWinrtGpuTest, DISABLED_ReuseVideoFrame)
WINML_TEST(ScenarioCppWinrtGpuTest, DeviceLostRecovery)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN_WITH_SETUP(ScenarioCppWinrtGpuSkipEdgeCoreTest, ScenarioCppWinrtGpuSkipEdgeCoreTestSetup)
WINML_TEST(ScenarioCppWinrtGpuSkipEdgeCoreTest, Scenario8SetDeviceSampleMyCameraDevice)
WINML_TEST(ScenarioCppWinrtGpuSkipEdgeCoreTest, Scenario8SetDeviceSampleD3D11Device )
WINML_TEST(ScenarioCppWinrtGpuSkipEdgeCoreTest, D2DInterop)
WINML_TEST_CLASS_END()