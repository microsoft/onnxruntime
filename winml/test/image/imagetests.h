//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

#include <string>
#include <utility>

#include <winrt/Windows.Media.h>
#include <winrt/Windows.Graphics.Imaging.h>

#include "googleTestMacros.h"

// class ImageTests
// {
// public:
//     // TEST_CLASS(ImageTests);
//     // TEST_CLASS_SETUP(TestClassSetup);
//     // TEST_METHOD_SETUP(TestMethodSetup);

//     // BEGIN_TEST_METHOD(mnistImageTests)
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:MnistTestsData.xml#mnistTable")
//     // END_TEST_METHOD()

//     // BEGIN_TEST_METHOD(ImageTest)
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:ImageTestsData.xml#Devices")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:ImageTestsData.xml#Models")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:ImageTestsData.xml#InputImageFiles")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:ImageTestsData.xml#OutputBindingStrategies")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:ImageTestsData.xml#InputPixelFormats")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:ImageTestsData.xml#EvaluationStrategies")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:ImageTestsData.xml#InputImageSources")
//     // END_TEST_METHOD()

//     // BEGIN_TEST_METHOD(BatchSupport)
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:BatchTestData.xml#Devices")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:BatchTestData.xml#Models")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:BatchTestData.xml#EvaluationStrategies")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:BatchTestData.xml#OutputBindingStrategies")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:BatchTestData.xml#VideoFrameSources")
//     //     TEST_METHOD_PROPERTY(L"DataSource", L"Table:BatchTestData.xml#OutputVideoFrameSources")
//     // END_TEST_METHOD()


//     // TEST_METHOD(LoadBindEvalModelWithoutImageMetadata);
//     // TEST_METHOD(LoadInvalidBindModelWithoutImageMetadata);
//     // TEST_METHOD(LoadBindModelWithoutImageMetadata);
//     // TEST_METHOD(ImageMetaDataTest);
//     // TEST_METHOD(ImageBindingInputAndOutput);
//     // TEST_METHOD(ImageBindingInputAndOutput_BindInputTensorAsInspectable);
//     // TEST_METHOD(ImageBindingTwiceSameFeatureValueOnGpu);
//     // TEST_METHOD(ImageBindingTwiceDifferentFeatureValueOnGpu);
//     // TEST_METHOD(ImageBindingStyleTransfer);
//     // TEST_METHOD(ImageBindingAsGPUTensor);
//     // TEST_METHOD(SynchronizeGPUWorkloads);


// private:
//     bool m_runGPUTests = true;
//     winrt::Windows::AI::MachineLearning::LearningModel m_model = nullptr;
//     winrt::Windows::AI::MachineLearning::LearningModelDevice m_device = nullptr;
//     winrt::Windows::AI::MachineLearning::LearningModelSession m_session = nullptr;
//     winrt::Windows::AI::MachineLearning::LearningModelBinding m_modelBinding = nullptr;
//     winrt::Windows::AI::MachineLearning::LearningModelEvaluationResult m_result = nullptr;

//     void LoadModel(const std::wstring& modelPath);

//     void PrepareModelSessionBinding(
//         const std::wstring& modelFileName,
//         winrt::Windows::AI::MachineLearning::LearningModelDeviceKind deviceKind,
//         std::optional<uint32_t> optimizedBatchSize);

//     bool BindInputValue(
//         const std::wstring& imageFileName,
//         const std::wstring& inputPixelFormat,
//         const std::wstring& modelPixelFormat,
//         InputImageSource inputImageSource,
//         winrt::Windows::AI::MachineLearning::LearningModelDeviceKind deviceKind);

//     winrt::Windows::Media::VideoFrame BindImageOutput(
//         ModelInputOutputType modelInputOutputType,
//         OutputBindingStrategy outputBindingStrategy,
//         const std::wstring& modelPixelFormat);

//     winrt::Windows::Foundation::Collections::IVector<winrt::Windows::Media::VideoFrame> BindImageOutput(
//         ModelInputOutputType modelInputOutputType,
//         OutputBindingStrategy outputBindingStrategy,
//         VideoFrameSource outputVideoFrameSource,
//         const std::wstring& modelPixelFormat,
//         const uint32_t& batchSize);

//     void EvaluateTest();

//     void VerifyResults(
//         winrt::Windows::Media::VideoFrame outputTensor,
//         const std::wstring& bm_imageFileName,
//         const std::wstring& modelPixelFormat);

//     void ValidateOutputImageMetaData(std::wstring path, winrt::Windows::Graphics::Imaging::BitmapAlphaMode expectedmode, winrt::Windows::Graphics::Imaging::BitmapPixelFormat expectedformat, bool supported);
//     void TestImageBindingStyleTransfer(const wchar_t* modelFileName, const wchar_t* inputDataImageFileName, wchar_t* outputDataImageFileName);
//     void SynchronizeGPUWorkloads(const wchar_t* modelFileName, const wchar_t* inputDataImageFileName);



//     void GetCleanSession(
//         winrt::Windows::AI::MachineLearning::LearningModelDeviceKind deviceKind,
//         std::wstring modelFilePath,
//         winrt::Windows::AI::MachineLearning::LearningModelDevice &device,
//         winrt::Windows::AI::MachineLearning::LearningModelSession &session);
//     void BindInputToSession(
//         BindingLocation bindLocation,
//         std::wstring inputDataLocation,
//         winrt::Windows::AI::MachineLearning::LearningModelSession& session,
//         winrt::Windows::AI::MachineLearning::LearningModelBinding& binding);
//     static void BindOutputToSession(
//         BindingLocation bindLocation,
//         winrt::Windows::AI::MachineLearning::LearningModelSession& session,
//         winrt::Windows::AI::MachineLearning::LearningModelBinding& binding);
// };
