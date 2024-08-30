// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "SqueezeNetValidator.h"
#include "protobufHelpers.h"
#include "fileHelpers.h"
#include "core/common/common.h"
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Streams.h>
#include <iostream>

using namespace wfc;
using namespace wgi;
using namespace wm;
using namespace ws;
using namespace wss;
using namespace winml;

namespace WinML::Engine::Test {

#define MAX_PROFILING_LOOP 100

static void BindImage(
  LearningModelBinding binding, const wchar_t* name, const wchar_t* fullImagePath, bool bindAsInspectable = false
) {
  auto imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
  auto stream = imagefile.OpenAsync(FileAccessMode::Read).get();
  auto decoder = BitmapDecoder::CreateAsync(stream).get();
  auto softwareBitmap = decoder.GetSoftwareBitmapAsync().get();
  auto frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

  if (bindAsInspectable) {
    binding.Bind(name, frame);
  } else {
    auto imagetensor = ImageFeatureValue::CreateFromVideoFrame(frame);
    binding.Bind(name, imagetensor);
  }
}

static void BindTensor(
  LearningModelBinding binding, const wchar_t* name, ITensor inputTensor, bool bindAsInspectable = false
) {
  if (inputTensor == nullptr) {
    throw winrt::hresult_invalid_argument(L"input tensor provided to squeezenet is null.");
  }

  if (bindAsInspectable) {
    binding.Bind(name, inputTensor.as<TensorFloat>().GetAsVectorView());
  } else {
    binding.Bind(name, inputTensor);
  }
}

template <typename T>
ITensor BindOutput(
  OutputBindingStrategy strategy,
  LearningModelBinding binding,
  const wchar_t* name,
  const IVectorView<int64_t> shape = nullptr
) {
  ITensor outputTensor = nullptr;
  switch (strategy) {
    case OutputBindingStrategy::Bound:
      outputTensor = T::Create(shape);
      binding.Bind(name, outputTensor);
      break;
    case OutputBindingStrategy::Empty:
      outputTensor = T::Create();
      binding.Bind(name, outputTensor);
      break;
    case OutputBindingStrategy::Unbound:
      __fallthrough;
    default:
      break;
  }

  return outputTensor;
}

ImageFeatureValue BindImageOutput(OutputBindingStrategy strategy, LearningModelBinding binding, const wchar_t* name) {
  ImageFeatureValue outputTensor = nullptr;
  switch (strategy) {
    case OutputBindingStrategy::Bound: {
      SoftwareBitmap bitmap(BitmapPixelFormat::Bgra8, 720, 720);
      VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(bitmap);
      outputTensor = ImageFeatureValue::CreateFromVideoFrame(frame);
      binding.Bind(name, outputTensor);
      break;
    }
    case OutputBindingStrategy::Unbound:
      __fallthrough;
  }

  return outputTensor;
}

void ModelValidator::FnsCandy16(
  const std::string& instance,
  LearningModelDeviceKind deviceKind,
  OutputBindingStrategy outputBindingStrategy,
  bool bindInputsAsIInspectable,
  float dataTolerance
) {
  ORT_UNUSED_PARAMETER(dataTolerance);
  // file name strings
  static wchar_t* modelFileName = L"winmlperf_coreml_FNS-Candy_prerelease_fp16.onnx";
  static wchar_t* inputDataImageFileName = L"fish_720.png";
  static wchar_t* outputDataFileName = L"output.png";
  static wchar_t* inputBindingName = L"inputImage";
  static const wchar_t* outputDataBindingName = L"outputImage";

  auto modulePath = FileHelpers::GetModulePath();
  auto fullModelPath = modulePath + modelFileName;
  auto outputFileName = modulePath + outputDataFileName;

  // WinML model creation
  LearningModel model = nullptr;
  model = LearningModel::LoadFromFilePath(fullModelPath);

  LearningModelSession modelSession = nullptr;
  modelSession = LearningModelSession(model, LearningModelDevice(deviceKind));

  LearningModelBinding modelBinding(modelSession);
  auto fullImagePath = modulePath + inputDataImageFileName;
  BindImage(modelBinding, inputBindingName, fullImagePath.c_str(), bindInputsAsIInspectable);

  // create the tensor for the actual output
  auto output = model.OutputFeatures().First().Current();
  if (output.Kind() != LearningModelFeatureKind::Tensor) {
    throw winrt::hresult_invalid_argument(L"Model output kind is not type Tensor");
  }

  auto shape = winrt::single_threaded_vector(std::vector<int64_t>{1, 1});
  auto outputTensor = BindImageOutput(outputBindingStrategy, modelBinding, outputDataBindingName);

  // Evaluate the model
  std::cout << "Calling EvaluateSync on instance" << instance << "\n";
  LearningModelEvaluationResult result = nullptr;
  result = modelSession.Evaluate(modelBinding, {});

  // Get results
  if (outputBindingStrategy == OutputBindingStrategy::Unbound) {
    // When output binding strategy is unbound, the output tensor was not set on bind.
    // Therefore, we need to retrieve it from the LearnignModelEvaluationResult
    // TODO: is this right? outputTensorT is unused...
    /*auto outputTensorT = */ result.Outputs().Lookup(outputDataBindingName).as<TensorFloat16Bit>();
  } else {
    if (result.Outputs().Lookup(outputDataBindingName) != outputTensor) {
      throw winrt::hresult_invalid_argument(L"Evaluation Results lookup don't match LearningModelBinding Output Tensor."
      );
    }

    auto softwareBitmap = outputTensor.VideoFrame().SoftwareBitmap();

    auto folder = StorageFolder::GetFolderFromPathAsync(modulePath.c_str()).get();
    auto imagefile = folder.CreateFileAsync(outputDataFileName, CreationCollisionOption::ReplaceExisting).get();
    auto stream = imagefile.OpenAsync(FileAccessMode::ReadWrite).get();
    auto encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), stream).get();
    encoder.SetSoftwareBitmap(softwareBitmap);
    encoder.FlushAsync();
  }
}

void ModelValidator::SqueezeNet(
  const std::string& instance,
  LearningModelDeviceKind deviceKind,
  float dataTolerance,
  bool bindAsImage,
  OutputBindingStrategy outputBindingStrategy,
  bool bindInputsAsIInspectable
) {
  // file name strings
  static wchar_t* modelFileName = L"model.onnx";
  static wchar_t* inputDataFileName = L"test_data_0_input.pb";
  static wchar_t* outputDataFileName = L"test_data_0_output.pb";
  static wchar_t* inputBindingName = L"data_0";
  static wchar_t* inputDataImageFileName = L"kitten_224.png";
  static const wchar_t* outputDataBindingName = L"softmaxout_1";

  auto modulePath = FileHelpers::GetModulePath();
  auto fullModelPath = modulePath + modelFileName;
  auto outputFileName = modulePath + outputDataFileName;

  // WinML model creation
  LearningModel model = nullptr;
  model = LearningModel::LoadFromFilePath(fullModelPath);

  LearningModelSession modelSession = nullptr;
  modelSession = LearningModelSession(model, LearningModelDevice(deviceKind));

  LearningModelBinding modelBinding(modelSession);

  if (bindAsImage) {
    std::wstring fullImagePath = modulePath + inputDataImageFileName;
    BindImage(modelBinding, inputBindingName, fullImagePath.c_str(), bindInputsAsIInspectable);
  } else {
    auto inputDataPath = modulePath + inputDataFileName;
    auto inputTensor = ProtobufHelpers::LoadTensorFromProtobufFile(inputDataPath, false);
    BindTensor(modelBinding, inputBindingName, inputTensor, bindInputsAsIInspectable);
  }

  // load up the expected output
  auto expectedResultsTensor = ProtobufHelpers::LoadTensorFromProtobufFile(outputFileName, false);
  if (expectedResultsTensor == nullptr) {
    throw winrt::hresult_invalid_argument(L"Expected Results from protobuf file are null.");
  }

  // create the tensor for the actual output
  auto output = model.OutputFeatures().First().Current();
  if (output.Kind() != LearningModelFeatureKind::Tensor) {
    throw winrt::hresult_invalid_argument(L"Expected output feature kind of model to be Tensor");
  }

  auto outputTensor =
    BindOutput<TensorFloat>(outputBindingStrategy, modelBinding, outputDataBindingName, expectedResultsTensor.Shape());

  // Evaluate the model
  std::cout << "Calling EvaluateSync on instance " << instance << "\n";
  LearningModelEvaluationResult result = nullptr;
  result = modelSession.Evaluate(modelBinding, {});

  // Get results
  if (outputBindingStrategy == OutputBindingStrategy::Unbound) {
    // When output binding strategy is unbound, the output tensor was not set on bind.
    // Therefore, we need to retrieve it from the LearnignModelEvaluationResult
    outputTensor = result.Outputs().Lookup(outputDataBindingName).as<ITensor>();
  } else {
    if (result.Outputs().Lookup(outputDataBindingName) != outputTensor) {
      throw winrt::hresult_error(
        E_UNEXPECTED, L"Evaluation Results lookup don't match LearningModelBinding output tensor."
      );
    }
  }

  auto outDataExpected = expectedResultsTensor.as<TensorFloat>().GetAsVectorView();
  auto outDataActual = outputTensor.as<TensorFloat>().GetAsVectorView();

  if (outDataActual.Size() != outDataExpected.Size()) {
    throw winrt::hresult_error(E_UNEXPECTED, L"Actual tensor data size doesn't match expected tensor data size.");
  }
  for (uint32_t i = 0; i < outDataActual.Size(); i++) {
    float delta = std::abs(outDataActual.GetAt(i) - outDataExpected.GetAt(i));
    if (delta > dataTolerance) {
      std::wstringstream ss;
      ss << "EXPECTED: " << outDataExpected.GetAt(i) << " , ACTUAL: " << outDataActual.GetAt(i) << "instance "
         << instance.c_str() << ", element " << i;
      throw winrt::hresult_error(E_UNEXPECTED, ss.str());
    }
  }
}
}  // namespace WinML::Engine::Test
