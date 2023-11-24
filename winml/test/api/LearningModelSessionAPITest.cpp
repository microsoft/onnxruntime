// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"

#include "APITest.h"
#include "CommonDeviceHelpers.h"
#include "LearningModelSessionAPITest.h"
#include "protobufHelpers.h"
#include "winrt/Windows.Storage.h"

#include <D3d11_4.h>
#include <dxgi1_6.h>
#include "Psapi.h"

#include <complex>

using namespace winrt;
using namespace winml;
using namespace wfc;

#ifndef BUILD_INBOX
// experimental
using namespace winml_experimental;
using Operator = winml_experimental::LearningModelOperator;

static const wchar_t MS_EXPERIMENTAL_DOMAIN[] = L"com.microsoft.experimental";
#endif

using wf::IPropertyValue;

#define INT64(x) static_cast<int64_t>(x)
#define SIZET(x) static_cast<size_t>(x)
#define INT32(x) static_cast<int32_t>(x)

static void LearningModelSessionAPITestsClassSetup() {
  init_apartment();
#ifdef BUILD_INBOX
  winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
}

static void CreateSessionDeviceDefault() {
  LearningModel learningModel = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
  WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void CreateSessionDeviceCpu() {
  LearningModel learningModel = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Cpu));
  WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
  // for the CPU device, make sure that we get back NULL and 0 for any device properties
  WINML_EXPECT_EQUAL(learningModelDevice.Direct3D11Device(), nullptr);
  LARGE_INTEGER id;
  id.QuadPart = APITest::GetAdapterIdQuadPart(learningModelDevice);
  WINML_EXPECT_EQUAL(id.LowPart, static_cast<DWORD>(0));
  WINML_EXPECT_EQUAL(id.HighPart, 0);
}

static void CreateSessionWithModelLoadedFromStream() {
  LearningModel learningModel = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  std::wstring path = FileHelpers::GetModulePath() + L"model.onnx";
  auto storageFile = ws::StorageFile::GetFileFromPathAsync(path).get();

  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromStream(storageFile));

  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
  WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void CreateSessionDeviceDirectX() {
  LearningModel learningModel = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectX));
  WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void CreateSessionDeviceDirectXHighPerformance() {
  LearningModel learningModel = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectXHighPerformance));
  WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void CreateSessionDeviceDirectXMinimumPower() {
  LearningModel learningModel = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

  WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectXMinPower));
  WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void AdapterIdAndDevice() {
  LearningModel learningModel = nullptr;
  LearningModelDevice learningModelDevice = nullptr;
  LearningModelSession learningModelSession = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

  com_ptr<IDXGIFactory6> factory;
  WINML_EXPECT_HRESULT_SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory6), factory.put_void()));
  com_ptr<IDXGIAdapter> adapter;

  learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectX);
  WINML_EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapters(0, adapter.put()));
  DXGI_ADAPTER_DESC desc;
  WINML_EXPECT_HRESULT_SUCCEEDED(adapter->GetDesc(&desc));
  LARGE_INTEGER id;
  id.QuadPart = APITest::GetAdapterIdQuadPart(learningModelDevice);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.LowPart, id.LowPart);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.HighPart, id.HighPart);
  WINML_EXPECT_TRUE(learningModelDevice.Direct3D11Device() != nullptr);

  learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectXHighPerformance);
  adapter = nullptr;
  WINML_EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapterByGpuPreference(
    0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, __uuidof(IDXGIAdapter), adapter.put_void()
  ));
  WINML_EXPECT_HRESULT_SUCCEEDED(adapter->GetDesc(&desc));
  id.QuadPart = APITest::GetAdapterIdQuadPart(learningModelDevice);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.LowPart, id.LowPart);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.HighPart, id.HighPart);
  WINML_EXPECT_TRUE(learningModelDevice.Direct3D11Device() != nullptr);

  adapter = nullptr;
  learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectXMinPower);
  WINML_EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapterByGpuPreference(
    0, DXGI_GPU_PREFERENCE_MINIMUM_POWER, __uuidof(IDXGIAdapter), adapter.put_void()
  ));
  WINML_EXPECT_HRESULT_SUCCEEDED(adapter->GetDesc(&desc));
  id.QuadPart = APITest::GetAdapterIdQuadPart(learningModelDevice);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.LowPart, id.LowPart);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.HighPart, id.HighPart);
  WINML_EXPECT_TRUE(learningModelDevice.Direct3D11Device() != nullptr);

  WINML_EXPECT_NO_THROW(learningModelSession = LearningModelSession(learningModel, learningModelDevice));
  WINML_EXPECT_EQUAL(learningModelSession.Device().AdapterId(), learningModelDevice.AdapterId());
}

static void EvaluateFeatures() {
  std::vector<int64_t> shape = {4};
  std::vector<winrt::hstring> data = {L"one", L"two", L"three", L"four"};

  // create from buffer
  auto tensor = TensorString::CreateFromArray(shape, data);
  WINML_EXPECT_EQUAL(tensor.GetAsVectorView().Size(), data.size());
  WINML_EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

  // create from vector view
  auto dataCopy = data;
  tensor = TensorString::CreateFromIterable(
    shape, winrt::single_threaded_vector<winrt::hstring>(std::move(dataCopy)).GetView()
  );
  WINML_EXPECT_EQUAL(tensor.GetAsVectorView().Size(), data.size());
  WINML_EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"id-tensor-string.onnx", learningModel));
  LearningModelSession session(learningModel);

  auto outputTensor = TensorString::Create();

  std::map<hstring, wf::IInspectable> featuresstandardmap;
  featuresstandardmap[L"X"] = tensor;
  featuresstandardmap[L"Y"] = outputTensor;
  auto featureswinrtmap = winrt::single_threaded_map(std::move(featuresstandardmap));
  session.EvaluateFeatures(featureswinrtmap, L"0");

  // verify identity model round-trip works
  WINML_EXPECT_EQUAL(outputTensor.GetAsVectorView().Size(), data.size());
  WINML_EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(outputTensor.GetAsVectorView())));
}

static void EvaluateFeaturesAsync() {
  std::vector<int64_t> shape = {4};
  std::vector<winrt::hstring> data = {L"one", L"two", L"three", L"four"};

  // create from buffer
  auto tensor = TensorString::CreateFromArray(shape, data);
  WINML_EXPECT_EQUAL(tensor.GetAsVectorView().Size(), data.size());
  WINML_EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

  // create from vector view
  auto dataCopy = data;
  tensor = TensorString::CreateFromIterable(
    shape, winrt::single_threaded_vector<winrt::hstring>(std::move(dataCopy)).GetView()
  );
  WINML_EXPECT_EQUAL(tensor.GetAsVectorView().Size(), data.size());
  WINML_EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"id-tensor-string.onnx", learningModel));
  LearningModelSession session(learningModel);

  auto outputTensor = TensorString::Create(shape);

  std::map<hstring, wf::IInspectable> featuresstandardmap;
  featuresstandardmap[L"X"] = tensor;
  featuresstandardmap[L"Y"] = outputTensor;
  auto featureswinrtmap = winrt::single_threaded_map(std::move(featuresstandardmap));
  session.EvaluateFeaturesAsync(featureswinrtmap, L"0").get();

  // verify identity model round-trip works
  WINML_EXPECT_EQUAL(outputTensor.GetAsVectorView().Size(), data.size());
  WINML_EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(outputTensor.GetAsVectorView())));
}

static void EvaluationProperties() {
  // load a model
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));
  // create a session
  LearningModelSession learningModelSession = nullptr;
  learningModelSession = LearningModelSession(learningModel);
  // set a property
  auto value = winrt::Windows::Foundation::PropertyValue::CreateBoolean(true);
  learningModelSession.EvaluationProperties().Insert(L"propName1", value);
  // get the property and make sure it's there with the right value
  auto value2 = learningModelSession.EvaluationProperties().Lookup(L"propName1");
  WINML_EXPECT_EQUAL(value2.as<IPropertyValue>().GetBoolean(), true);
}

static LearningModelSession CreateSession(LearningModel model) {
  LearningModelDevice device(nullptr);
  WINML_EXPECT_NO_THROW(device = LearningModelDevice(LearningModelDeviceKind::DirectX));

  LearningModelSession session(nullptr);
  if (CommonDeviceHelpers::IsFloat16Supported(device)) {
    WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device));
  } else {
    WINML_EXPECT_THROW_SPECIFIC(
      session = LearningModelSession(model, device),
      winrt::hresult_error,
      [](const winrt::hresult_error& e) -> bool { return e.code() == DXGI_ERROR_UNSUPPORTED; }
    );
  }

  return session;
}

static void CreateSessionWithCastToFloat16InModel() {
  // load a model
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"fp16-truncate-with-cast.onnx", learningModel));

  CreateSession(learningModel);
}

static void CreateSessionWithFloat16InitializersInModel() {
  // load a model
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"fp16-initializer.onnx", learningModel));

  CreateSession(learningModel);
}

static void EvaluateSessionAndCloseModelHelper(LearningModelDeviceKind kind, bool close_model_on_session_creation) {
  auto shape = std::vector<int64_t>{1, 1000};

  auto model = ProtobufHelpers::CreateModel(TensorKind::Float, shape, 1000);

  auto device = LearningModelDevice(kind);
  auto options = LearningModelSessionOptions();

  // close the model on session creation
  options.CloseModelOnSessionCreation(close_model_on_session_creation);

  // ensure you can create a session from the model
  LearningModelSession session(nullptr);

  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device, options));

  std::vector<float> input(1000);
  std::iota(std::begin(input), std::end(input), 0.0f);
  auto tensor_input = TensorFloat::CreateFromArray(shape, input);
  auto binding = LearningModelBinding(session);
  binding.Bind(L"input", tensor_input);

  LearningModelEvaluationResult result(nullptr);
  WINML_EXPECT_NO_THROW(result = session.Evaluate(binding, L""));

  if (close_model_on_session_creation) {
    // ensure that the model has been closed
    WINML_EXPECT_THROW_SPECIFIC(
      LearningModelSession(model, device, options),
      winrt::hresult_error,
      [](const winrt::hresult_error& e) -> bool { return e.code() == E_INVALIDARG; }
    );
  } else {
    WINML_EXPECT_NO_THROW(LearningModelSession(model, device, options));
  }
}

static void EvaluateSessionAndCloseModel() {
  WINML_EXPECT_NO_THROW(::EvaluateSessionAndCloseModelHelper(LearningModelDeviceKind::Cpu, true));
  WINML_EXPECT_NO_THROW(::EvaluateSessionAndCloseModelHelper(LearningModelDeviceKind::Cpu, false));
}

static void NamedDimensionOverride() {
  LearningModel model = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"fns-candy.onnx", model));

  LearningModelDevice device(nullptr);
  WINML_EXPECT_NO_THROW(device = LearningModelDevice(LearningModelDeviceKind::Cpu));

  // the model input shape. the batch size, n, is overriden to 5
  uint32_t n = 5;
  int64_t c = 3, h = 720, w = 720;

  LearningModelSessionOptions options;
  options.OverrideNamedDimension(L"None", n);

  // Verifies that if a Dim name doesn't exist the named dimension override does not interfere with successful evaluation
  // The override is still expected to be present in the internal onnxruntime override data
  options.OverrideNamedDimension(L"DimNameThatDoesntExist", n);

  LearningModelSession session(nullptr);
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device, options));

#ifndef BUILD_INBOX
  Experimental::LearningModelSessionExperimental experimental_session(session);
  Experimental::LearningModelSessionOptionsExperimental experimental_options = experimental_session.Options();
  wfc::IMapView<winrt::hstring, uint32_t> internal_overrides = experimental_options.GetNamedDimensionOverrides();

  WINML_EXPECT_EQUAL(internal_overrides.Lookup(L"None"), n);
  WINML_EXPECT_EQUAL(internal_overrides.Lookup(L"DimNameThatDoesntExist"), n);
#endif

  ILearningModelFeatureDescriptor descriptor = model.InputFeatures().GetAt(0);
  TensorFeatureDescriptor tensorDescriptor = nullptr;
  descriptor.as(tensorDescriptor);
  std::vector<int64_t> shape{n, c, h, w};
  int64_t size = n * c * h * w;
  std::vector<float> buffer;
  buffer.resize(static_cast<size_t>(size));
  auto featureValue = TensorFloat::CreateFromIterable(shape, winrt::single_threaded_vector<float>(std::move(buffer)));
  LearningModelBinding binding(session);
  binding.Bind(descriptor.Name(), featureValue);

  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

static void CloseSession() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));
  LearningModelSession session = nullptr;

  /*
    HANDLE currentProcessHandle = NULL;
    try
    {
        currentProcessHandle = GetCurrentProcess();
    }
    catch (...)
    {
        VERIFY_FAIL(L"Failed to get current process handle.");
    }
    PROCESS_MEMORY_COUNTERS pmc = { 0 };
    SIZE_T beforeSessionCloseWorkingSetSize = 0;
    SIZE_T afterSessionCloseWorkingSetSize = 0;
    bool getProcessMemoryInfoSuccess = false;
    */
  WINML_EXPECT_NO_THROW(session = LearningModelSession(learningModel));

  /*
    // Get the current process memory info after session creation.
    getProcessMemoryInfoSuccess = GetProcessMemoryInfo(currentProcessHandle, &pmc, sizeof(pmc));
    if (!getProcessMemoryInfoSuccess)
    {
        VERIFY_FAIL(L"Failed to get current process memory info.");
    }
    beforeSessionCloseWorkingSetSize = pmc.WorkingSetSize;
    pmc = { 0 };
    */
  WINML_EXPECT_NO_THROW(session.Close());

  /*
    Bug 23659026: Working set difference tolerance is too tight for LearningModelSessionAPITests::CloseSession
    https://microsoft.visualstudio.com/OS/_workitems/edit/23659026

    // Check that working set size has dropped after session close
    getProcessMemoryInfoSuccess = GetProcessMemoryInfo(currentProcessHandle, &pmc, sizeof(pmc));
    if (!getProcessMemoryInfoSuccess)
    {
        VERIFY_FAIL(L"Failed to get current process memory info.");
    }
    afterSessionCloseWorkingSetSize = pmc.WorkingSetSize;
    pmc = { 0 };

    // expected working set difference of session close. It is approximately 2x the size of the weights of model.onnx
    // there needs to be a tolerance because the working set difference varies from run to run.

    // Bug 23739697: Closing Session API in LearningModelSessionAPITests::CloseSession doesn't always result in ~2x working set memory reduction.
    // https://microsoft.visualstudio.com/OS/_workitems/edit/23739697
    float tolerance = 0.4f;
    int64_t expectedWorkingSetDifference = 9662464;
    VERIFY_IS_LESS_THAN(expectedWorkingSetDifference - (beforeSessionCloseWorkingSetSize - afterSessionCloseWorkingSetSize), expectedWorkingSetDifference * tolerance);
    */

  // verify that model still has metadata info after session close
  std::wstring author(learningModel.Author());
  WINML_EXPECT_EQUAL(author, L"onnx-caffe2");

  // verify that session throws RO_E_CLOSED error
  std::vector<float> input(1 * 3 * 224 * 224, 0);
  std::vector<int64_t> shape = {1, 3, 224, 224};
  auto tensor_input = TensorFloat::CreateFromArray(shape, input);
  WINML_EXPECT_THROW_SPECIFIC(
    LearningModelBinding binding(session),
    winrt::hresult_error,
    [](const winrt::hresult_error& e) -> bool { return e.code() == RO_E_CLOSED; }
  );
}

#if !defined(BUILD_INBOX)
static void WindowFunction(const wchar_t* window_operator_name, TensorKind kind, const std::vector<float>& expected) {
  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> output_shape = {32};
  auto double_data_type = TensorInt64Bit::CreateFromArray({}, {11});

  auto window_operator = Operator(window_operator_name).SetInput(L"size", L"Input").SetOutput(L"output", L"Output");

  if (kind == TensorKind::Double) {
    window_operator.SetAttribute(L"output_datatype", double_data_type);
  }

  auto model = LearningModelBuilder::Create(17)
                 .Inputs()
                 .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input", TensorKind::Int64, scalar_shape))
                 .Outputs()
                 .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", kind, output_shape))
                 .Operators()
                 .Add(window_operator)
                 .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  binding.Bind(L"Input", TensorInt64Bit::CreateFromArray(scalar_shape, {32}));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // Check results
  constexpr float error_threshold = .001f;
  if (kind == TensorKind::Float) {
    auto y_tensor = result.Outputs().Lookup(L"Output").as<TensorFloat>();
    auto y_ivv = y_tensor.GetAsVectorView();
    for (int i = 0; i < output_shape[0]; i++) {
      WINML_EXPECT_TRUE(abs(y_ivv.GetAt(i) - expected[i]) < error_threshold);
    }
  }
  if (kind == TensorKind::Double) {
    auto y_tensor = result.Outputs().Lookup(L"Output").as<TensorDouble>();
    auto y_ivv = y_tensor.GetAsVectorView();
    for (int i = 0; i < output_shape[0]; i++) {
      WINML_EXPECT_TRUE(abs(y_ivv.GetAt(i) - expected[i]) < error_threshold);
    }
  }
  printf("\n");
}
#endif

static void SaveSoftwareBitmap(const wchar_t* filename, winrt::Windows::Graphics::Imaging::SoftwareBitmap bitmap) {
  std::wstring modulePath = FileHelpers::GetModulePath();
  winrt::Windows::Storage::StorageFolder folder =
    winrt::Windows::Storage::StorageFolder::GetFolderFromPathAsync(modulePath).get();
  winrt::Windows::Storage::StorageFile file =
    folder.CreateFileAsync(filename, winrt::Windows::Storage::CreationCollisionOption::ReplaceExisting).get();
  winrt::Windows::Storage::Streams::IRandomAccessStream write_stream =
    file.OpenAsync(winrt::Windows::Storage::FileAccessMode::ReadWrite).get();
  winrt::Windows::Graphics::Imaging::BitmapEncoder encoder =
    winrt::Windows::Graphics::Imaging::BitmapEncoder::CreateAsync(
      winrt::Windows::Graphics::Imaging::BitmapEncoder::JpegEncoderId(), write_stream
    )
      .get();
  encoder.SetSoftwareBitmap(bitmap);
  encoder.FlushAsync().get();
}

#if !defined(BUILD_INBOX)
static void DiscreteFourierTransform_2D(LearningModelDeviceKind kind) {
  using namespace winrt::Windows::Storage;
  using namespace winrt::Windows::Storage::Streams;
  using namespace winrt::Windows::Graphics::Imaging;
  using namespace winrt::Windows::Media;
  std::wstring fullImagePath = FileHelpers::GetModulePath() + L"kitten_224.png";

  winrt::Windows::Storage::StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
  IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
  VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

  auto corrected_image = winrt::Windows::Media::VideoFrame(
    winrt::Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8, INT32(256), INT32(256)
  );

  frame.CopyToAsync(corrected_image).get();

  auto width = corrected_image.SoftwareBitmap().PixelWidth();
  auto height = corrected_image.SoftwareBitmap().PixelHeight();

  std::vector<int64_t> input_shape = {1, 1, height, width};
  std::vector<int64_t> output_shape = {1, 1, height, width};

  printf("N-Dimensional Discrete Fourier Transform");
  printf("\n  Input Shape: [");
  for (size_t i = 0; i < input_shape.size(); i++) {
    printf("%d,", static_cast<int>(input_shape[i]));
  }
  printf("]");
  printf("\n  Expected Output Shape: [");
  for (size_t i = 0; i < output_shape.size(); i++) {
    printf("%d,", static_cast<int>(output_shape[i]));
  }
  printf("]");
  printf("\n  Axis: [1,2]");
  printf("\n  Is Onesided: false");

  auto builder =
    LearningModelBuilder::Create(17)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.Signal", TensorKind::Float, input_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Spectra", TensorKind::Float, output_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Inverse", TensorKind::Float, output_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Error", TensorKind::Float, output_shape))
      .Operators()
      .Add(Operator(L"Reshape")
             .SetInput(L"data", L"Input.Signal")
             .SetConstant(
               L"shape", TensorInt64Bit::CreateFromArray({4}, {INT64(1), INT64(height), INT64(width), INT64(1)})
             )
             .SetOutput(L"reshaped", L"reshaped_output"))
      .Operators()
      .Add(Operator(L"DFT")
             .SetInput(L"input", L"reshaped_output")
             .SetAttribute(L"axis", TensorInt64Bit::CreateFromArray({}, {INT64(1)}))
             .SetOutput(L"output", L"DFT.Output.1"))
      .Operators()
      .Add(Operator(L"DFT")
             .SetInput(L"input", L"DFT.Output.1")
             .SetAttribute(L"axis", TensorInt64Bit::CreateFromArray({}, {INT64(2)}))
             .SetOutput(L"output", L"DFT.Output.2"))
      .Operators()
      .Add(Operator(L"DFT")
             .SetInput(L"input", L"DFT.Output.2")
             .SetAttribute(L"axis", TensorInt64Bit::CreateFromArray({}, {INT64(2)}))
             .SetAttribute(L"inverse", TensorInt64Bit::CreateFromArray({}, {INT64(1)}))
             .SetOutput(L"output", L"IDFT.Output.1"))
      .Operators()
      .Add(Operator(L"DFT")
             .SetInput(L"input", L"IDFT.Output.1")
             .SetAttribute(L"axis", TensorInt64Bit::CreateFromArray({}, {INT64(1)}))
             .SetAttribute(L"inverse", TensorInt64Bit::CreateFromArray({}, {INT64(1)}))
             .SetOutput(L"output", L"IDFT.Output.2"))
      .Operators()
      .Add(Operator(L"ReduceSumSquare")
             .SetInput(L"data", L"DFT.Output.2")
             .SetAttribute(L"axes", TensorInt64Bit::CreateFromArray({1}, {3}))
             .SetAttribute(L"keepdims", TensorInt64Bit::CreateFromArray({}, {0}))
             .SetOutput(L"reduced", L"magnitude_squared"))
      .Operators()
      .Add(Operator(L"Sqrt").SetInput(L"X", L"magnitude_squared").SetOutput(L"Y", L"sqrt_magnitude"))
      .Operators()
      .Add(Operator(L"ReduceSumSquare")
             .SetInput(L"data", L"IDFT.Output.2")
             .SetAttribute(L"axes", TensorInt64Bit::CreateFromArray({1}, {3}))
             .SetAttribute(L"keepdims", TensorInt64Bit::CreateFromArray({}, {0}))
             .SetOutput(L"reduced", L"magnitude_squared2"))
      .Operators()
      .Add(Operator(L"Sqrt").SetInput(L"X", L"magnitude_squared2").SetOutput(L"Y", L"sqrt_magnitude2"))
      .Operators()
      .Add(Operator(L"Reshape")
             .SetInput(L"data", L"sqrt_magnitude")
             .SetConstant(
               L"shape", TensorInt64Bit::CreateFromArray({4}, {INT64(1), INT64(1), INT64(height), INT64(width)})
             )
             .SetOutput(L"reshaped", L"Output.Spectra"))
      .Operators()
      .Add(Operator(L"Reshape")
             .SetInput(L"data", L"sqrt_magnitude2")
             .SetConstant(
               L"shape", TensorInt64Bit::CreateFromArray({4}, {INT64(1), INT64(1), INT64(height), INT64(width)})
             )
             .SetOutput(L"reshaped", L"Output.Inverse"))
      .Operators()
      .Add(Operator(L"Sub")
             .SetInput(L"A", L"Input.Signal")
             .SetInput(L"B", L"Output.Inverse")
             .SetOutput(L"C", L"Output.Error"));

  auto model = builder.CreateModel();
  auto device = LearningModelDevice(kind);
  LearningModelSession session(model, device);
  LearningModelBinding binding(session);

  // Bind input
  binding.Bind(L"Input.Signal", frame);

  // Bind output
  auto spectra = VideoFrame(BitmapPixelFormat::Bgra8, INT32(width), INT32(height));
  binding.Bind(L"Output.Spectra", spectra);
  auto inverse = VideoFrame(BitmapPixelFormat::Bgra8, INT32(width), INT32(height));
  binding.Bind(L"Output.Inverse", inverse);

  // Evaluate
  auto start = std::chrono::high_resolution_clock::now();
  auto result = session.Evaluate(binding, L"");
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf("\n  Evaluate Took: %fus\n", evaluate_duration_in_microseconds.count());

  auto error = result.Outputs().Lookup(L"Output.Error").as<TensorFloat>();
  auto error_ivv = error.GetAsVectorView();
  for (auto i = 0; i < height * width; i++) {
    constexpr float error_threshold = .001f;
    WINML_EXPECT_TRUE(abs(error_ivv.GetAt(i)) < error_threshold);
  }

  /*
  * Output input, output and model
  SaveSoftwareBitmap(L"fft2d.jpg", spectra.SoftwareBitmap());
  SaveSoftwareBitmap(L"fft2d_inverse.jpg", inverse.SoftwareBitmap());
  builder.Save(L"fft2d.onnx");
  */

  printf("\n");
}

template <typename T>
static void DiscreteFourierTransform(
  LearningModelDeviceKind kind,
  const std::vector<T>& input,
  const std::vector<int64_t>& input_shape,
  const std::vector<std::complex<float>>& expected_output,
  size_t axis,
  size_t dft_length,
  bool is_onesided = false
) {
  // Calculate expected output shape
  auto output_shape = input_shape;
  if (output_shape.size() != 2) {
    // If the input is not 2 dimensional, the last dimension is the complex component.
    // DFT should always output complex results, and so we can comfortably coerce the last dim to 2
    output_shape[output_shape.size() - 1] = 2;
  } else {
    // DFT should always output complex results. If input was 2 dimensional (real), we can comfortably append the last dim as 2
    output_shape.push_back(2);
  }
  output_shape[axis] = is_onesided ? (1 + (dft_length >> 1)) : dft_length;

  printf("Discrete Fourier Transform");
  printf("\n  Input Shape: [");
  for (size_t i = 0; i < input_shape.size(); i++) {
    printf("%d,", static_cast<int>(input_shape[i]));
  }
  printf("]");
  printf("\n  Expected Output Shape: [");
  for (size_t i = 0; i < output_shape.size(); i++) {
    printf("%d,", static_cast<int>(output_shape[i]));
  }
  printf("]");
  printf("\n  Axis: %d", static_cast<int>(axis));
  printf("\n  DFT Length: %d", static_cast<int>(dft_length));
  printf("\n  Is Onesided: %s", is_onesided ? "true" : "false");

  auto model =
    LearningModelBuilder::Create(17)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.Signal", TensorKind::Float, input_shape))
      .Inputs()
      .AddConstant(L"Input.DFTLength", TensorInt64Bit::CreateFromArray({}, {INT64(dft_length)}))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Spectra", TensorKind::Float, output_shape))
      .Operators()
      .Add(Operator(L"DFT")
             .SetInput(L"input", L"Input.Signal")
             .SetInput(L"dft_length", L"Input.DFTLength")
             .SetAttribute(L"axis", TensorInt64Bit::CreateFromArray({}, {INT64(axis)}))
             .SetAttribute(L"onesided", TensorInt64Bit::CreateFromArray({}, {is_onesided}))
             .SetOutput(L"output", L"Output.Spectra"))
      .CreateModel();
  auto device = LearningModelDevice(kind);
  LearningModelSession session(model, device);
  LearningModelBinding binding(session);

  auto is_real_input = input_shape.size() == 2 || input_shape[input_shape.size() - 1] == 1;
  uint32_t input_stride = is_real_input ? 1 : 2;

  // Populate binding
  auto input_begin = const_cast<float*>(reinterpret_cast<const float*>(input.data()));
  auto input_floats = winrt::array_view<float>(input_begin, static_cast<uint32_t>(input.size() * input_stride));
  binding.Bind(L"Input.Signal", TensorFloat::CreateFromArray(input_shape, input_floats));

  // Evaluate
  auto start = std::chrono::high_resolution_clock::now();
  auto result = session.Evaluate(binding, L"");
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf("\n  Evaluate Took: %fus", evaluate_duration_in_microseconds.count());

  // Check results
  auto y_tensor = result.Outputs().Lookup(L"Output.Spectra").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  for (uint32_t i = 0; i < y_ivv.Size(); i += 2) {
    // Check results
    constexpr float error_threshold = .001f;

    auto inRealRange = abs(y_ivv.GetAt(i) - expected_output[i / 2].real()) < error_threshold;
    auto inImagRange = abs(y_ivv.GetAt(i + 1) - expected_output[i / 2].imag()) < error_threshold;
    auto inRange = inRealRange && inImagRange;

    if (!inRange) {
      printf(
        "[%d] ACTUAL(%f  +  %fi) EXPECTED(%f  +  %fi)\n",
        (int)i / 2,
        y_ivv.GetAt(i),
        y_ivv.GetAt(i + 1),
        expected_output[i / 2].real(),
        expected_output[i / 2].imag()
      );
    }

    WINML_EXPECT_TRUE(inRange);
  }
  printf("\n\n");
}

#endif

template <typename T>
static auto MakePureFrequency(float frequency_in_hertz, size_t signal_size, size_t sample_rate) {
  float amplitude = 4;
  float angular_velocity = frequency_in_hertz * 2 * static_cast<float>(M_PI);
  std::vector<T> signal(signal_size);
  for (size_t i = 0; i < signal_size; i++) {
    T time = i / static_cast<T>(sample_rate);
    signal[i] = amplitude * cos(angular_velocity * time);
  }
  return signal;
}

template <typename T>
static auto MakeMiddleC(size_t signal_size, size_t sample_rate) {
  float middle_c_in_hertz = 261.626f;
  return MakePureFrequency<T>(middle_c_in_hertz, signal_size, sample_rate);
}

template <typename T>
static auto MakeC2(size_t signal_size, size_t sample_rate) {
  float middle_c_in_hertz = 261.626f * 2;
  return MakePureFrequency<T>(middle_c_in_hertz, signal_size, sample_rate);
}

template <typename T>
static auto MakeC4(size_t signal_size, size_t sample_rate) {
  float middle_c_in_hertz = 261.626f * 4;
  return MakePureFrequency<T>(middle_c_in_hertz, signal_size, sample_rate);
}

template <typename T>
static auto MakeThreeTones(size_t signal_size, size_t sample_rate) {
  auto middle_c = MakeMiddleC<T>(signal_size, sample_rate);
  auto c2 = MakeC2<T>(signal_size, sample_rate);
  auto c4 = MakeC4<T>(signal_size, sample_rate);
  for (size_t i = 0; i < signal_size; i++) {
    middle_c[i] = (i < signal_size / 3) ? middle_c[i]
      : (i < 2 * signal_size / 3)       ? (middle_c[i] + c2[i])
                                        : (middle_c[i] + c2[i] + c4[i]);
  }
  return middle_c;
}

#if !defined(BUILD_INBOX)
static void STFT(
  size_t batch_size, size_t signal_size, size_t dft_size, size_t hop_size, size_t sample_rate, bool is_onesided = false
) {
  auto n_dfts = static_cast<size_t>(1 + floor((signal_size - dft_size) / hop_size));
  auto input_shape = std::vector<int64_t>{1, INT64(signal_size)};
  auto output_shape = std::vector<int64_t>{
    INT64(batch_size), INT64(n_dfts), is_onesided ? ((INT64(dft_size) >> 1) + 1) : INT64(dft_size), 2
  };
  auto dft_length = TensorInt64Bit::CreateFromArray({}, {INT64(dft_size)});

  auto model =
    LearningModelBuilder::Create(17)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.TimeSignal", TensorKind::Float, input_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.STFT", TensorKind::Float, output_shape))
      .Outputs()
      .Add(
        LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.HannWindow", TensorKind::Float, {INT64(dft_size)})
      )
      .Operators()
      .Add(Operator(L"HannWindow").SetConstant(L"size", dft_length).SetOutput(L"output", L"Output.HannWindow"))
      .Operators()
      .Add(Operator(L"STFT")
             .SetAttribute(L"onesided", TensorInt64Bit::CreateFromArray({}, {INT64(is_onesided)}))
             .SetInput(L"signal", L"Input.TimeSignal")
             .SetInput(L"window", L"Output.HannWindow")
             .SetConstant(L"frame_length", dft_length)
             .SetConstant(L"frame_step", TensorInt64Bit::CreateFromArray({}, {INT64(hop_size)}))
             .SetOutput(L"output", L"Output.STFT"))
      .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Create signal binding
  auto signal = MakeMiddleC<float>(signal_size, sample_rate);
  //printf("\n");
  //printf("Input.TimeSignal:\n");
  //for (size_t i = 0; i < dft_size; i++) {
  //  printf("%f, ", signal[i]);
  //}

  // Bind
  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromArray(input_shape, signal));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  /*
  printf("\n");
  printf("Output.HannWindow\n");
  auto window_tensor = result.Outputs().Lookup(L"Output.HannWindow").as<TensorFloat>();
  auto window_ivv = window_tensor.GetAsVectorView();
  for (uint32_t i = 0; i < window_ivv.Size(); i++) {
    printf("%f, ", window_ivv.GetAt(i));
  }
  printf("\n");

  printf("Output.STFT\n");
  // Check results
  auto y_tensor = result.Outputs().Lookup(L"Output.STFT").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  auto size = y_ivv.Size();
  WINML_EXPECT_EQUAL(size, n_dfts * output_shape[2] * 2);
  for (size_t dft_idx = 0; dft_idx < n_dfts; dft_idx++) {
    for (size_t i = 0; INT64(i) < output_shape[2]; i++) {
      auto real_idx = static_cast<uint32_t>((i * 2) + (2 * dft_idx * output_shape[2]));
      printf("(%d, %f , %fi), ", static_cast<uint32_t>(i), y_ivv.GetAt(real_idx), y_ivv.GetAt(real_idx + 1));
    }
  }

  printf("\n");
  */
}
#endif

static void ModelBuilding_MelWeightMatrix() {
#if !defined(BUILD_INBOX)
  std::vector<int64_t> output_shape = {INT64(9), INT64(8)};
  auto builder = LearningModelBuilder::Create(17)
                   .Outputs()
                   .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(
                     L"Output.MelWeightMatrix", TensorKind::Float, output_shape
                   ))
                   .Operators()
                   .Add(Operator(L"MelWeightMatrix")
                          .SetConstant(L"num_mel_bins", TensorInt64Bit::CreateFromArray({}, {INT64(8)}))
                          .SetConstant(L"dft_length", TensorInt64Bit::CreateFromArray({}, {INT64(16)}))
                          .SetConstant(L"sample_rate", TensorInt64Bit::CreateFromArray({}, {INT64(8192)}))
                          .SetConstant(L"lower_edge_hertz", TensorFloat::CreateFromArray({}, {0}))
                          .SetConstant(L"upper_edge_hertz", TensorFloat::CreateFromArray({}, {8192 / 2.f}))
                          .SetOutput(L"output", L"Output.MelWeightMatrix"));
  auto model = builder.CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  auto result = session.Evaluate(binding, L"");

  /*
  printf("\n");
  printf("Output.MelWeightMatrix\n");
  {
    auto y_tensor = result.Outputs().Lookup(L"Output.MelWeightMatrix").as<TensorFloat>();
    auto y_ivv = y_tensor.GetAsVectorView();
    for (unsigned i = 0; i < y_ivv.Size(); i++) {
      printf("%f, ", y_ivv.GetAt(i));
    }
  }
  */

  printf("\n");
#endif
}

#if !defined(BUILD_INBOX)
static void MelSpectrogramOnThreeToneSignal(
  size_t batch_size,
  size_t signal_size,
  size_t window_size,
  size_t dft_size,
  size_t hop_size,
  size_t n_mel_bins,
  size_t sampling_rate
) {
  auto n_dfts = static_cast<size_t>(1 + floor((signal_size - dft_size) / hop_size));
  auto onesided_dft_size = (dft_size >> 1) + 1;
  std::vector<int64_t> signal_shape = {INT64(batch_size), INT64(signal_size)};
  std::vector<int64_t> mel_spectrogram_shape = {INT64(batch_size), 1, INT64(n_dfts), INT64(n_mel_bins)};

  auto builder =
    LearningModelBuilder::Create(17)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.TimeSignal", TensorKind::Float, signal_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(
        L"Output.MelSpectrogram", TensorKind::Float, mel_spectrogram_shape
      ))
      .Operators()
      .Add(Operator(L"HannWindow")
             .SetConstant(L"size", TensorInt64Bit::CreateFromArray({}, {INT64(window_size)}))
             .SetOutput(L"output", L"hann_window"))
      .Operators()
      .Add(Operator(L"STFT")
             .SetName(L"STFT_NAMED_NODE")
             .SetInput(L"signal", L"Input.TimeSignal")
             .SetInput(L"window", L"hann_window")
             .SetConstant(L"frame_length", TensorInt64Bit::CreateFromArray({}, {INT64(dft_size)}))
             .SetConstant(L"frame_step", TensorInt64Bit::CreateFromArray({}, {INT64(hop_size)}))
             .SetOutput(L"output", L"stft_output"))
      .Operators()
      .Add(Operator(L"ReduceSumSquare")
             .SetInput(L"data", L"stft_output")
             .SetAttribute(L"axes", TensorInt64Bit::CreateFromArray({1}, {3}))
             .SetAttribute(L"keepdims", TensorInt64Bit::CreateFromArray({}, {0}))
             .SetOutput(L"reduced", L"magnitude_squared"))
      .Operators()
      .Add(Operator(L"Div")
             .SetInput(L"A", L"magnitude_squared")
             .SetConstant(L"B", TensorFloat::CreateFromArray({}, {static_cast<float>(dft_size)}))
             .SetOutput(L"C", L"power_frames"))
      .Operators()
      .Add(Operator(L"MelWeightMatrix")
             .SetConstant(L"num_mel_bins", TensorInt64Bit::CreateFromArray({}, {INT64(n_mel_bins)}))
             .SetConstant(L"dft_length", TensorInt64Bit::CreateFromArray({}, {INT64(dft_size)}))
             .SetConstant(L"sample_rate", TensorInt64Bit::CreateFromArray({}, {INT64(sampling_rate)}))
             .SetConstant(L"lower_edge_hertz", TensorFloat::CreateFromArray({}, {0}))
             .SetConstant(L"upper_edge_hertz", TensorFloat::CreateFromArray({}, {sampling_rate / 2.f}))
             .SetOutput(L"output", L"mel_weight_matrix"))
      .Operators()
      .Add(Operator(L"Reshape")
             .SetInput(L"data", L"power_frames")
             .SetConstant(
               L"shape", TensorInt64Bit::CreateFromArray({2}, {INT64(batch_size * n_dfts), INT64(onesided_dft_size)})
             )
             .SetOutput(L"reshaped", L"reshaped_output"))
      .Operators()
      .Add(Operator(L"MatMul")
             .SetInput(L"A", L"reshaped_output")
             .SetInput(L"B", L"mel_weight_matrix")
             .SetOutput(L"Y", L"mel_spectrogram"))
      .Operators()
      .Add(Operator(L"Reshape")
             .SetInput(L"data", L"mel_spectrogram")
             .SetConstant(L"shape", TensorInt64Bit::CreateFromArray({4}, mel_spectrogram_shape))
             .SetOutput(L"reshaped", L"Output.MelSpectrogram"));
  auto model = builder.CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Bind input
  auto signal = MakeThreeTones<float>(signal_size, sampling_rate);
  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromArray(signal_shape, signal));

  // Bind output
  auto output_image = winrt::Windows::Media::VideoFrame(
    winrt::Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8, INT32(n_mel_bins), INT32(n_dfts)
  );
  binding.Bind(L"Output.MelSpectrogram", output_image);

  // Evaluate
  auto start = std::chrono::high_resolution_clock::now();
  auto result = session.Evaluate(binding, L"");
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf("Evaluate Took: %fus\n", evaluate_duration_in_microseconds.count());

  // Check the output video frame object by saving output image to disk
  std::wstring out_name = L"mel_spectrogram.jpg";

  // Save the output
  std::wstring modulePath = FileHelpers::GetModulePath();
  winrt::Windows::Storage::StorageFolder folder =
    winrt::Windows::Storage::StorageFolder::GetFolderFromPathAsync(modulePath).get();
  winrt::Windows::Storage::StorageFile file =
    folder.CreateFileAsync(out_name, winrt::Windows::Storage::CreationCollisionOption::ReplaceExisting).get();
  winrt::Windows::Storage::Streams::IRandomAccessStream write_stream =
    file.OpenAsync(winrt::Windows::Storage::FileAccessMode::ReadWrite).get();
  winrt::Windows::Graphics::Imaging::BitmapEncoder encoder =
    winrt::Windows::Graphics::Imaging::BitmapEncoder::CreateAsync(
      winrt::Windows::Graphics::Imaging::BitmapEncoder::JpegEncoderId(), write_stream
    )
      .get();
  encoder.SetSoftwareBitmap(output_image.SoftwareBitmap());
  encoder.FlushAsync().get();

  // Save the model
  builder.Save(L"spectrogram.onnx");
  printf("\n");
}
#endif

static void ModelBuilding_StandardDeviationNormalization() {
#ifndef BUILD_INBOX
  int64_t height = 256;
  int64_t width = 256;
  int64_t channels = 3;
  std::vector<int64_t> input_shape = {1, height, width, channels};
  std::vector<int64_t> output_shape = {1, channels, height, width};
  auto sub_model = LearningModelBuilder::Create(13)
                     .Inputs()
                     .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(
                       L"Input", L"The NHWC image", TensorKind::Float, input_shape
                     ))
                     .Inputs()
                     .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Means", TensorKind::Float, {channels}))
                     .Outputs()
                     .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(
                       L"Output", L"The NCHW image normalized with mean and stddev.", TensorKind::Float, input_shape
                     ))
                     .Operators()
                     .Add(Operator(L"Sub").SetInput(L"A", L"Input").SetInput(L"B", L"Means").SetOutput(L"C", L"Output"))
                     .CreateModel();
  auto div_model =
    LearningModelBuilder::Create(13)
      .Inputs()
      .Add(
        LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input", L"The NHWC image", TensorKind::Float, input_shape)
      )
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"StdDevs", TensorKind::Float, {channels}))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(
        L"Output", L"The NCHW image normalized with mean and stddev.", TensorKind::Float, input_shape
      ))
      .Operators()
      .Add(Operator(L"Div").SetInput(L"A", L"Input").SetInput(L"B", L"StdDevs").SetOutput(L"C", L"Output"))
      .CreateModel();
  auto transpose_model =
    LearningModelBuilder::Create(13)
      .Inputs()
      .Add(
        LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input", L"The NHWC image", TensorKind::Float, input_shape)
      )
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(
        L"Output", L"The NCHW image normalized with mean and stddev.", TensorKind::Float, output_shape
      ))
      .Operators()
      .Add(Operator(L"Transpose")
             .SetInput(L"data", L"Input")
             .SetAttribute(L"perm", TensorInt64Bit::CreateFromArray({4}, {0, 3, 1, 2}))
             .SetOutput(L"transposed", L"Output"))
      .CreateModel();

  auto sub_experimental = winml_experimental::LearningModelExperimental(sub_model);
  winml_experimental::LearningModelJoinOptions div_join_options;
  div_join_options.Link(sub_model.OutputFeatures().GetAt(0).Name(), div_model.InputFeatures().GetAt(0).Name());
  div_join_options.JoinedNodePrefix(L"DivModel.");
  auto joined_model = sub_experimental.JoinModel(div_model, div_join_options);

  auto joined_model_experimental = winml_experimental::LearningModelExperimental(joined_model);
  winml_experimental::LearningModelJoinOptions transpose_join_options;
  transpose_join_options.Link(
    joined_model.OutputFeatures().GetAt(0).Name(), transpose_model.InputFeatures().GetAt(0).Name()
  );
  transpose_join_options.JoinedNodePrefix(L"TransposeModel.");
  auto final_model = joined_model_experimental.JoinModel(transpose_model, transpose_join_options);

  auto final_model_experimental = winml_experimental::LearningModelExperimental(final_model);
  final_model_experimental.Save(L"ModelBuilding_StandardDeviationNormalization.onnx");

  auto session = LearningModelSession(final_model, LearningModelDevice(LearningModelDeviceKind::Cpu));
  LearningModelBinding binding(session);

  // Bind
  auto input = std::vector<float>(SIZET(height * width * channels), 1);
  binding.Bind(L"Input", TensorFloat::CreateFromArray(input_shape, input));
  auto channels_shape = std::vector<int64_t>(SIZET(1), 3);
  binding.Bind(L"Means", TensorFloat::CreateFromArray(channels_shape, {2, 2, 2}));
  binding.Bind(L"DivModel.StdDevs", TensorFloat::CreateFromArray(channels_shape, {.1f, .1f, .1f}));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

#endif
}

static void ModelBuilding_Gemm() {
#ifndef BUILD_INBOX
  std::vector<int64_t> shape = {3, 3};
  auto model = LearningModelBuilder::Create(13)
                 .Inputs()
                 .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputA", TensorKind::Float, shape))
                 .Inputs()
                 .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputB", TensorKind::Float, shape))
                 .Inputs()
                 .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputC", TensorKind::Float, shape))
                 .Outputs()
                 .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"OutputY", TensorKind::Float, shape))
                 .Operators()
                 .Add(Operator(L"Gemm")
                        .SetInput(L"A", L"InputA")
                        .SetInput(L"B", L"InputB")
                        .SetInput(L"C", L"InputC")
                        .SetOutput(L"Y", L"OutputY"))
                 .CreateModel();
#endif
}

static void ModelBuilding_DynamicMatmul() {
#ifndef BUILD_INBOX
  std::vector<int64_t> a_shape = {318, 129};
  std::vector<int64_t> b_shape = {129, 1024};

  auto model =
    LearningModelBuilder::Create(13)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputA", TensorKind::Float, a_shape))
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputB", TensorKind::Float, b_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", TensorKind::Float, {a_shape[0], b_shape[1]}))
      .Operators()
      .Add(Operator(L"MatMul").SetInput(L"A", L"InputA").SetInput(L"B", L"InputB").SetOutput(L"Y", L"Output"))
      .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Bind A
  auto a_matrix = std::vector<float>(SIZET(a_shape[0] * a_shape[1]), 1);
  binding.Bind(L"InputA", TensorFloat::CreateFromArray(a_shape, a_matrix));

  // Bind B
  auto b_matrix = std::vector<float>(SIZET(b_shape[0] * b_shape[1]), 1);
  binding.Bind(L"InputB", TensorFloat::CreateFromArray(b_shape, b_matrix));

  // Evaluate
  auto start = std::chrono::high_resolution_clock::now();
  auto result = session.Evaluate(binding, L"");
  auto end = std::chrono::high_resolution_clock::now();

  // Print duration
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf("Evaluate Took: %fus\n", evaluate_duration_in_microseconds.count());
#endif
}

static void ModelBuilding_ConstantMatmul() {
#ifndef BUILD_INBOX
  std::vector<int64_t> a_shape = {318, 129};
  std::vector<int64_t> b_shape = {129, 1024};

  auto model =
    LearningModelBuilder::Create(13)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputA", TensorKind::Float, a_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", TensorKind::Float, {a_shape[0], b_shape[1]}))
      .Operators()
      .Add(Operator(L"MatMul")
             .SetInput(L"A", L"InputA")
             .SetConstant(
               L"B", TensorFloat::CreateFromArray(b_shape, std::vector<float>(SIZET(b_shape[0] * b_shape[1]), 1))
             )
             .SetOutput(L"Y", L"Output"))
      .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Bind input
  auto a_matrix = std::vector<float>(SIZET(a_shape[0] * a_shape[1]), 1);
  binding.Bind(L"InputA", TensorFloat::CreateFromArray(a_shape, a_matrix));

  // Evaluate
  auto start = std::chrono::high_resolution_clock::now();
  auto result = session.Evaluate(binding, L"");
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf("Evaluate Took: %fus\n", evaluate_duration_in_microseconds.count());
#endif
}

#if !defined(BUILD_INBOX)

enum class Mode : uint32_t {
  Bilinear,
  Nearest,
  Bicubic,
};

enum class PaddingMode : uint32_t {
  Zeros,
  Border,
  Reflection,
};

template <typename T, typename U>
static void GridSample(
  LearningModelDeviceKind kind,
  const std::vector<T>& input,
  const std::vector<int64_t>& input_dims,
  const std::vector<U>& grid,
  const std::vector<int64_t>& grid_dims,
  bool align_corners,
  Mode mode,
  PaddingMode padding_mode
) {
  const hstring modes[] = {L"bilinear", L"nearest", L"bicubic"};

  const hstring padding_modes[] = {L"zeros", L"border", L"reflection"};

  auto model =
    LearningModelBuilder::Create(17)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input", TensorKind::Float, input_dims))
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Grid", TensorKind::Float, grid_dims))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", TensorKind::Float, {-1, -1, -1, -1}))
      .Operators()
      .Add(Operator(L"GridSample")
             .SetInput(L"X", L"Input")
             .SetInput(L"grid", L"Grid")
             .SetAttribute(L"align_corners", TensorInt64Bit::CreateFromArray({}, {INT64(align_corners)}))
             .SetAttribute(L"mode", TensorString::CreateFromArray({}, {modes[static_cast<uint32_t>(mode)]}))
             .SetAttribute(
               L"padding_mode", TensorString::CreateFromArray({}, {padding_modes[static_cast<uint32_t>(padding_mode)]})
             )
             .SetOutput(L"Y", L"Output"))
      .CreateModel();
  auto cpu_device = LearningModelDevice(LearningModelDeviceKind::Cpu);
  auto device = LearningModelDevice(kind);
  LearningModelSession device_session(model, device);
  LearningModelBinding device_binding(device_session);
  LearningModelSession cpu_session(model, cpu_device);
  LearningModelBinding cpu_binding(cpu_session);

  device_binding.Bind(L"Input", TensorFloat::CreateFromShapeArrayAndDataArray(input_dims, input));
  device_binding.Bind(L"Grid", TensorFloat::CreateFromShapeArrayAndDataArray(grid_dims, grid));
  cpu_binding.Bind(L"Input", TensorFloat::CreateFromShapeArrayAndDataArray(input_dims, input));
  cpu_binding.Bind(L"Grid", TensorFloat::CreateFromShapeArrayAndDataArray(grid_dims, grid));

  auto cpu_result = cpu_session.Evaluate(cpu_binding, L"");

  // Evaluate
  auto start = std::chrono::high_resolution_clock::now();
  auto device_result = device_session.Evaluate(device_binding, L"");
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf(
    "GridSample[Mode=%ls, PaddingMode=%ls, AlignCorners=%s] took %fus.\n",
    modes[static_cast<uint32_t>(mode)].c_str(),
    padding_modes[static_cast<uint32_t>(padding_mode)].c_str(),
    align_corners ? "True" : "False",
    evaluate_duration_in_microseconds.count()
  );

  // Check results
  constexpr float error_threshold = .001f;
  auto device_y_tensor = device_result.Outputs().Lookup(L"Output").as<TensorFloat>();
  auto device_y_ivv = device_y_tensor.GetAsVectorView();
  auto cpu_y_tensor = cpu_result.Outputs().Lookup(L"Output").as<TensorFloat>();
  auto cpu_y_ivv = cpu_y_tensor.GetAsVectorView();
  WINML_EXPECT_EQUAL(device_y_ivv.Size(), cpu_y_ivv.Size());
  for (uint32_t i = 0; i < device_y_ivv.Size(); i++) {
    bool in_range = abs(device_y_ivv.GetAt(i) - cpu_y_ivv.GetAt(i)) < error_threshold;
    if (!in_range) {
      printf("[%d] ACTUAL(%f) EXPECTED(%f)\n", (int)i, device_y_ivv.GetAt(i), cpu_y_ivv.GetAt(i));
    }
    WINML_EXPECT_TRUE(in_range);
  }
}

static void GridSampleRunner(
  LearningModelDeviceKind kind,
  const std::vector<float>& input,
  const std::vector<int64_t>& input_dims,
  const std::vector<float>& grid,
  const std::vector<int64_t>& grid_dims
) {
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Bilinear, PaddingMode::Zeros);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Bilinear, PaddingMode::Border);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Bilinear, PaddingMode::Reflection);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Nearest, PaddingMode::Zeros);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Nearest, PaddingMode::Border);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Nearest, PaddingMode::Reflection);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Bicubic, PaddingMode::Zeros);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Bicubic, PaddingMode::Border);
  GridSample(kind, input, input_dims, grid, grid_dims, false, Mode::Bicubic, PaddingMode::Reflection);

  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Bilinear, PaddingMode::Zeros);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Bilinear, PaddingMode::Border);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Bilinear, PaddingMode::Reflection);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Nearest, PaddingMode::Zeros);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Nearest, PaddingMode::Border);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Nearest, PaddingMode::Reflection);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Bicubic, PaddingMode::Zeros);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Bicubic, PaddingMode::Border);
  GridSample(kind, input, input_dims, grid, grid_dims, true, Mode::Bicubic, PaddingMode::Reflection);
}

static void ModelBuilding_GridSample_Internal(LearningModelDeviceKind kind) {
  std::vector<float> input = {
    0.00f,
    1.00f,
    2.00f,
    3.00f,
    4.00f,
    5.00f,
    6.00f,
    7.00f,
    8.00f,
    9.00f,
    10.00f,
    11.00f,
    12.00f,
    13.00f,
    14.00f,
    15.00f,
  };

  std::vector<float> grid = {
    0.00f,  1.00f,  2.00f,  3.00f,  4.00f,  5.00f,  6.00f,  7.00f,  8.00f,  9.00f,  10.00f, 11.00f, 12.00f,
    13.00f, 14.00f, 15.00f, 16.00f, 17.00f, 18.00f, 19.00f, 20.00f, 21.00f, 22.00f, 23.00f, 24.00f, 25.00f,
    26.00f, 27.00f, 28.00f, 29.00f, 30.00f, 31.00f, 32.00f, 33.00f, 34.00f, 35.00f, 36.00f, 37.00f, 38.00f,
    39.00f, 40.00f, 41.00f, 42.00f, 43.00f, 44.00f, 45.00f, 46.00f, 47.00f, 48.00f, 49.00f,
  };
  std::transform(grid.begin(), grid.end(), grid.begin(), [&](auto& in) { return in / grid.size(); });
  std::vector<int64_t> input_dims = {1, 1, 4, 4};
  std::vector<int64_t> grid_dims = {1, 5, 5, 2};

  GridSampleRunner(kind, input, input_dims, grid, grid_dims);

  input = {0.0f, 1.0f, 2.0f, 3.0f, 4.0, 5.0f};
  grid = {
    -10.0000f,
    -10.0000f,
    -5.0000f,
    -5.0000f,
    -0.2000f,
    -0.2000f,
    10.0000f,
    10.0000f,

    10.0000f,
    10.0000f,
    -0.2000f,
    -0.2000f,
    5.0000f,
    5.0000f,
    10.0000f,
    10.0000f
  };
  input_dims = {1, 1, 3, 2};
  grid_dims = {1, 2, 4, 2};

  GridSampleRunner(kind, input, input_dims, grid, grid_dims);
}

static void ModelBuilding_DiscreteFourierTransform_Internal(LearningModelDeviceKind kind) {
  std::vector<float> real_input = {
    1.00f, 2.00f, 3.00f, 4.00f, 5.00f, 6.00f, 7.00f, 8.00f, 1.00f, 2.00f, 3.00f, 4.00f, 5.00f, 6.00f,
    7.00f, 8.00f, 1.00f, 2.00f, 3.00f, 4.00f, 5.00f, 6.00f, 7.00f, 8.00f, 1.00f, 2.00f, 3.00f, 4.00f,
    5.00f, 6.00f, 7.00f, 8.00f, 1.00f, 2.00f, 3.00f, 4.00f, 5.00f, 6.00f, 7.00f, 8.00f,
  };

  std::vector<std::complex<float>> real_expected_axis_0_two_sided = {
    { 5.000f, 0.000f},
    {10.000f, 0.000f},
    {15.000f, 0.000f},
    {20.000f, 0.000f},
    {25.000f, 0.000f},
    {30.000f, 0.000f},
    {35.000f, 0.000f},
    {40.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    { 0.000f, 0.000f},
    {-0.000f, 0.000f},
    {-0.000f, 0.000f},
    {-0.000f, 0.000f},
    {-0.000f, 0.000f},
    {-0.000f, 0.000f},
    {-0.000f, 0.000f},
    {-0.000f, 0.000f},
    {-0.000f, 0.000f},
  };
  DiscreteFourierTransform(kind, real_input, {1, 5, 8, 1}, real_expected_axis_0_two_sided, 1, 5, false /*onesided*/);

  std::vector<std::complex<float>> real_expected_axis_1_two_sided = {
    {36.000f,  0.000f},
    {-4.000f,  9.657f},
    {-4.000f,  4.000f},
    {-4.000f,  1.657f},
    {-4.000f,  0.000f},
    {-4.000f, -1.657f},
    {-4.000f, -4.000f},
    {-4.000f, -9.657f},
    {36.000f,  0.000f},
    {-4.000f,  9.657f},
    {-4.000f,  4.000f},
    {-4.000f,  1.657f},
    {-4.000f,  0.000f},
    {-4.000f, -1.657f},
    {-4.000f, -4.000f},
    {-4.000f, -9.657f},
    {36.000f,  0.000f},
    {-4.000f,  9.657f},
    {-4.000f,  4.000f},
    {-4.000f,  1.657f},
    {-4.000f,  0.000f},
    {-4.000f, -1.657f},
    {-4.000f, -4.000f},
    {-4.000f, -9.657f},
    {36.000f,  0.000f},
    {-4.000f,  9.657f},
    {-4.000f,  4.000f},
    {-4.000f,  1.657f},
    {-4.000f,  0.000f},
    {-4.000f, -1.657f},
    {-4.000f, -4.000f},
    {-4.000f, -9.657f},
    {36.000f,  0.000f},
    {-4.000f,  9.657f},
    {-4.000f,  4.000f},
    {-4.000f,  1.657f},
    {-4.000f,  0.000f},
    {-4.000f, -1.657f},
    {-4.000f, -4.000f},
    {-4.000f, -9.657f},
  };
  DiscreteFourierTransform(kind, real_input, {1, 5, 8, 1}, real_expected_axis_1_two_sided, 2, 8, false /*onesided*/);

  std::vector<std::complex<float>> input = {
    { 1.00f, 0.00f},
    { 2.00f, 0.00f},
    { 3.00f, 0.00f},
    { 4.00f, 0.00f},
    { 5.00f, 0.00f},
    { 6.00f, 0.00f},
    { 7.00f, 0.00f},
    { 8.00f, 0.00f},
    { 1.00f, 0.00f},
    { 2.00f, 0.00f},
    { 3.00f, 0.00f},
    { 4.00f, 0.00f},
    { 5.00f, 0.00f},
    { 6.00f, 0.00f},
    { 7.00f, 0.00f},
    { 8.00f, 0.00f},
    { 1.00f, 0.00f},
    { 2.00f, 0.00f},
    { 3.00f, 0.00f},
    { 4.00f, 0.00f},
    { 5.00f, 0.00f},
    { 6.00f, 0.00f},
    { 7.00f, 0.00f},
    { 8.00f, 0.00f},
    { 1.00f, 0.00f},
    { 2.00f, 0.00f},
    { 3.00f, 0.00f},
    { 4.00f, 0.00f},
    { 5.00f, 0.00f},
    { 6.00f, 0.00f},
    { 7.00f, 0.00f},
    { 8.00f, 0.00f},
    { 1.00f, 0.00f},
    { 2.00f, 0.00f},
    { 3.00f, 0.00f},
    { 4.00f, 0.00f},
    { 5.00f, 0.00f},
    { 6.00f, 0.00f},
    { 7.00f, 0.00f},
    { 8.00f, 0.00f},

    { 2.00f, 1.00f},
    { 4.00f, 2.00f},
    { 6.00f, 3.00f},
    { 8.00f, 4.00f},
    {10.00f, 5.00f},
    {12.00f, 6.00f},
    {14.00f, 7.00f},
    {16.00f, 8.00f},
    { 2.00f, 1.00f},
    { 4.00f, 2.00f},
    { 6.00f, 3.00f},
    { 8.00f, 4.00f},
    {10.00f, 5.00f},
    {12.00f, 6.00f},
    {14.00f, 7.00f},
    {16.00f, 8.00f},
    { 2.00f, 1.00f},
    { 4.00f, 2.00f},
    { 6.00f, 3.00f},
    { 8.00f, 4.00f},
    {10.00f, 5.00f},
    {12.00f, 6.00f},
    {14.00f, 7.00f},
    {16.00f, 8.00f},
    { 2.00f, 1.00f},
    { 4.00f, 2.00f},
    { 6.00f, 3.00f},
    { 8.00f, 4.00f},
    {10.00f, 5.00f},
    {12.00f, 6.00f},
    {14.00f, 7.00f},
    {16.00f, 8.00f},
    { 2.00f, 1.00f},
    { 4.00f, 2.00f},
    { 6.00f, 3.00f},
    { 8.00f, 4.00f},
    {10.00f, 5.00f},
    {12.00f, 6.00f},
    {14.00f, 7.00f},
    {16.00f, 8.00f},
  };

  std::vector<std::complex<float>> expected_axis_0_two_sided = {
    { 5.000f,  0.000f},
    {10.000f,  0.000f},
    {15.000f,  0.000f},
    {20.000f,  0.000f},
    {25.000f,  0.000f},
    {30.000f,  0.000f},
    {35.000f,  0.000f},
    {40.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},

    {10.000f,  5.000f},
    {20.000f, 10.000f},
    {30.000f, 15.000f},
    {40.000f, 20.000f},
    {50.000f, 25.000f},
    {60.000f, 30.000f},
    {70.000f, 35.000f},
    {80.000f, 40.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f},
    {-0.000f,  0.000f}
  };

  DiscreteFourierTransform(kind, input, {2, 5, 8, 2}, expected_axis_0_two_sided, 1, 5, false /*onesided*/);

  std::vector<std::complex<float>> expected_axis_0_two_sided_small_dft_length = {
    { 4.000f,  0.000f},
    { 8.000f,  0.000f},
    {12.000f,  0.000f},
    {16.000f,  0.000f},
    {20.000f,  0.000f},
    {24.000f,  0.000f},
    {28.000f,  0.000f},
    {32.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},

    { 8.000f,  4.000f},
    {16.000f,  8.000f},
    {24.000f, 12.000f},
    {32.000f, 16.000f},
    {40.000f, 20.000f},
    {48.000f, 24.000f},
    {56.000f, 28.000f},
    {64.000f, 32.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
  };
  DiscreteFourierTransform(
    kind, input, {2, 5, 8, 2}, expected_axis_0_two_sided_small_dft_length, 1, 4, false /*onesided*/
  );

  std::vector<std::complex<float>> expected_axis_0_two_sided_bigger_dft_length = {
    {  5.000000f,   0.000000f},
    { 10.000000f,   0.000000f},
    { 15.000000f,   0.000000f},
    { 20.000000f,   0.000000f},
    { 25.000000f,   0.000000f},
    { 30.000000f,   0.000000f},
    { 35.000000f,   0.000000f},
    { 40.000000f,   0.000000f},
    { -0.500000f,  -0.866025f},
    { -1.000000f,  -1.732051f},
    { -1.500000f,  -2.598076f},
    { -2.000000f,  -3.464101f},
    { -2.500000f,  -4.330126f},
    { -3.000000f,  -5.196152f},
    { -3.500000f,  -6.062176f},
    { -4.000000f,  -6.928203f},
    {  0.500000f,  -0.866025f},
    {  1.000000f,  -1.732051f},
    {  1.500000f,  -2.598076f},
    {  1.999999f,  -3.464102f},
    {  2.499999f,  -4.330127f},
    {  2.999999f,  -5.196152f},
    {  3.499999f,  -6.062178f},
    {  3.999999f,  -6.928203f},
    {  1.000000f,  -0.000000f},
    {  2.000000f,  -0.000001f},
    {  3.000000f,  -0.000001f},
    {  4.000000f,  -0.000002f},
    {  5.000000f,  -0.000002f},
    {  6.000000f,  -0.000002f},
    {  7.000000f,  -0.000003f},
    {  8.000000f,  -0.000003f},
    {  0.500000f,   0.866025f},
    {  1.000001f,   1.732051f},
    {  1.500001f,   2.598076f},
    {  2.000001f,   3.464102f},
    {  2.500002f,   4.330127f},
    {  3.000002f,   5.196153f},
    {  3.500002f,   6.062179f},
    {  4.000003f,   6.928204f},
    { -0.500000f,   0.866026f},
    { -1.000000f,   1.732052f},
    { -1.500000f,   2.598077f},
    { -2.000000f,   3.464104f},
    { -2.500000f,   4.330130f},
    { -2.999999f,   5.196155f},
    { -3.500000f,   6.062181f},
    { -4.000000f,   6.928207f},

    { 10.000000f,   5.000000f},
    { 20.000000f,  10.000000f},
    { 30.000000f,  15.000000f},
    { 40.000000f,  20.000000f},
    { 50.000000f,  25.000000f},
    { 60.000000f,  30.000000f},
    { 70.000000f,  35.000000f},
    { 80.000000f,  40.000000f},
    { -0.133975f,  -2.232050f},
    { -0.267949f,  -4.464101f},
    { -0.401925f,  -6.696153f},
    { -0.535898f,  -8.928202f},
    { -0.669872f, -11.160252f},
    { -0.803849f, -13.392305f},
    { -0.937822f, -15.624352f},
    { -1.071796f, -17.856403f},
    {  1.866025f,  -1.232051f},
    {  3.732050f,  -2.464102f},
    {  5.598075f,  -3.696153f},
    {  7.464101f,  -4.928204f},
    {  9.330126f,  -6.160254f},
    { 11.196151f,  -7.392306f},
    { 13.062176f,  -8.624355f},
    { 14.928202f,  -9.856407f},
    {  2.000000f,   0.999999f},
    {  4.000001f,   1.999998f},
    {  6.000001f,   2.999998f},
    {  8.000002f,   3.999997f},
    { 10.000003f,   4.999996f},
    { 12.000002f,   5.999995f},
    { 14.000003f,   6.999995f},
    { 16.000004f,   7.999993f},
    {  0.133975f,   2.232051f},
    {  0.267951f,   4.464102f},
    {  0.401926f,   6.696153f},
    {  0.535901f,   8.928205f},
    {  0.669876f,  11.160257f},
    {  0.803851f,  13.392306f},
    {  0.937826f,  15.624360f},
    {  1.071802f,  17.856409f},
    { -1.866026f,   1.232052f},
    { -3.732052f,   2.464104f},
    { -5.598077f,   3.696155f},
    { -7.464104f,   4.928207f},
    { -9.330130f,   6.160261f},
    {-11.196154f,   7.392309f},
    {-13.062180f,   8.624363f},
    {-14.928207f,   9.856415f},
  };

  DiscreteFourierTransform(
    kind, input, {2, 5, 8, 2}, expected_axis_0_two_sided_bigger_dft_length, 1, 6, false /*onesided*/
  );

  std::vector<std::complex<float>> expected_axis_0_one_sided = {
    { 5.000f,  0.000f},
    {10.000f,  0.000f},
    {15.000f,  0.000f},
    {20.000f,  0.000f},
    {25.000f,  0.000f},
    {30.000f,  0.000f},
    {35.000f,  0.000f},
    {40.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},

    {10.000f,  5.000f},
    {20.000f, 10.000f},
    {30.000f, 15.000f},
    {40.000f, 20.000f},
    {50.000f, 25.000f},
    {60.000f, 30.000f},
    {70.000f, 35.000f},
    {80.000f, 40.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
    {-0.000f,  0.000f},
    { 0.000f,  0.000f},
  };
  DiscreteFourierTransform(kind, input, {2, 5, 8, 2}, expected_axis_0_one_sided, 1, 5, true /*onesided*/);

  std::vector<std::complex<float>> expected_axis_1_two_sided = {
    { 36.000f,   0.000f},
    { -4.000f,   9.657f},
    { -4.000f,   4.000f},
    { -4.000f,   1.657f},
    { -4.000f,   0.000f},
    { -4.000f,  -1.657f},
    { -4.000f,  -4.000f},
    { -4.000f,  -9.657f},
    { 36.000f,   0.000f},
    { -4.000f,   9.657f},
    { -4.000f,   4.000f},
    { -4.000f,   1.657f},
    { -4.000f,   0.000f},
    { -4.000f,  -1.657f},
    { -4.000f,  -4.000f},
    { -4.000f,  -9.657f},
    { 36.000f,   0.000f},
    { -4.000f,   9.657f},
    { -4.000f,   4.000f},
    { -4.000f,   1.657f},
    { -4.000f,   0.000f},
    { -4.000f,  -1.657f},
    { -4.000f,  -4.000f},
    { -4.000f,  -9.657f},
    { 36.000f,   0.000f},
    { -4.000f,   9.657f},
    { -4.000f,   4.000f},
    { -4.000f,   1.657f},
    { -4.000f,   0.000f},
    { -4.000f,  -1.657f},
    { -4.000f,  -4.000f},
    { -4.000f,  -9.657f},
    { 36.000f,   0.000f},
    { -4.000f,   9.657f},
    { -4.000f,   4.000f},
    { -4.000f,   1.657f},
    { -4.000f,   0.000f},
    { -4.000f,  -1.657f},
    { -4.000f,  -4.000f},
    { -4.000f,  -9.657f},

    { 72.000f,  36.000f},
    {-17.657f,  15.314f},
    {-12.000f,   4.000f},
    { -9.657f,  -0.686f},
    { -8.000f,  -4.000f},
    { -6.343f,  -7.314f},
    { -4.000f, -12.000f},
    {  1.657f, -23.314f},
    { 72.000f,  36.000f},
    {-17.657f,  15.314f},
    {-12.000f,   4.000f},
    { -9.657f,  -0.686f},
    { -8.000f,  -4.000f},
    { -6.343f,  -7.314f},
    { -4.000f, -12.000f},
    {  1.657f, -23.314f},
    { 72.000f,  36.000f},
    {-17.657f,  15.314f},
    {-12.000f,   4.000f},
    { -9.657f,  -0.686f},
    { -8.000f,  -4.000f},
    { -6.343f,  -7.314f},
    { -4.000f, -12.000f},
    {  1.657f, -23.314f},
    { 72.000f,  36.000f},
    {-17.657f,  15.314f},
    {-12.000f,   4.000f},
    { -9.657f,  -0.686f},
    { -8.000f,  -4.000f},
    { -6.343f,  -7.314f},
    { -4.000f, -12.000f},
    {  1.657f, -23.314f},
    { 72.000f,  36.000f},
    {-17.657f,  15.314f},
    {-12.000f,   4.000f},
    { -9.657f,  -0.686f},
    { -8.000f,  -4.000f},
    { -6.343f,  -7.314f},
    { -4.000f, -12.000f},
    {  1.657f, -23.314f},
  };
  DiscreteFourierTransform(kind, input, {2, 5, 8, 2}, expected_axis_1_two_sided, 2, 8, false /*onesided*/);

  std::vector<std::complex<float>> expected_axis_1_one_sided = {
    { 36.000f,  0.000f},
    { -4.000f,  9.657f},
    { -4.000f,  4.000f},
    { -4.000f,  1.657f},
    { -4.000f,  0.000f},
    { 36.000f,  0.000f},
    { -4.000f,  9.657f},
    { -4.000f,  4.000f},
    { -4.000f,  1.657f},
    { -4.000f,  0.000f},
    { 36.000f,  0.000f},
    { -4.000f,  9.657f},
    { -4.000f,  4.000f},
    { -4.000f,  1.657f},
    { -4.000f,  0.000f},
    { 36.000f,  0.000f},
    { -4.000f,  9.657f},
    { -4.000f,  4.000f},
    { -4.000f,  1.657f},
    { -4.000f,  0.000f},
    { 36.000f,  0.000f},
    { -4.000f,  9.657f},
    { -4.000f,  4.000f},
    { -4.000f,  1.657f},
    { -4.000f,  0.000f},
    { 72.000f, 36.000f},
    {-17.657f, 15.314f},
    {-12.000f,  4.000f},
    { -9.657f, -0.686f},
    { -8.000f, -4.000f},
    { 72.000f, 36.000f},
    {-17.657f, 15.314f},
    {-12.000f,  4.000f},
    { -9.657f, -0.686f},
    { -8.000f, -4.000f},
    { 72.000f, 36.000f},
    {-17.657f, 15.314f},
    {-12.000f,  4.000f},
    { -9.657f, -0.686f},
    { -8.000f, -4.000f},
    { 72.000f, 36.000f},
    {-17.657f, 15.314f},
    {-12.000f,  4.000f},
    { -9.657f, -0.686f},
    { -8.000f, -4.000f},
    { 72.000f, 36.000f},
    {-17.657f, 15.314f},
    {-12.000f,  4.000f},
    { -9.657f, -0.686f},
    { -8.000f, -4.000f},
  };
  DiscreteFourierTransform(kind, input, {2, 5, 8, 2}, expected_axis_1_one_sided, 2, 8, true /*onesided*/);

  DiscreteFourierTransform_2D(kind);
}
#endif

static void ModelBuilding_GridSampleDeviceDirectX() {
#if !defined(BUILD_INBOX)
  ModelBuilding_GridSample_Internal(LearningModelDeviceKind::DirectX);
#endif
}

static void ModelBuilding_DiscreteFourierTransform() {
#if !defined(BUILD_INBOX)
  ModelBuilding_DiscreteFourierTransform_Internal(LearningModelDeviceKind::Cpu);
#endif
}

static void ModelBuilding_DiscreteFourierTransformDeviceDirectX() {
#if !defined(BUILD_INBOX)
  ModelBuilding_DiscreteFourierTransform_Internal(LearningModelDeviceKind::DirectX);
#endif
}

#if !defined(BUILD_INBOX)
static void DiscreteFourierTransformInverse(size_t axis, LearningModelDeviceKind kind) {
  std::vector<int64_t> shape = {2, 5, 8, 1};
  std::vector<int64_t> output_shape = {2, 5, 8, 2};

  auto model =
    LearningModelBuilder::Create(17)
      .Inputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.TimeSignal", TensorKind::Float, shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Spectra", TensorKind::Float, output_shape))
      .Outputs()
      .Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Inverse", TensorKind::Float, output_shape))
      .Operators()
      .Add(Operator(L"DFT")
             .SetInput(L"input", L"Input.TimeSignal")
             .SetAttribute(L"axis", TensorInt64Bit::CreateFromArray({}, {INT64(axis)}))
             .SetOutput(L"output", L"Output.Spectra"))
      .Operators()
      .Add(Operator(L"DFT")
             .SetInput(L"input", L"Output.Spectra")
             .SetAttribute(L"axis", TensorInt64Bit::CreateFromArray({}, {INT64(axis)}))
             .SetAttribute(L"inverse", TensorInt64Bit::CreateFromArray({}, {INT64(1)}))
             .SetOutput(L"output", L"Output.Inverse"))
      .CreateModel();

  auto device = LearningModelDevice(kind);
  LearningModelSession session(model, device);
  LearningModelBinding binding(session);

  auto input_vector = std::vector<float>{
    1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,
    5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,

    2,  4,  6,  8,  10, 12, 14, 16, 2,  4,  6,  8,  10, 12, 14, 16, 2,  4,  6,  8,
    10, 12, 14, 16, 2,  4,  6,  8,  10, 12, 14, 16, 2,  4,  6,  8,  10, 12, 14, 16,
  };
  // Populate binding
  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromArray(shape, input_vector));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // Check results
  auto y_tensor = result.Outputs().Lookup(L"Output.Inverse").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  for (uint32_t i = 0; i < y_ivv.Size(); i += 2) {
    constexpr float error_threshold = .001f;
    WINML_EXPECT_TRUE(abs(y_ivv.GetAt(i) - input_vector[i / 2]) < error_threshold);
    WINML_EXPECT_TRUE(abs(y_ivv.GetAt(i + 1) - 0) < error_threshold);
  }
}
#endif

static void ModelBuilding_DiscreteFourierTransformInverseIdentity() {
#if !defined(BUILD_INBOX)
  DiscreteFourierTransformInverse(1, LearningModelDeviceKind::Cpu);
  DiscreteFourierTransformInverse(2, LearningModelDeviceKind::Cpu);
#endif
}

static void ModelBuilding_DiscreteFourierTransformInverseIdentityDeviceDirectX() {
#if !defined(BUILD_INBOX)
  // Only powers of 2 dft supported on GPU currently!
  // DiscreteFourierTransformInverse(1, LearningModelDeviceKind::DirectX);
  DiscreteFourierTransformInverse(2, LearningModelDeviceKind::DirectX);
#endif
}

static void ModelBuilding_HannWindow() {
#if !defined(BUILD_INBOX)
  auto expected =
    std::vector<float>{0.000000f, 0.009607f, 0.038060f, 0.084265f, 0.146447f, 0.222215f, 0.308658f, 0.402455f,
                       0.500000f, 0.597545f, 0.691342f, 0.777785f, 0.853553f, 0.915735f, 0.961940f, 0.990393f,
                       1.000000f, 0.990393f, 0.961940f, 0.915735f, 0.853553f, 0.777785f, 0.691342f, 0.597545f,
                       0.500000f, 0.402455f, 0.308658f, 0.222215f, 0.146447f, 0.084265f, 0.038060f, 0.009607f};
  WindowFunction(L"HannWindow", TensorKind::Float, expected);
  WindowFunction(L"HannWindow", TensorKind::Double, expected);
#endif
}

static void ModelBuilding_HammingWindow() {
#if !defined(BUILD_INBOX)
  auto expected =
    std::vector<float>{0.086957f, 0.095728f, 0.121707f, 0.163894f, 0.220669f, 0.289848f, 0.368775f, 0.454415f,
                       0.543478f, 0.632541f, 0.718182f, 0.797108f, 0.866288f, 0.923062f, 0.965249f, 0.991228f,
                       1.000000f, 0.991228f, 0.965249f, 0.923062f, 0.866288f, 0.797108f, 0.718182f, 0.632541f,
                       0.543478f, 0.454415f, 0.368775f, 0.289848f, 0.220669f, 0.163894f, 0.121707f, 0.095728f};
  WindowFunction(L"HammingWindow", TensorKind::Float, expected);
  WindowFunction(L"HammingWindow", TensorKind::Double, expected);
#endif
}

static void ModelBuilding_BlackmanWindow() {
#if !defined(BUILD_INBOX)
  auto expected =
    std::vector<float>{0.000000f, 0.003518f, 0.014629f, 0.034880f, 0.066447f, 0.111600f, 0.172090f, 0.248544f,
                       0.340000f, 0.443635f, 0.554773f, 0.667170f, 0.773553f, 0.866349f, 0.938508f, 0.984303f,
                       1.000000f, 0.984303f, 0.938508f, 0.866349f, 0.773553f, 0.667170f, 0.554773f, 0.443635f,
                       0.340000f, 0.248544f, 0.172090f, 0.111600f, 0.066447f, 0.034880f, 0.014629f, 0.003518f};
  WindowFunction(L"BlackmanWindow", TensorKind::Float, expected);
  WindowFunction(L"BlackmanWindow", TensorKind::Double, expected);
#endif
}

static void ModelBuilding_STFT() {
#if !defined(BUILD_INBOX)
  size_t batch_size = 1;
  size_t sample_rate = 8192;
  float signal_duration_in_seconds = 5.f;
  size_t signal_size = static_cast<size_t>(sample_rate * signal_duration_in_seconds);
  size_t dft_size = 256;
  size_t hop_size = 128;

  // stft
  STFT(batch_size, signal_size, dft_size, hop_size, sample_rate, true);
  STFT(batch_size, signal_size, dft_size, hop_size, sample_rate, false);
#endif
}

static void ModelBuilding_MelSpectrogramOnThreeToneSignal() {
#if !defined(BUILD_INBOX)
  size_t batch_size = 1;
  size_t sample_rate = 8192;
  float signal_duration_in_seconds = 5.f;
  size_t signal_size = static_cast<size_t>(sample_rate * signal_duration_in_seconds);
  size_t dft_size = 256;
  size_t hop_size = 128;
  size_t window_size = 256;
  size_t n_mel_bins = 1024;

  MelSpectrogramOnThreeToneSignal(batch_size, signal_size, dft_size, window_size, hop_size, n_mel_bins, sample_rate);
#endif
}

static void SetIntraOpNumThreads() {
  auto shape = std::vector<int64_t>{1, 1000};
  auto model = ProtobufHelpers::CreateModel(TensorKind::Float, shape, 1000);
  auto device = LearningModelDevice(LearningModelDeviceKind::Cpu);
  auto options = LearningModelSessionOptions();
  auto nativeOptions = options.as<ILearningModelSessionOptionsNative>();

  // Set the number of intra op threads to half of logical cores.
  uint32_t desiredThreads = std::thread::hardware_concurrency() / 2;
  WINML_EXPECT_NO_THROW(nativeOptions->SetIntraOpNumThreadsOverride(desiredThreads));
  // Create session and grab the number of intra op threads to see if is set properly
  LearningModelSession session = nullptr;
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device, options));
  auto nativeSession = session.as<ILearningModelSessionNative>();
  uint32_t numIntraOpThreads;
  WINML_EXPECT_NO_THROW(nativeSession->GetIntraOpNumThreads(&numIntraOpThreads));
  WINML_EXPECT_EQUAL(desiredThreads, numIntraOpThreads);

  // Check to see that bind and evaluate continue to work when setting the intra op thread count
  std::vector<float> input(1000);
  std::iota(std::begin(input), std::end(input), 0.0f);
  auto tensor_input = TensorFloat::CreateFromArray(shape, input);
  auto binding = LearningModelBinding(session);
  binding.Bind(L"input", tensor_input);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // Check to verify that the default number of threads in LearningModelSession is equal to the number of logical cores.
  session = LearningModelSession(model, device);
  nativeSession = session.as<ILearningModelSessionNative>();
  WINML_EXPECT_NO_THROW(nativeSession->GetIntraOpNumThreads(&numIntraOpThreads));
  WINML_EXPECT_EQUAL(std::thread::hardware_concurrency(), numIntraOpThreads);
}

static void SetIntraOpThreadSpinning() {
  auto device = LearningModelDevice(LearningModelDeviceKind::Cpu);
  auto shape = std::vector<int64_t>{1, 1000};
  auto model = ProtobufHelpers::CreateModel(TensorKind::Float, shape, 1000);

  std::vector<float> input(1000);
  std::iota(std::begin(input), std::end(input), 0.0f);
  auto tensor_input = TensorFloat::CreateFromArray(shape, input);

  auto spinDisabled = LearningModelSessionOptions();
  auto spinDisabledNative = spinDisabled.as<ILearningModelSessionOptionsNative1>();
  spinDisabledNative->SetIntraOpThreadSpinning(false);

  // ensure disabled thread spin is internally disabled and can evaluate without error
  LearningModelSession sessionSpinDisabled = nullptr;
  WINML_EXPECT_NO_THROW(sessionSpinDisabled = LearningModelSession(model, device, spinDisabled));
  auto nativeSessionSpinDisabled = sessionSpinDisabled.as<ILearningModelSessionNative1>();
  boolean allowSpinning = true;
  nativeSessionSpinDisabled->GetIntraOpThreadSpinning(&allowSpinning);
  WINML_EXPECT_FALSE(allowSpinning);

  auto binding = LearningModelBinding(sessionSpinDisabled);
  binding.Bind(L"input", tensor_input);
  WINML_EXPECT_NO_THROW(sessionSpinDisabled.Evaluate(binding, L""));

  // ensure enabled thread spin is internally enabled and can evaluate without error
  auto spinEnabled = LearningModelSessionOptions();
  auto spinEnabledNative = spinEnabled.as<ILearningModelSessionOptionsNative1>();
  spinEnabledNative->SetIntraOpThreadSpinning(true);

  LearningModelSession sessionSpinEnabled = nullptr;
  WINML_EXPECT_NO_THROW(sessionSpinEnabled = LearningModelSession(model, device, spinEnabled));
  auto nativeSessionSpinEnabled = sessionSpinEnabled.as<ILearningModelSessionNative1>();
  nativeSessionSpinEnabled->GetIntraOpThreadSpinning(&allowSpinning);
  WINML_EXPECT_TRUE(allowSpinning);

  binding = LearningModelBinding(sessionSpinEnabled);
  binding.Bind(L"input", tensor_input);
  WINML_EXPECT_NO_THROW(sessionSpinEnabled.Evaluate(binding, L""));

  // ensure options by default allow spinning
  auto spinDefault = LearningModelSessionOptions();
  LearningModelSession sessionSpinDefault = nullptr;
  WINML_EXPECT_NO_THROW(sessionSpinDefault = LearningModelSession(model, device, spinDefault));
  auto nativeSessionSpinDefault = sessionSpinDefault.as<ILearningModelSessionNative1>();
  allowSpinning = false;
  nativeSessionSpinDefault->GetIntraOpThreadSpinning(&allowSpinning);
  WINML_EXPECT_TRUE(allowSpinning);
}

static void SetName() {
#ifndef BUILD_INBOX
  // load the model with name 'squeezenet_old'
  LearningModel model = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", model));
  auto model_name = model.Name();
  auto squeezenet_old = to_hstring("squeezenet_old");
  WINML_EXPECT_EQUAL(model_name, squeezenet_old);

  // ensure the model name can be changed to 'new name'
  auto experimental_model = winml_experimental::LearningModelExperimental(model);
  auto new_name = to_hstring("new name");
  experimental_model.SetName(new_name);
  model_name = model.Name();
  WINML_EXPECT_EQUAL(model_name, new_name);

  // ensure the model protobuf was actually modified
  std::wstring path = FileHelpers::GetModulePath() + L"model_name_changed.onnx";
  experimental_model.Save(path);
  LearningModel model_name_changed = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model_name_changed.onnx", model_name_changed));
  model_name = model_name_changed.Name();
  WINML_EXPECT_EQUAL(model_name, new_name);
#endif
}

const LearningModelSessionAPITestsApi& getapi() {
  static LearningModelSessionAPITestsApi api = {
    LearningModelSessionAPITestsClassSetup,
    CreateSessionDeviceDefault,
    CreateSessionDeviceCpu,
    CreateSessionWithModelLoadedFromStream,
    CreateSessionDeviceDirectX,
    CreateSessionDeviceDirectXHighPerformance,
    CreateSessionDeviceDirectXMinimumPower,
    AdapterIdAndDevice,
    EvaluateFeatures,
    EvaluateFeaturesAsync,
    EvaluationProperties,
    CreateSessionWithCastToFloat16InModel,
    CreateSessionWithFloat16InitializersInModel,
    EvaluateSessionAndCloseModel,
    NamedDimensionOverride,
    CloseSession,
    SetIntraOpNumThreads,
    SetIntraOpThreadSpinning,
    ModelBuilding_Gemm,
    ModelBuilding_StandardDeviationNormalization,
    ModelBuilding_DynamicMatmul,
    ModelBuilding_ConstantMatmul,
    ModelBuilding_DiscreteFourierTransform,
    ModelBuilding_DiscreteFourierTransformInverseIdentity,
    ModelBuilding_DiscreteFourierTransformDeviceDirectX,
    ModelBuilding_DiscreteFourierTransformInverseIdentityDeviceDirectX,
    ModelBuilding_GridSampleDeviceDirectX,
    ModelBuilding_HannWindow,
    ModelBuilding_HammingWindow,
    ModelBuilding_BlackmanWindow,
    ModelBuilding_STFT,
    ModelBuilding_MelSpectrogramOnThreeToneSignal,
    ModelBuilding_MelWeightMatrix,
    SetName
  };

  if (SkipGpuTests()) {
    api.CreateSessionDeviceDirectX = SkipTest;
    api.CreateSessionDeviceDirectXHighPerformance = SkipTest;
    api.CreateSessionDeviceDirectXMinimumPower = SkipTest;
    api.CreateSessionWithCastToFloat16InModel = SkipTest;
    api.CreateSessionWithFloat16InitializersInModel = SkipTest;
    api.AdapterIdAndDevice = SkipTest;
    api.ModelBuilding_DiscreteFourierTransformDeviceDirectX = SkipTest;
    api.ModelBuilding_DiscreteFourierTransformInverseIdentityDeviceDirectX = SkipTest;
    api.ModelBuilding_GridSampleDeviceDirectX = SkipTest;
  }
  if (RuntimeParameterExists(L"EdgeCore")) {
    api.AdapterIdAndDevice = SkipTest;
  }
  if (RuntimeParameterExists(L"noIDXGIFactory6Tests")) {
    api.CreateSessionDeviceDirectXHighPerformance = SkipTest;
    api.CreateSessionDeviceDirectXMinimumPower = SkipTest;
    api.AdapterIdAndDevice = SkipTest;
  }
  if (SkipTestsImpactedByOpenMP()) {
    api.SetIntraOpNumThreads = SkipTest;
  }
  return api;
}
