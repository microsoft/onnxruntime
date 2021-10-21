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

static void CreateSessionDeviceDefault()
{
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

static void CreateSessionWithModelLoadedFromStream()
{
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
  WINML_EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, __uuidof(IDXGIAdapter), adapter.put_void()));
  WINML_EXPECT_HRESULT_SUCCEEDED(adapter->GetDesc(&desc));
  id.QuadPart = APITest::GetAdapterIdQuadPart(learningModelDevice);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.LowPart, id.LowPart);
  WINML_EXPECT_EQUAL(desc.AdapterLuid.HighPart, id.HighPart);
  WINML_EXPECT_TRUE(learningModelDevice.Direct3D11Device() != nullptr);

  adapter = nullptr;
  learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectXMinPower);
  WINML_EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_MINIMUM_POWER, __uuidof(IDXGIAdapter), adapter.put_void()));
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
      shape, winrt::single_threaded_vector<winrt::hstring>(std::move(dataCopy)).GetView());
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
      shape, winrt::single_threaded_vector<winrt::hstring>(std::move(dataCopy)).GetView());
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
        [](const winrt::hresult_error& e) -> bool {
          return e.code() == DXGI_ERROR_UNSUPPORTED;
        });
  }

  return session;
}

static void CreateSessionWithCastToFloat16InModel() {
  // load a model
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"fp16-truncate-with-cast.onnx", learningModel));

  CreateSession(learningModel);
}

static void CreateSessionWithFloat16InitializersInModel()
{
    // load a model
    LearningModel learningModel = nullptr;
    WINML_EXPECT_NO_THROW(APITest::LoadModel(L"fp16-initializer.onnx", learningModel));

  CreateSession(learningModel);
}

static void EvaluateSessionAndCloseModelHelper(
    LearningModelDeviceKind kind,
    bool close_model_on_session_creation) {
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
        [](const winrt::hresult_error& e) -> bool {
          return e.code() == E_INVALIDARG;
        });
  } else {
    WINML_EXPECT_NO_THROW(LearningModelSession(model, device, options));
  }
}

static void EvaluateSessionAndCloseModel() {
  WINML_EXPECT_NO_THROW(::EvaluateSessionAndCloseModelHelper(LearningModelDeviceKind::Cpu, true));
  WINML_EXPECT_NO_THROW(::EvaluateSessionAndCloseModelHelper(LearningModelDeviceKind::Cpu, false));
}

static void NamedDimensionOverride() 
{
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
  std::vector<int64_t> shape{n,c,h,w};
  int64_t size = n*c*h*w;
  std::vector<float> buffer;
  buffer.resize(static_cast<size_t>(size));
  auto featureValue = TensorFloat::CreateFromIterable(shape, winrt::single_threaded_vector<float>(std::move(buffer)));
  LearningModelBinding binding(session);
  binding.Bind(descriptor.Name(), featureValue);

  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

static void CloseSession()
{
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
  WINML_EXPECT_THROW_SPECIFIC(LearningModelBinding binding(session),
                              winrt::hresult_error,
                              [](const winrt::hresult_error& e) -> bool {
                                return e.code() == RO_E_CLOSED;
                              });
}

#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
static void WindowFunction(const wchar_t* window_operator_name, TensorKind kind) {
  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> output_shape = {32};
  auto double_data_type = TensorInt64Bit::CreateFromArray({}, {11});

  auto window_operator =
    Operator(window_operator_name, MS_EXPERIMENTAL_DOMAIN)
      .SetInput(L"size", L"Input")
      .SetOutput(L"output", L"Output");

  if (kind == TensorKind::Double) {
    window_operator.SetAttribute(L"output_datatype", double_data_type);
  }
    
  auto model = 
      LearningModelBuilder::Create(13)
              .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input", TensorKind::Int64, scalar_shape))
              .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", kind, output_shape))
              .Operators().Add(window_operator)
              .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  binding.Bind(L"Input", TensorInt64Bit::CreateFromArray(scalar_shape, {32}));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // Check results
  printf("Output\n");
  if (kind == TensorKind::Float) {
    auto y_tensor = result.Outputs().Lookup(L"Output").as<TensorFloat>();
    auto y_ivv = y_tensor.GetAsVectorView();
    for (int i = 0; i < output_shape[0]; i++) {
      printf("%f, ", y_ivv.GetAt(i));
    }
  }
  if (kind == TensorKind::Double) {
    auto y_tensor = result.Outputs().Lookup(L"Output").as<TensorDouble>();
    auto y_ivv = y_tensor.GetAsVectorView();
    for (int i = 0; i < output_shape[0]; i++) {
      printf("%f, ", y_ivv.GetAt(i));
    }
  }
  printf("\n");
}
#endif

#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
static void DiscreteFourierTransform(bool is_onesided = false) {
  std::vector<int64_t> shape = {1, 5};
  std::vector<int64_t> output_shape = {1, 5, 2};
  output_shape[1] = is_onesided ? (1 + (shape[1] >> 1)) : shape[1];
   
  auto model =
      LearningModelBuilder::Create(13)
        .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.Signal", TensorKind::Float, shape))
        .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Spectra", TensorKind::Float, output_shape))
        .Operators().Add(Operator(L"DFT", MS_EXPERIMENTAL_DOMAIN)
          .SetInput(L"input", L"Input.Signal")
          .SetAttribute(L"onesided", TensorInt64Bit::CreateFromArray({}, {is_onesided}))
          .SetOutput(L"output", L"Output.Spectra"))
        .CreateModel();
  
  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Populate binding
  binding.Bind(L"Input.Signal", TensorFloat::CreateFromArray(shape, {1, 2, 3, 4, 5}));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // Check results
  printf("Output.Spectra\n");
  auto y_tensor = result.Outputs().Lookup(L"Output.Spectra").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  for (int i = 0; i < output_shape[0] * output_shape[1] * 2; i += 2) {
    printf("(%f + %fi), ", y_ivv.GetAt(i), y_ivv.GetAt(i + 1));
  }
  printf("\n");
}
#endif

template <typename T>
static auto MakePureFrequency(float frequency_in_hertz, size_t signal_size, size_t sample_rate) {
  float amplitude = 4;
  float angular_velocity = frequency_in_hertz * 2 * 3.1415f;
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
    middle_c[i] = (i < signal_size / 3) ?
                    middle_c[i] :
                    (i < 2*signal_size/3) ?
                        (middle_c[i] + c2[i]) :
                        (middle_c[i] + c2[i] + c4[i]);
  }
  return middle_c;
}

#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
static void STFT(size_t batch_size, size_t signal_size, size_t dft_size,
    size_t hop_size, size_t sample_rate, bool is_onesided = false) {
  auto n_dfts = static_cast<size_t>(1 + floor((signal_size - dft_size) / hop_size));
  auto input_shape = std::vector<int64_t>{1, INT64(signal_size)};
  auto output_shape =
    std::vector<int64_t>{
      INT64(batch_size),
      INT64(n_dfts),
      is_onesided ? ((INT64(dft_size) >> 1) + 1) : INT64(dft_size),
      2
    };
  auto dft_length = TensorInt64Bit::CreateFromArray({}, {INT64(dft_size)});
  
  auto model =
      LearningModelBuilder::Create(13)
          .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.TimeSignal", TensorKind::Float, input_shape))
          .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.STFT", TensorKind::Float, output_shape))
          .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.HannWindow", TensorKind::Float, {INT64(dft_size)}))
          .Operators().Add(Operator(L"HannWindow", MS_EXPERIMENTAL_DOMAIN)
              .SetConstant(L"size", dft_length)
              .SetOutput(L"output", L"Output.HannWindow"))
          .Operators().Add(Operator(L"STFT", MS_EXPERIMENTAL_DOMAIN)
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
  printf("\n");
  printf("Input.TimeSignal:\n");
  for (size_t i = 0; i < dft_size; i++) {
    printf("%f, ", signal[i]);
  }

  // Bind
  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromArray(input_shape, signal));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

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
}
#endif

static void ModelBuilding_MelWeightMatrix() {
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
  std::vector<int64_t> output_shape = {INT64(9), INT64(8)};
  auto builder =
    LearningModelBuilder::Create(13)
      .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.MelWeightMatrix", TensorKind::Float, output_shape))
      .Operators().Add(Operator(L"MelWeightMatrix", MS_EXPERIMENTAL_DOMAIN)
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

  printf("\n");
  printf("Output.MelWeightMatrix\n");
  {
    auto y_tensor = result.Outputs().Lookup(L"Output.MelWeightMatrix").as<TensorFloat>();
    auto y_ivv = y_tensor.GetAsVectorView();
    for (unsigned i = 0; i < y_ivv.Size(); i++) {
      printf("%f, ", y_ivv.GetAt(i));
    }
  }

  printf("\n");
#endif
}

#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
static void MelSpectrogramOnThreeToneSignal(
    size_t batch_size, size_t signal_size, size_t window_size, size_t dft_size,
    size_t hop_size, size_t n_mel_bins, size_t sampling_rate) {
  auto n_dfts = static_cast<size_t>(1 + floor((signal_size - dft_size) / hop_size));
  auto onesided_dft_size = (dft_size >> 1) + 1;
  std::vector<int64_t> signal_shape = {INT64(batch_size), INT64(signal_size)};
  std::vector<int64_t> mel_spectrogram_shape = {INT64(batch_size), 1, INT64(n_dfts), INT64(n_mel_bins)};

  auto builder =
    LearningModelBuilder::Create(13)
      .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.TimeSignal", TensorKind::Float, signal_shape))
      .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.MelSpectrogram", TensorKind::Float, mel_spectrogram_shape))
      .Operators().Add(Operator(L"HannWindow", MS_EXPERIMENTAL_DOMAIN)
        .SetConstant(L"size", TensorInt64Bit::CreateFromArray({}, {INT64(window_size)}))
        .SetOutput(L"output", L"hann_window"))
      .Operators().Add(Operator(L"STFT", MS_EXPERIMENTAL_DOMAIN)
        .SetName(L"STFT_NAMED_NODE")
        .SetInput(L"signal", L"Input.TimeSignal")
        .SetInput(L"window", L"hann_window")
        .SetConstant(L"frame_length", TensorInt64Bit::CreateFromArray({}, {INT64(dft_size)}))
        .SetConstant(L"frame_step", TensorInt64Bit::CreateFromArray({}, {INT64(hop_size)}))
        .SetOutput(L"output", L"stft_output"))
      .Operators().Add(Operator(L"ReduceSumSquare")
        .SetInput(L"data", L"stft_output")
        .SetAttribute(L"axes", TensorInt64Bit::CreateFromArray({1}, {3}))
        .SetAttribute(L"keepdims", TensorInt64Bit::CreateFromArray({}, {0}))
        .SetOutput(L"reduced", L"magnitude_squared"))
      .Operators().Add(Operator(L"Div")
        .SetInput(L"A", L"magnitude_squared")
        .SetConstant(L"B", TensorFloat::CreateFromArray({}, {static_cast<float>(dft_size)}))
        .SetOutput(L"C", L"power_frames"))
      .Operators().Add(Operator(L"MelWeightMatrix", MS_EXPERIMENTAL_DOMAIN)
        .SetConstant(L"num_mel_bins", TensorInt64Bit::CreateFromArray({}, {INT64(n_mel_bins)}))
        .SetConstant(L"dft_length", TensorInt64Bit::CreateFromArray({}, {INT64(dft_size)}))
        .SetConstant(L"sample_rate", TensorInt64Bit::CreateFromArray({}, {INT64(sampling_rate)}))
        .SetConstant(L"lower_edge_hertz", TensorFloat::CreateFromArray({}, {0}))
        .SetConstant(L"upper_edge_hertz", TensorFloat::CreateFromArray({}, {sampling_rate / 2.f}))
        .SetOutput(L"output", L"mel_weight_matrix"))
      .Operators().Add(Operator(L"Reshape")
        .SetInput(L"data", L"power_frames")
        .SetConstant(L"shape", TensorInt64Bit::CreateFromArray({2}, {INT64(batch_size * n_dfts), INT64(onesided_dft_size)}))
        .SetOutput(L"reshaped", L"reshaped_output"))
      .Operators().Add(Operator(L"MatMul")
        .SetInput(L"A", L"reshaped_output")
        .SetInput(L"B", L"mel_weight_matrix")
        .SetOutput(L"Y", L"mel_spectrogram"))
      .Operators().Add(Operator(L"Reshape")
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
  auto output_image =
    winrt::Windows::Media::VideoFrame(
      winrt::Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8,
      INT32(n_mel_bins),
      INT32(n_dfts));
  binding.Bind(L"Output.MelSpectrogram", output_image);

  // Evaluate
  auto start = std::chrono::high_resolution_clock::now();
  auto result = session.Evaluate(binding, L"");
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> evaluate_duration_in_microseconds = end - start;
  printf("Evaluate Took: %f\n", evaluate_duration_in_microseconds.count());

  // Check the output video frame object by saving output image to disk
  std::wstring out_name = L"mel_spectrogram.jpg";

  // Save the output
  std::wstring modulePath = FileHelpers::GetModulePath();
  winrt::Windows::Storage::StorageFolder folder = winrt::Windows::Storage::StorageFolder::GetFolderFromPathAsync(modulePath).get();
  winrt::Windows::Storage::StorageFile file = folder.CreateFileAsync(out_name, winrt::Windows::Storage::CreationCollisionOption::ReplaceExisting).get();
  winrt::Windows::Storage::Streams::IRandomAccessStream write_stream = file.OpenAsync(winrt::Windows::Storage::FileAccessMode::ReadWrite).get();
  winrt::Windows::Graphics::Imaging::BitmapEncoder encoder = winrt::Windows::Graphics::Imaging::BitmapEncoder::CreateAsync(winrt::Windows::Graphics::Imaging::BitmapEncoder::JpegEncoderId(), write_stream).get();
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
  LearningModelBuilder::Create(13)
    .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input", L"The NHWC image", TensorKind::Float, input_shape))
    .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Means", TensorKind::Float, {channels}))
    .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"StdDevs", TensorKind::Float, {channels}))
    .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", L"The NCHW image normalized with mean and stddev.", TensorKind::Float, output_shape))
    .Operators().Add(Operator(L"Sub")
                       .SetInput(L"A", L"Input")
                       .SetInput(L"B", L"Means")
                       .SetOutput(L"C", L"SubOutput"))
    .Operators().Add(Operator(L"Div")
                       .SetInput(L"A", L"SubOutput")
                       .SetInput(L"B", L"StdDevs")
                       .SetOutput(L"C", L"DivOutput"))
    .Operators().Add(Operator(L"Transpose")
                       .SetInput(L"data", L"DivOutput")
                       .SetAttribute(L"perm", TensorInt64Bit::CreateFromArray({4}, {0,3,1,2}))
                       .SetOutput(L"transposed", L"Output"))
    .Save(L"StandardDeviationNormalization.onnx");
  //.CreateModel();
#endif
}

static void ModelBuilding_Gemm() {
#ifndef BUILD_INBOX
  std::vector<int64_t> shape = {3, 3};
  auto model =
    LearningModelBuilder::Create(13)
      .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputA", TensorKind::Float, shape))
      .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputB", TensorKind::Float, shape))
      .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputC", TensorKind::Float, shape))
      .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"OutputY", TensorKind::Float, shape))
      .Operators().Add(Operator(L"Gemm")
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
          .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputA", TensorKind::Float, a_shape))
          .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputB", TensorKind::Float, b_shape))
          .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", TensorKind::Float, {a_shape[0], b_shape[1]}))
          .Operators().Add(Operator(L"MatMul")
                        .SetInput(L"A", L"InputA")
                        .SetInput(L"B", L"InputB")
                        .SetOutput(L"Y", L"Output"))
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
  printf("Evaluate Took: %f\n", evaluate_duration_in_microseconds.count());
#endif
}

static void ModelBuilding_ConstantMatmul() {
#ifndef BUILD_INBOX
  std::vector<int64_t> a_shape = {318, 129};
  std::vector<int64_t> b_shape = {129, 1024};

  auto model =
    LearningModelBuilder::Create(13)
      .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"InputA", TensorKind::Float, a_shape))
      .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output", TensorKind::Float, {a_shape[0], b_shape[1]}))
      .Operators().Add(Operator(L"MatMul")
        .SetInput(L"A", L"InputA")
        .SetConstant(L"B", TensorFloat::CreateFromArray(b_shape, std::vector<float>(SIZET(b_shape[0] * b_shape[1]), 1)))
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
  printf("Evaluate Took: %f\n", evaluate_duration_in_microseconds.count());
#endif
}

static void ModelBuilding_DiscreteFourierTransform() {
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
  DiscreteFourierTransform(false /*onesided*/);
  DiscreteFourierTransform(true /*onesided*/);
#endif
}

static void ModelBuilding_DiscreteFourierTransformInverseIdentity() {
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
  std::vector<int64_t> shape = {1, 5};
  std::vector<int64_t> output_shape = {1, shape[1], 2};

  auto model =
      LearningModelBuilder::Create(13)
          .Inputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Input.TimeSignal", TensorKind::Float, shape))
          .Outputs().Add(LearningModelBuilder::CreateTensorFeatureDescriptor(L"Output.Spectra", TensorKind::Float, output_shape))
          .Operators().Add(Operator(L"DFT", MS_EXPERIMENTAL_DOMAIN)
                             .SetInput(L"input", L"Input.TimeSignal")
                             .SetOutput(L"output", L"DFTOutput"))
          .Operators().Add(Operator(L"IDFT", MS_EXPERIMENTAL_DOMAIN)
                             .SetInput(L"input", L"DFTOutput")
                             .SetOutput(L"output", L"Output.Spectra"))
          .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Populate binding
  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromArray(shape, {1, 2, 3, 4, 5}));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // Check results
  printf("Output.Spectra\n");
  auto y_tensor = result.Outputs().Lookup(L"Output.Spectra").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  for (int i = 0; i < output_shape[0] * output_shape[1] * 2; i += 2) {
    printf("(%f + %fi), ", y_ivv.GetAt(i), y_ivv.GetAt(i + 1));
  }
  printf("\n");
#endif
}

static void ModelBuilding_HannWindow() {
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
  WindowFunction(L"HannWindow", TensorKind::Float);
  WindowFunction(L"HannWindow", TensorKind::Double);
#endif
}

static void ModelBuilding_HammingWindow() {
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
  WindowFunction(L"HammingWindow", TensorKind::Float);
  WindowFunction(L"HammingWindow", TensorKind::Double);
#endif
}

static void ModelBuilding_BlackmanWindow() {
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
  WindowFunction(L"BlackmanWindow", TensorKind::Float);
  WindowFunction(L"BlackmanWindow", TensorKind::Double);
#endif
}

static void ModelBuilding_STFT() {
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
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
#if !defined(BUILD_INBOX) && defined(BUILD_MS_EXPERIMENTAL_OPS)
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


const LearningModelSessionAPITestsApi& getapi() {
  static LearningModelSessionAPITestsApi api =
  {
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
    ModelBuilding_HannWindow,
    ModelBuilding_HammingWindow,
    ModelBuilding_BlackmanWindow,
    ModelBuilding_STFT,
    ModelBuilding_MelSpectrogramOnThreeToneSignal,
    ModelBuilding_MelWeightMatrix,
  };

  if (SkipGpuTests()) {
    api.CreateSessionDeviceDirectX = SkipTest;
    api.CreateSessionDeviceDirectXHighPerformance = SkipTest;
    api.CreateSessionDeviceDirectXMinimumPower = SkipTest;
    api.CreateSessionWithCastToFloat16InModel = SkipTest;
    api.CreateSessionWithFloat16InitializersInModel = SkipTest;
    api.AdapterIdAndDevice = SkipTest;
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
