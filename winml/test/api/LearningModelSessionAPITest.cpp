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

using wf::IPropertyValue;

static void LearningModelSessionAPITestsClassSetup() {
  init_apartment();
}

static void CreateSessionDeviceDefault()
{
    LearningModel learningModel = nullptr;
    LearningModelDevice learningModelDevice = nullptr;
    WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

    WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::Default));
    WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void CreateSessionDeviceCpu()
{
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

static void CreateSessionDeviceDirectX()
{
    LearningModel learningModel = nullptr;
    LearningModelDevice learningModelDevice = nullptr;
    WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

    WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectX));
    WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void CreateSessionDeviceDirectXHighPerformance()
{
    LearningModel learningModel = nullptr;
    LearningModelDevice learningModelDevice = nullptr;
    WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));

    WINML_EXPECT_NO_THROW(learningModelDevice = LearningModelDevice(LearningModelDeviceKind::DirectXHighPerformance));
    WINML_EXPECT_NO_THROW(LearningModelSession(learningModel, learningModelDevice));
}

static void CreateSessionDeviceDirectXMinimumPower()
{
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

static void EvaluateFeatures()
{
    std::vector<int64_t> shape = { 4 };
    std::vector<winrt::hstring> data = { L"one", L"two", L"three", L"four" };

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

static void EvaluateFeaturesAsync()
{
    std::vector<int64_t> shape = { 4 };
    std::vector<winrt::hstring> data = { L"one", L"two", L"three", L"four" };

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

static void EvaluationProperties()
{
    // load a model
    LearningModel learningModel = nullptr;
    WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));
    // create a session
    LearningModelSession learningModelSession = nullptr;
    learningModelSession = LearningModelSession(learningModel);
    // set a property
    auto value = wf::PropertyValue::CreateBoolean(true);
    learningModelSession.EvaluationProperties().Insert(L"propName1", value);
    // get the property and make sure it's there with the right value
    auto value2 = learningModelSession.EvaluationProperties().Lookup(L"propName1");
    WINML_EXPECT_EQUAL(value2.as<IPropertyValue>().GetBoolean(), true);
}

static LearningModelSession CreateSession(LearningModel model)
{
    LearningModelDevice device(nullptr);
    WINML_EXPECT_NO_THROW(device = LearningModelDevice(LearningModelDeviceKind::DirectX));

    LearningModelSession session(nullptr);
    if (CommonDeviceHelpers::IsFloat16Supported(device))
    {
        WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device));
    }
    else
    {
        WINML_EXPECT_THROW_SPECIFIC(
            session = LearningModelSession(model, device),
            winrt::hresult_error,
            [](const winrt::hresult_error& e) -> bool
        {
            return e.code() == DXGI_ERROR_UNSUPPORTED;
        });
    }

    return session;
}

static void CreateSessionWithCastToFloat16InModel()
{
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
    bool close_model_on_session_creation)
{
    auto shape = std::vector<int64_t>{ 1, 1000 };

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
    auto tensor_input = TensorFloat::CreateFromShapeArrayAndDataArray(shape, input);
    auto binding = LearningModelBinding(session);
    binding.Bind(L"input", tensor_input);

    LearningModelEvaluationResult result(nullptr);
    WINML_EXPECT_NO_THROW(result = session.Evaluate(binding, L""));

    if (close_model_on_session_creation)
    {
        // ensure that the model has been closed
        WINML_EXPECT_THROW_SPECIFIC(
            LearningModelSession(model, device, options),
            winrt::hresult_error,
            [](const winrt::hresult_error& e) -> bool
        {
            return e.code() == E_INVALIDARG;
        });
    }
    else
    {
        WINML_EXPECT_NO_THROW(LearningModelSession(model, device, options));
    }
}

static void EvaluateSessionAndCloseModel()
{
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
  int64_t n = 5, c = 3, h = 720, w = 720;

  LearningModelSessionOptions options;
  options.OverrideNamedDimension(L"None", n);
  
  // Verifies that if a Dim name doesn't exist the named dimension override does nothing
  options.OverrideNamedDimension(L"DimNameThatDoesntExist", n);

  LearningModelSession session(nullptr);
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device, options));

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
    std::vector<int64_t> shape = { 1, 3, 224, 224 };
    auto tensor_input = TensorFloat::CreateFromShapeArrayAndDataArray(shape, input);
    WINML_EXPECT_THROW_SPECIFIC(LearningModelBinding binding(session),
        winrt::hresult_error,
        [](const winrt::hresult_error &e) -> bool
    {
        return e.code() == RO_E_CLOSED;
    });
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
    auto tensor_input = TensorFloat::CreateFromShapeArrayAndDataArray(shape, input);
    auto binding = LearningModelBinding(session);
    binding.Bind(L"input", tensor_input);
    WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

    // Check to verify that the default number of threads in LearningModelSession is equal to the number of logical cores.
    session = LearningModelSession(model, device);
    nativeSession = session.as<ILearningModelSessionNative>();
    WINML_EXPECT_NO_THROW(nativeSession->GetIntraOpNumThreads(&numIntraOpThreads));
    WINML_EXPECT_EQUAL(std::thread::hardware_concurrency(), numIntraOpThreads);
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
    SetIntraOpNumThreads
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
