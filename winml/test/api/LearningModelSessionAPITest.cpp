#include "testPch.h"
#include "APITest.h"

#include "winrt/Windows.Storage.h"

#include "DeviceHelpers.h"
#include "protobufHelpers.h"

#include <D3d11_4.h>
#include <dxgi1_6.h>
#include "Psapi.h"

using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;

using winrt::Windows::Foundation::IPropertyValue;

class LearningModelSessionAPITests : public APITest
{};

class LearningModelSessionAPITestsGpu : public APITest
{
protected:
    void SetUp() override {
        GPUTEST
    }
};

class LearningModelSessionAPITestsSkipEdgeCore : public LearningModelSessionAPITestsGpu
{
protected:
    void SetUp() override {
        LearningModelSessionAPITestsGpu::SetUp();
        SKIP_EDGECORE
    }
};

TEST_F(LearningModelSessionAPITests, CreateSessionDeviceDefault)
{
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));

    EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::Default));
    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
}

TEST_F(LearningModelSessionAPITests, CreateSessionDeviceCpu)
{
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));

    EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::Cpu));
    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
    // for the CPU device, make sure that we get back NULL and 0 for any device properties
    EXPECT_FALSE(m_device.Direct3D11Device());
    LARGE_INTEGER id;
    id.QuadPart = GetAdapterIdQuadPart();
    EXPECT_EQ(id.LowPart, static_cast<DWORD>(0));
    EXPECT_EQ(id.HighPart, 0);
}

TEST_F(LearningModelSessionAPITests, CreateSessionWithModelLoadedFromStream)
{
    std::wstring path = FileHelpers::GetModulePath() + L"model.onnx";
    auto storageFile = winrt::Windows::Storage::StorageFile::GetFileFromPathAsync(path).get();

    EXPECT_NO_THROW(m_model = LearningModel::LoadFromStream(storageFile));

    EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::Default));
    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
}

TEST_F(LearningModelSessionAPITestsGpu, CreateSessionDeviceDirectX)
{
    GPUTEST
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));

    EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::DirectX));
    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
}

TEST_F(LearningModelSessionAPITestsGpu, CreateSessionDeviceDirectXHighPerformance)
{
    GPUTEST
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));

    EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::DirectXHighPerformance));
    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
}

TEST_F(LearningModelSessionAPITestsGpu, CreateSessionDeviceDirectXMinimumPower)
{
    GPUTEST
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));

    EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::DirectXMinPower));
    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
}

TEST_F(LearningModelSessionAPITestsSkipEdgeCore, AdapterIdAndDevice)
{
    GPUTEST
    SKIP_EDGECORE
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));

    com_ptr<IDXGIFactory6> factory;
    EXPECT_HRESULT_SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory6), factory.put_void()));
    com_ptr<IDXGIAdapter> adapter;

    m_device = LearningModelDevice(LearningModelDeviceKind::DirectX);
    EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapters(0, adapter.put()));
    DXGI_ADAPTER_DESC desc;
    EXPECT_HRESULT_SUCCEEDED(adapter->GetDesc(&desc));
    LARGE_INTEGER id;
    id.QuadPart = GetAdapterIdQuadPart();
    EXPECT_EQ(desc.AdapterLuid.LowPart, id.LowPart);
    EXPECT_EQ(desc.AdapterLuid.HighPart, id.HighPart);
    EXPECT_TRUE(m_device.Direct3D11Device() != nullptr);

    m_device = LearningModelDevice(LearningModelDeviceKind::DirectXHighPerformance);
    adapter = nullptr;
    EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, __uuidof(IDXGIAdapter), adapter.put_void()));
    EXPECT_HRESULT_SUCCEEDED(adapter->GetDesc(&desc));
    id.QuadPart = GetAdapterIdQuadPart();
    EXPECT_EQ(desc.AdapterLuid.LowPart, id.LowPart);
    EXPECT_EQ(desc.AdapterLuid.HighPart, id.HighPart);
    EXPECT_TRUE(m_device.Direct3D11Device() != nullptr);

    adapter = nullptr;
    m_device = LearningModelDevice(LearningModelDeviceKind::DirectXMinPower);
    EXPECT_HRESULT_SUCCEEDED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_MINIMUM_POWER, __uuidof(IDXGIAdapter), adapter.put_void()));
    EXPECT_HRESULT_SUCCEEDED(adapter->GetDesc(&desc));
    id.QuadPart = GetAdapterIdQuadPart();
    EXPECT_EQ(desc.AdapterLuid.LowPart, id.LowPart);
    EXPECT_EQ(desc.AdapterLuid.HighPart, id.HighPart);
    EXPECT_TRUE(m_device.Direct3D11Device() != nullptr);

    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
    EXPECT_EQ(m_session.Device().AdapterId(), m_device.AdapterId());
}

TEST_F(LearningModelSessionAPITests, EvaluateFeatures)
{
    std::vector<int64_t> shape = { 4 };
    std::vector<winrt::hstring> data = { L"one", L"two", L"three", L"four" };

    // create from buffer
    auto tensor = TensorString::CreateFromArray(shape, data);
    EXPECT_EQ(tensor.GetAsVectorView().Size(), data.size());
    EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

    // create from vector view
    auto dataCopy = data;
    tensor = TensorString::CreateFromIterable(
        shape, winrt::single_threaded_vector<winrt::hstring>(std::move(dataCopy)).GetView());
    EXPECT_EQ(tensor.GetAsVectorView().Size(), data.size());
    EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

    EXPECT_NO_THROW(LoadModel(L"id-tensor-string.onnx"));
    LearningModelSession session(m_model);

    auto outputTensor = TensorString::Create();

    std::map<hstring, winrt::Windows::Foundation::IInspectable> featuresstandardmap;
    featuresstandardmap[L"X"] = tensor;
    featuresstandardmap[L"Y"] = outputTensor;
    auto featureswinrtmap = winrt::single_threaded_map(std::move(featuresstandardmap));
    session.EvaluateFeatures(featureswinrtmap, L"0");

    // verify identity model round-trip works
    EXPECT_EQ(outputTensor.GetAsVectorView().Size(), data.size());
    EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(outputTensor.GetAsVectorView())));
}

TEST_F(LearningModelSessionAPITests, EvaluateFeaturesAsync)
{
    std::vector<int64_t> shape = { 4 };
    std::vector<winrt::hstring> data = { L"one", L"two", L"three", L"four" };

    // create from buffer
    auto tensor = TensorString::CreateFromArray(shape, data);
    EXPECT_EQ(tensor.GetAsVectorView().Size(), data.size());
    EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

    // create from vector view
    auto dataCopy = data;
    tensor = TensorString::CreateFromIterable(
        shape, winrt::single_threaded_vector<winrt::hstring>(std::move(dataCopy)).GetView());
    EXPECT_EQ(tensor.GetAsVectorView().Size(), data.size());
    EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(tensor.GetAsVectorView())));

    EXPECT_NO_THROW(LoadModel(L"id-tensor-string.onnx"));
    LearningModelSession session(m_model);

    auto outputTensor = TensorString::Create(shape);

    std::map<hstring, winrt::Windows::Foundation::IInspectable> featuresstandardmap;
    featuresstandardmap[L"X"] = tensor;
    featuresstandardmap[L"Y"] = outputTensor;
    auto featureswinrtmap = winrt::single_threaded_map(std::move(featuresstandardmap));
    session.EvaluateFeaturesAsync(featureswinrtmap, L"0").get();

    // verify identity model round-trip works
    EXPECT_EQ(outputTensor.GetAsVectorView().Size(), data.size());
    EXPECT_TRUE(std::equal(data.cbegin(), data.cend(), begin(outputTensor.GetAsVectorView())));
}

TEST_F(LearningModelSessionAPITests, EvaluationProperties)
{
    // load a model
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));
    // create a session
    m_session = LearningModelSession(m_model);
    // set a property
    auto value = winrt::Windows::Foundation::PropertyValue::CreateBoolean(true);
    m_session.EvaluationProperties().Insert(L"propName1", value);
    // get the property and make sure it's there with the right value
    auto value2 = m_session.EvaluationProperties().Lookup(L"propName1");
    EXPECT_EQ(value2.as<IPropertyValue>().GetBoolean(), true);
}

static LearningModelSession CreateSession(LearningModel model)
{
    LearningModelDevice device(nullptr);
    EXPECT_NO_THROW(device = LearningModelDevice(LearningModelDeviceKind::DirectX));

    LearningModelSession session(nullptr);
    if (DeviceHelpers::IsFloat16Supported(device))
    {
        EXPECT_NO_THROW(session = LearningModelSession(model, device));
    }
    else
    {
        EXPECT_THROW_SPECIFIC(
            session = LearningModelSession(model, device),
            winrt::hresult_error,
            [](const winrt::hresult_error& e) -> bool
        {
            return e.code() == DXGI_ERROR_UNSUPPORTED;
        });
    }

    return session;
}

TEST_F(LearningModelSessionAPITests, CreateSessionWithCastToFloat16InModel)
{
    // load a model
    EXPECT_NO_THROW(LoadModel(L"fp16-truncate-with-cast.onnx"));

    CreateSession(m_model);
}

TEST_F(LearningModelSessionAPITests, DISABLED_CreateSessionWithFloat16InitializersInModel)
{
    // Disabled due to https://microsoft.visualstudio.com/DefaultCollection/OS/_workitems/edit/21624720:
    // Model fails to resolve due to ORT using incorrect IR version within partition

    // load a model
    EXPECT_NO_THROW(LoadModel(L"fp16-initializer.onnx"));

    CreateSession(m_model);
}

static void EvaluateSessionAndCloseModel(
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

    EXPECT_NO_THROW(session = LearningModelSession(model, device, options));

    std::vector<float> input(1000);
    std::iota(std::begin(input), std::end(input), 0.0f);
    auto tensor_input = TensorFloat::CreateFromShapeArrayAndDataArray(shape, input);
    auto binding = LearningModelBinding(session);
    binding.Bind(L"input", tensor_input);

    LearningModelEvaluationResult result(nullptr);
    EXPECT_NO_THROW(result = session.Evaluate(binding, L""));

    if (close_model_on_session_creation)
    {
        // ensure that the model has been closed
        EXPECT_THROW_SPECIFIC(
            LearningModelSession(model, device, options),
            winrt::hresult_error,
            [](const winrt::hresult_error& e) -> bool
        {
            return e.code() == E_INVALIDARG;
        });
    }
    else
    {
        EXPECT_NO_THROW(LearningModelSession(model, device, options));
    }
}

TEST_F(LearningModelSessionAPITests, EvaluateSessionAndCloseModel)
{
    EXPECT_NO_THROW(::EvaluateSessionAndCloseModel(LearningModelDeviceKind::Cpu, true));
    EXPECT_NO_THROW(::EvaluateSessionAndCloseModel(LearningModelDeviceKind::Cpu, false));
}

TEST_F(LearningModelSessionAPITests, CloseSession)
{
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));
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
    EXPECT_NO_THROW(session = LearningModelSession(m_model));

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
    EXPECT_NO_THROW(session.Close());

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
    std::wstring author(m_model.Author());
    EXPECT_EQ(author, L"onnx-caffe2");

    // verify that session throws RO_E_CLOSED error
    std::vector<float> input(1 * 3 * 224 * 224, 0);
    std::vector<int64_t> shape = { 1, 3, 224, 224 };
    auto tensor_input = TensorFloat::CreateFromShapeArrayAndDataArray(shape, input);
    EXPECT_THROW_SPECIFIC(LearningModelBinding binding(session),
        winrt::hresult_error,
        [](const winrt::hresult_error &e) -> bool
    {
        return e.code() == RO_E_CLOSED;
    });
 }
