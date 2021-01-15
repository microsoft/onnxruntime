// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"

#include "APITest.h"
#include "CommonDeviceHelpers.h"
#include "LearningModelSessionAPITest.h"
#include "protobufHelpers.h"
#include "winrt/Windows.Storage.h"

#include "winrt/Microsoft.AI.MachineLearning.Experimental.h"

#include <D3d11_4.h>
#include <dxgi1_6.h>
#include "Psapi.h"

using namespace winrt;
using namespace winml;
using namespace wfc;

using wf::IPropertyValue;

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
  auto tensor_input = TensorFloat::CreateFromShapeArrayAndDataArray(shape, input);
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
  std::vector<int64_t> shape = {1, 3, 224, 224};
  auto tensor_input = TensorFloat::CreateFromShapeArrayAndDataArray(shape, input);
  WINML_EXPECT_THROW_SPECIFIC(LearningModelBinding binding(session),
                              winrt::hresult_error,
                              [](const winrt::hresult_error& e) -> bool {
                                return e.code() == RO_E_CLOSED;
                              });
}

static void TestSpectrogram(int64_t batch_size, int64_t num_samples, int64_t onesided_dft_length,
    int64_t n_mel_bins, int64_t sampling_rate) {
  printf("\nTestSpectrogram\n");

  using namespace winml_experimental;
  using Operator = winml_experimental::LearningModelOperator;

  static const wchar_t MS_DOMAIN[] = L"com.microsoft";

  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> input_shape = {batch_size, num_samples};
  std::vector<int64_t> dft_shape = {batch_size, onesided_dft_length, 2};
  std::vector<int64_t> powerframe_shape = {batch_size, onesided_dft_length, 1};
  std::vector<int64_t> mel_matrix_shape = {onesided_dft_length, n_mel_bins};
  std::vector<int64_t> mel_spectrogram_shape = {batch_size, n_mel_bins};

  // Constant initializers

  // real slice
  auto real_slice_start = TensorInt32Bit::CreateFromShapeArrayAndDataArray({3}, {0, 0, 0});
  auto real_slice_ends = TensorInt32Bit::CreateFromShapeArrayAndDataArray({3}, {INT_MAX, INT_MAX, 1});

  // complex slice
  auto complex_slice_start = TensorInt32Bit::CreateFromShapeArrayAndDataArray({3}, {0, 0, 1});
  auto complex_slice_ends = TensorInt32Bit::CreateFromShapeArrayAndDataArray({3}, {INT_MAX, INT_MAX, 2});
  auto power_of_2_exponent = TensorFloat::CreateFromShapeArrayAndDataArray({}, {2});
  auto number_of_samples = TensorFloat::CreateFromShapeArrayAndDataArray({}, { static_cast<float>(num_samples) });

  // input slice
  auto input_slice_start = TensorInt32Bit::CreateFromShapeArrayAndDataArray({1}, {1});
  auto input_slice_ends = TensorInt32Bit::CreateFromShapeArrayAndDataArray({1}, {2});

  auto mel_input_shape = TensorInt64Bit::CreateFromShapeArrayAndDataArray({2}, {batch_size, onesided_dft_length});

  // Mel spectrogram
  auto num_mel_bins = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {n_mel_bins});
  auto sample_rate = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {sampling_rate});
  auto lower_edge_hertz = TensorFloat::CreateFromShapeArrayAndDataArray({}, {0});
  auto upper_edge_hertz = TensorFloat::CreateFromShapeArrayAndDataArray({}, {sampling_rate/2.f});

  auto model =
      LearningModelBuilder::Create()
          // Inputs
          .Inputs().Add(TensorFeatureDescriptor(L"Input.TimeSignal", L"The input time domain signal", TensorKind::Float, input_shape))
          // Outputs
          .Outputs().Add(TensorFeatureDescriptor(L"Output.OnesidedDFT", L"The onesided output of the frequency domain spectra", TensorKind::Float, dft_shape))
          .Outputs().Add(TensorFeatureDescriptor(L"Output.PowerFrames", L"The real output of the frequency domain spectra", TensorKind::Float, powerframe_shape))
          .Outputs().Add(TensorFeatureDescriptor(L"Output.MelWeightMatrix", L"The mel matrix", TensorKind::Float, mel_matrix_shape))
          .Outputs().Add(TensorFeatureDescriptor(L"Output.MelSpectrogram", L"The output spectrogram", TensorKind::Float, mel_spectrogram_shape))
          // Constants
          .Inputs().AddConstant(L"power_of_2_exponent", power_of_2_exponent)
          // The graph
          .Operators().Add(Operator(L"Shape", L"shape0")
              .SetInput(L"data", L"Input.TimeSignal")
              .SetOutput(L"shape", L"input_shape"))
          .Operators().Add(Operator(L"Slice", L"input_slice")
              .SetInput(L"data", L"input_shape")
              .SetConstant(L"starts", input_slice_start) // make_tensor<int32_t>({1}, {2}))
              .SetConstant(L"ends", input_slice_ends)
              .SetOutput(L"output", L"num_samples"))
          .Operators().Add(Operator(L"HannWindow", L"hann0", MS_DOMAIN)
              .SetInput(L"size", L"num_samples")
              .SetOutput(L"output", L"hann_window"))
          .Operators().Add(Operator(L"Mul", L"mul0")
              .SetInput(L"A", L"hann_window")
              .SetInput(L"B", L"Input.TimeSignal")
              .SetOutput(L"C", L"windowed_signal"))
          .Operators().Add(Operator(L"DFT", L"dft0", MS_DOMAIN)
              .SetInput(L"input", L"windowed_signal")
              .SetOutput(L"output", L"Output.OnesidedDFT"))
          .Operators().Add(Operator(L"Slice", L"real_slice")
              .SetInput(L"data", L"Output.OnesidedDFT")
              .SetConstant(L"starts", real_slice_start)
              .SetConstant(L"ends", real_slice_ends)
              .SetOutput(L"output", L"reals"))
          .Operators().Add(Operator(L"Slice", L"complex_slice")
              .SetInput(L"data", L"Output.OnesidedDFT")
              .SetConstant(L"starts", complex_slice_start)
              .SetConstant(L"ends", complex_slice_ends)
              .SetOutput(L"output", L"imaginaries"))
          .Operators().Add(Operator(L"Pow", L"real_pow")
              .SetInput(L"X", L"reals")
              .SetInput(L"Y", L"power_of_2_exponent")
              .SetOutput(L"Z", L"reals_squared"))
          .Operators().Add(Operator(L"Pow", L"complex_pow")
              .SetInput(L"X", L"imaginaries")
              .SetInput(L"Y", L"power_of_2_exponent")
              .SetOutput(L"Z", L"imaginaries_squared"))
          .Operators().Add(Operator(L"Add", L"add0")
              .SetInput(L"A", L"reals_squared")
              .SetInput(L"B", L"imaginaries_squared")
              .SetOutput(L"C", L"magnitude_squared"))
          .Operators().Add(Operator(L"Div", L"div0")
              .SetInput(L"A", L"magnitude_squared")
              .SetConstant(L"B", number_of_samples)
              .SetOutput(L"C", L"Output.PowerFrames"))
          .Operators().Add(Operator(L"MelWeightMatrix", L"melweightmatrix0", MS_DOMAIN)
              .SetConstant(L"num_mel_bins", num_mel_bins)
              .SetInput(L"dft_length", L"num_samples")
              .SetConstant(L"sample_rate", sample_rate)
              .SetConstant(L"lower_edge_hertz", lower_edge_hertz)
              .SetConstant(L"upper_edge_hertz", upper_edge_hertz)
              .SetOutput(L"output", L"Output.MelWeightMatrix"))
          .Operators().Add(Operator(L"Reshape", L"reshape0")
              .SetInput(L"data", L"Output.PowerFrames")
              .SetConstant(L"shape", mel_input_shape)
              .SetOutput(L"reshaped", L"ReshapedOutput"))      
          .Operators().Add(Operator(L"MatMul", L"matmul0")
              .SetInput(L"A", L"ReshapedOutput")
              .SetInput(L"B", L"Output.MelWeightMatrix")
              .SetOutput(L"Y", L"Output.MelSpectrogram"))
          .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Populate binding
  std::vector<float> input = {1, 2, 3, 4, 1, 0, 0, 0, 1, 2, 3, 4, 1, 0, 0, 0, 1, 2, 3, 4, 1, 0, 0, 0, 1, 2, 3, 4, 1, 0, 0, 0};
  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromShapeArrayAndDataArray(input_shape, input));

  // These are constant initializers. Constants should be automatically generated and set... but since that is not implemented... they need to be duplicated
  binding.Bind(L"real_slice.starts", real_slice_start);
  binding.Bind(L"real_slice.ends", real_slice_ends);
  binding.Bind(L"complex_slice.starts", complex_slice_start);
  binding.Bind(L"complex_slice.ends", complex_slice_ends);
  binding.Bind(L"input_slice.starts", input_slice_start);
  binding.Bind(L"input_slice.ends", input_slice_ends);
  binding.Bind(L"div0.B", number_of_samples); 
  binding.Bind(L"melweightmatrix0.num_mel_bins", num_mel_bins);
  binding.Bind(L"melweightmatrix0.sample_rate", sample_rate);
  binding.Bind(L"melweightmatrix0.lower_edge_hertz", lower_edge_hertz);
  binding.Bind(L"melweightmatrix0.upper_edge_hertz", upper_edge_hertz);
  binding.Bind(L"reshape0.shape", mel_input_shape);
  binding.Bind(L"power_of_2_exponent", power_of_2_exponent);

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // Check results
  printf("DFT\n");
  auto dft_tensor = result.Outputs().Lookup(L"Output.OnesidedDFT").as<TensorFloat>();
  auto dft_ivv = dft_tensor.GetAsVectorView();
  for (int i = 0; i < dft_shape[0] * dft_shape[1] * 2; i += 2) {
    printf("%f + %fi\n", dft_ivv.GetAt(i), dft_ivv.GetAt(i + 1));
  }

  printf("PowerFrames\n");
  auto y_tensor = result.Outputs().Lookup(L"Output.PowerFrames").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  for (int i = 0; i < powerframe_shape[0] * powerframe_shape[1]; i++) {
    printf("%f\n", y_ivv.GetAt(i));
  }

  auto mel_tensor = result.Outputs().Lookup(L"Output.MelWeightMatrix").as<TensorFloat>();
  auto mel_ivv = mel_tensor.GetAsVectorView();
  printf("MelWeightMatrix\n");
  for (int i = 0; i < mel_matrix_shape[0]; i++) {
    for (int j = 0; j < mel_matrix_shape[1]; j++) {
      printf("%.2f,", mel_ivv.GetAt((i * static_cast<int>(mel_matrix_shape[1])) + j));
    }
    printf("\n");
  }

  auto final_tensor = result.Outputs().Lookup(L"Output.MelSpectrogram").as<TensorFloat>();
  auto final_ivv = final_tensor.GetAsVectorView();
  printf("MelTransformedOutput\n");
  for (int i = 0; i < mel_spectrogram_shape[0] * mel_spectrogram_shape[1]; i++) {
    printf("%f\n", final_ivv.GetAt(i));
  }
}

static void TestWindowFunction(const wchar_t* window_operator) {
  printf("\n%ls\n", window_operator);
    using namespace winml_experimental;
    using Operator = winml_experimental::LearningModelOperator;

    std::vector<int64_t> scalar_shape = {};
    std::vector<int64_t> output_shape = {32};
    auto double_data_type = TensorInt32Bit::CreateFromArray(std::vector<int64_t>({1}), std::vector<int32_t>({7}));
    auto model = 
        LearningModelBuilder::Create()
                .Inputs().Add(TensorFeatureDescriptor(L"Input", L"The input time domain signal", TensorKind::Int64, scalar_shape))
                .Outputs().Add(TensorFeatureDescriptor(L"Output", L"The output frequency domain spectra", TensorKind::Float, output_shape))
                .Operators().Add(Operator(window_operator, L"window0", L"com.microsoft").SetInput(L"size", L"Input").SetOutput(L"output", L"Output"))//.SetAttribute(L"output_datatype", double_data_type))
                .CreateModel();

    LearningModelSession session(model);
    LearningModelBinding binding(session);

    std::vector<int64_t> x = { 256 };
    binding.Bind(L"Input", TensorInt64Bit::CreateFromShapeArrayAndDataArray(scalar_shape, x));

    // Evaluate
    auto result = session.Evaluate(binding, L"");

    // Check results
    auto y_tensor = result.Outputs().Lookup(L"Output").as<TensorFloat>();
    auto y_ivv = y_tensor.GetAsVectorView();
    for (int i = 0; i < output_shape[0]; i ++) {
      printf("%f\n", y_ivv.GetAt(i));
    }
}


static void TestDFT() {
  printf("\nDFT\n");
  using namespace winml_experimental;
  using Operator = winml_experimental::LearningModelOperator;

  std::vector<int64_t> shape = {1, 5};
  std::vector<int64_t> output_shape = {1, 5, 2};

  //bool onesided = true;
  //if (onesided) {
  float fft_output = std::floor(output_shape[1] / 2.f) + 1.f;
  output_shape[1] = static_cast<int64_t>(fft_output);
  //}

  auto model =
      LearningModelBuilder::Create()
          .Inputs().Add(TensorFeatureDescriptor(L"Input", L"The input time domain signal", TensorKind::Float, shape))
          .Outputs().Add(TensorFeatureDescriptor(L"Output", L"The output frequency domain spectra", TensorKind::Float, output_shape))
          .Operators().Add(Operator(L"DFT", L"dft0", L"com.microsoft").SetInput(L"input", L"Input").SetOutput(L"output", L"Output"))
          //.Operators().Add(Operator(L"DFT", L"dft0", L"com.microsoft").SetInput(L"input", L"Input").SetOutput(L"output", L"dft0_output"))
          //.Operators().Add(Operator(L"IDFT", L"idft0", L"com.microsoft").SetInput(L"input", L"dft0_output").SetOutput(L"output", L"Output"))
          .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Populate binding
  //std::vector<float> x = {1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0};
  //std::vector<float> x = {1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0};
  //std::vector<float> x = {1, 2, 3, 4, 1, 0, 0, 0, 1, 2, 3, 4, 1, 0, 0, 0, 1, 2, 3, 4, 1, 0, 0, 0, 1, 2, 3, 4, 1, 0, 0, 0};
  //std::vector<float> x = {1, 0, 2, 0, 3, 0, 4, 0};
  std::vector<float> x = {
      1, 2, 3, 4, 5};
  binding.Bind(L"Input", TensorFloat::CreateFromShapeArrayAndDataArray(shape, x));

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // Check results
  auto y_tensor = result.Outputs().Lookup(L"Output").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  for (int i = 0; i < output_shape[0] * output_shape[1] * 2; i += 2) {
    printf("%f + %fi\n", y_ivv.GetAt(i), y_ivv.GetAt(i + 1));
  }
}

template <typename T>
static auto make_pure_frequency(float frequency_in_hertz, int64_t signal_size, int64_t sample_rate) {
  float amplitude = 4;
  float angular_velocity = frequency_in_hertz * 2 * 3.1415f;
  std::vector<T> signal(signal_size);
  for (int64_t i = 0; i < signal_size; i++) {
    T time = i / static_cast<T>(sample_rate);
    signal[i] = amplitude * cos(angular_velocity * time);
  }
  return signal;
}

template <typename T>
static auto make_middle_c(int64_t signal_size, int64_t sample_rate) {
  float middle_c_in_hertz = 261.626f;
  return make_pure_frequency<T>(middle_c_in_hertz, signal_size, sample_rate);
}

template <typename T>
static auto make_c2(int64_t signal_size, int64_t sample_rate) {
  float middle_c_in_hertz = 261.626f * 2;
  return make_pure_frequency<T>(middle_c_in_hertz, signal_size, sample_rate);
}

template <typename T>
static auto make_c4(int64_t signal_size, int64_t sample_rate) {
  float middle_c_in_hertz = 261.626f * 4;
  return make_pure_frequency<T>(middle_c_in_hertz, signal_size, sample_rate);
}

template <typename T>
static auto make_3_tones(int64_t signal_size, int64_t sample_rate) {
  auto middle_c = make_middle_c<T>(signal_size, sample_rate);
  auto c2 = make_c2<T>(signal_size, sample_rate);
  auto c4 = make_c4<T>(signal_size, sample_rate);
  for (int64_t i = 0; i < signal_size; i++) {
    middle_c[i] = (i < signal_size / 3) ?
                    middle_c[i] :
                    (i < 2*signal_size/3) ?
                        (middle_c[i] + c2[i]) :
                        (middle_c[i] + c2[i] + c4[i]);
  }
  return middle_c;
}


static void TestSTFT(int64_t batch_size, int64_t signal_size, int64_t dft_size, int64_t hop_size) {
  printf("\nSTFT\n");
  using namespace winml_experimental;
  using Operator = winml_experimental::LearningModelOperator;

  static const wchar_t MS_DOMAIN[] = L"com.microsoft";

  auto number_of_dfts = static_cast<int64_t>(ceil((signal_size - dft_size) / hop_size));

  std::vector<int64_t> input_shape = {1, signal_size};
  int64_t onesided_dft_size = static_cast<int64_t>(floor(dft_size / 2.f) + 1);
  std::vector<int64_t> output_shape = {
      batch_size,
      number_of_dfts,
      onesided_dft_size,
      2
  };

  // input slice
  auto input_slice_start = TensorInt32Bit::CreateFromShapeArrayAndDataArray({1}, {1});
  auto input_slice_ends = TensorInt32Bit::CreateFromShapeArrayAndDataArray({1}, {2});

  auto frame_step = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {hop_size});
  auto dft_length = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {dft_size});

  std::vector<int64_t> window_shape = {dft_size};

  auto model =
      LearningModelBuilder::Create()
          .Inputs().Add(TensorFeatureDescriptor(L"Input.TimeSignal", L"The input time domain signal", TensorKind::Float, input_shape))
          .Outputs().Add(TensorFeatureDescriptor(L"Output.STFT", L"The output frequency domain spectra", TensorKind::Float, output_shape))
          .Outputs().Add(TensorFeatureDescriptor(L"Output.HannWindow", L"The HannWindow used", TensorKind::Float, window_shape))
          .Operators().Add(Operator(L"HannWindow", L"hann0", MS_DOMAIN)
              .SetConstant(L"size", dft_length)
              .SetOutput(L"output", L"Output.HannWindow"))
          .Operators().Add(Operator(L"STFT", L"stft0", MS_DOMAIN)
              .SetInput(L"signal", L"Input.TimeSignal")
              .SetInput(L"window", L"Output.HannWindow")
              .SetConstant(L"frame_length", dft_length)
              .SetConstant(L"frame_step", frame_step)
              .SetOutput(L"output", L"Output.STFT"))
          .CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);


  printf("\nSignal DFT First Input\n\n");

  // Populate binding
  auto signal = make_middle_c<float>(signal_size, 8192);
  for (int64_t i = 0; i < dft_size + 128; i++) {
    printf("%f\n", signal[i]);
  }


  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromShapeArrayAndDataArray(input_shape, signal));
  binding.Bind(L"hann0.size", dft_length);
  binding.Bind(L"stft0.frame_length", dft_length);
  binding.Bind(L"stft0.frame_step", frame_step);

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  
  printf("\n Hann Window\n\n");
  auto window_tensor = result.Outputs().Lookup(L"Output.HannWindow").as<TensorFloat>();
  auto window_ivv = window_tensor.GetAsVectorView();
  for (int64_t i = 0; i < window_ivv.Size(); i++) {
    printf("%f \n", window_ivv.GetAt(static_cast<uint32_t>(i)));
  }

  printf("\n STFT Output\n\n");
  // Check results
  auto y_tensor = result.Outputs().Lookup(L"Output.STFT").as<TensorFloat>();
  auto y_ivv = y_tensor.GetAsVectorView();
  auto size = y_ivv.Size();

  if (size == number_of_dfts * onesided_dft_size * 2) {
    for (int64_t dft_idx = 0; dft_idx < number_of_dfts; dft_idx++) {
      for (int64_t i = 0; i < onesided_dft_size; i++) {
        auto real_idx = static_cast<uint32_t>((i * 2) + (2 * dft_idx * onesided_dft_size));
        printf("%d , %f , %f\n", static_cast<uint32_t>(i), y_ivv.GetAt(real_idx), y_ivv.GetAt(real_idx + 1));
      }
    }
  }
 }

static void TestMiddleCSpectrogram(
     int64_t batch_size, int64_t signal_size, int64_t window_size, int64_t dft_size, int64_t hop_size,
    int64_t n_mel_bins, int64_t sampling_rate) {
  printf("\nTest MiddleC Spectrogram\n");

  using namespace winml_experimental;
  using Operator = winml_experimental::LearningModelOperator;

  static const wchar_t MS_DOMAIN[] = L"com.microsoft";

  auto number_of_dfts = static_cast<int64_t>(ceil((signal_size - dft_size) / hop_size));
  // Input and output shapes
  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> input_shape = {batch_size, signal_size};
  std::vector<int64_t> mel_spectrogram_shape = {batch_size, 1, number_of_dfts, n_mel_bins};

  // Constant initializers
  auto power_of_2_exponent = TensorFloat::CreateFromShapeArrayAndDataArray({}, {2});
  auto frame_step = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {hop_size});
  auto window_length = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {window_size});
  auto dft_length = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {dft_size});
  auto dft_length_as_float = TensorFloat::CreateFromShapeArrayAndDataArray({}, {static_cast<float>(dft_size)});

  // real slice
  auto real_slice_start = TensorInt32Bit::CreateFromShapeArrayAndDataArray({4}, {0, 0, 0, 0});
  auto real_slice_ends = TensorInt32Bit::CreateFromShapeArrayAndDataArray({4}, {INT_MAX, INT_MAX, INT_MAX, 1});

  // complex slice
  auto complex_slice_start = TensorInt32Bit::CreateFromShapeArrayAndDataArray({4}, {0, 0, 0, 1});
  auto complex_slice_ends = TensorInt32Bit::CreateFromShapeArrayAndDataArray({4}, {INT_MAX, INT_MAX, INT_MAX, 2});

  int64_t onesided_dft_size = static_cast<int64_t>(floor(dft_size / 2.f) + 1);
  auto mel_input_shape = TensorInt64Bit::CreateFromShapeArrayAndDataArray({2}, {batch_size * number_of_dfts, onesided_dft_size});
  auto output_shape = TensorInt64Bit::CreateFromShapeArrayAndDataArray({4}, mel_spectrogram_shape);

  // Mel spectrogram
  auto num_mel_bins = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {n_mel_bins});
  auto sample_rate = TensorInt64Bit::CreateFromShapeArrayAndDataArray({}, {sampling_rate});
  auto lower_edge_hertz = TensorFloat::CreateFromShapeArrayAndDataArray({}, {0});
  auto upper_edge_hertz = TensorFloat::CreateFromShapeArrayAndDataArray({}, {sampling_rate / 2.f});

  auto builder =
      LearningModelBuilder::Create()
          // Inputs
          .Inputs()
          .Add(TensorFeatureDescriptor(L"Input.TimeSignal", L"The input time domain signal", TensorKind::Float, input_shape))
          .Inputs()
          .AddConstant(L"power_of_2_exponent", power_of_2_exponent)
          // Outputs
          .Outputs()
          .Add(TensorFeatureDescriptor(L"Output.MelSpectrogram", L"The output spectrogram", TensorKind::Float, mel_spectrogram_shape))
          // The graph
          .Operators()
          .Add(Operator(L"HannWindow", L"hann0", MS_DOMAIN)
                   .SetConstant(L"size", window_length)  // SetConstant not implemented: bind hann0.size to window_length
                   .SetOutput(L"output", L"hann_window"))
          .Operators()
          .Add(Operator(L"STFT", L"stft0", MS_DOMAIN)
                   .SetInput(L"signal", L"Input.TimeSignal")
                   .SetInput(L"window", L"hann_window")
                   .SetConstant(L"frame_length", dft_length)  // SetConstant not implemented: bind stft0.dft_length to dft_length
                   .SetConstant(L"frame_step", frame_step)    // SetConstant not implemented: bind stft0.frame_step to frame_step
                   .SetOutput(L"output", L"stft_output"))
          .Operators()
          .Add(Operator(L"Slice", L"real_slice")
                   .SetInput(L"data", L"stft_output")
                   .SetConstant(L"starts", real_slice_start)
                   .SetConstant(L"ends", real_slice_ends)
                   .SetOutput(L"output", L"reals"))
          .Operators()
          .Add(Operator(L"Slice", L"complex_slice")
                   .SetInput(L"data", L"stft_output")
                   .SetConstant(L"starts", complex_slice_start)
                   .SetConstant(L"ends", complex_slice_ends)
                   .SetOutput(L"output", L"imaginaries"))
          .Operators()
          .Add(Operator(L"Pow", L"real_pow")
                   .SetInput(L"X", L"reals")
                   .SetInput(L"Y", L"power_of_2_exponent")  // SetConstant not implemented: bind real_pow.Y to power_of_2_exponent
                   .SetOutput(L"Z", L"reals_squared"))
          .Operators()
          .Add(Operator(L"Pow", L"complex_pow")
                   .SetInput(L"X", L"imaginaries")
                   .SetInput(L"Y", L"power_of_2_exponent")  // SetConstant not implemented: bind complex_pow.Y to power_of_2_exponent
                   .SetOutput(L"Z", L"imaginaries_squared"))
          .Operators()
          .Add(Operator(L"Add", L"add0")
                   .SetInput(L"A", L"reals_squared")
                   .SetInput(L"B", L"imaginaries_squared")
                   .SetOutput(L"C", L"magnitude_squared"))
          .Operators()
          .Add(Operator(L"Div", L"div0")
                   .SetInput(L"A", L"magnitude_squared")
                   .SetConstant(L"B", dft_length_as_float)  // SetConstant not implemented: bind div0.B to dft_length_as_float
                   .SetOutput(L"C", L"power_frames"))
          .Operators()
          .Add(Operator(L"MelWeightMatrix", L"melweightmatrix0", MS_DOMAIN)
                   .SetConstant(L"num_mel_bins", num_mel_bins)  // SetConstant not implemented: bind melweightmatrix0.num_mel_bins to num_mel_bins
                   .SetConstant(L"dft_length", dft_length)      // SetConstant not implemented: bind melweightmatrix0.dft_length to dft_length
                   .SetConstant(L"sample_rate", sample_rate)
                   .SetConstant(L"lower_edge_hertz", lower_edge_hertz)  // SetConstant not implemented: bind melweightmatrix0.lower_edge_hertz to lower_edge_hertz
                   .SetConstant(L"upper_edge_hertz", upper_edge_hertz)  // SetConstant not implemented: bind melweightmatrix0.upper_edge_hertz to upper_edge_hertz
                   .SetOutput(L"output", L"mel_weight_matrix"))
          .Operators()
          .Add(Operator(L"Reshape", L"reshape0")
                   .SetInput(L"data", L"power_frames")
                   .SetConstant(L"shape", mel_input_shape)
                   .SetOutput(L"reshaped", L"reshaped_output"))
          .Operators()
          .Add(Operator(L"MatMul", L"matmul0")
                   .SetInput(L"A", L"reshaped_output")
                   .SetInput(L"B", L"mel_weight_matrix")
                   .SetOutput(L"Y", L"mel_spectrogram"))
          .Operators()
          .Add(Operator(L"Reshape", L"reshape1")
                   .SetInput(L"data", L"mel_spectrogram")
                   .SetConstant(L"shape", output_shape)
                   .SetOutput(L"reshaped", L"Output.MelSpectrogram"));

  builder.Save(L"e:\\spectrogram.onnx");
  auto model = builder.CreateModel();

  LearningModelSession session(model);
  LearningModelBinding binding(session);

  // Populate binding
  auto signal = make_3_tones<float>(signal_size, 8192);
  binding.Bind(L"Input.TimeSignal", TensorFloat::CreateFromShapeArrayAndDataArray(input_shape, signal));
 
  // These are constant initializers. Constants should be automatically generated and set... but since that is not implemented... they need to be duplicated
  binding.Bind(L"hann0.size", window_length);
  binding.Bind(L"stft0.frame_length", dft_length);
  binding.Bind(L"stft0.frame_step", frame_step);
  binding.Bind(L"real_slice.starts", real_slice_start);
  binding.Bind(L"real_slice.ends", real_slice_ends);
  binding.Bind(L"complex_slice.starts", complex_slice_start);
  binding.Bind(L"complex_slice.ends", complex_slice_ends);

  binding.Bind(L"div0.B", dft_length_as_float);
  binding.Bind(L"melweightmatrix0.dft_length", dft_length);
  binding.Bind(L"melweightmatrix0.num_mel_bins", num_mel_bins);
  binding.Bind(L"melweightmatrix0.sample_rate", sample_rate);
  binding.Bind(L"melweightmatrix0.lower_edge_hertz", lower_edge_hertz);
  binding.Bind(L"melweightmatrix0.upper_edge_hertz", upper_edge_hertz);
  binding.Bind(L"reshape0.shape", mel_input_shape);
  binding.Bind(L"power_of_2_exponent", power_of_2_exponent);
  binding.Bind(L"reshape1.shape", output_shape);
  
  winrt::Windows::Media::VideoFrame output_image(
      winrt::Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8,
      static_cast<int32_t>(n_mel_bins),
      static_cast<int32_t>(number_of_dfts));

  binding.Bind(L"Output.MelSpectrogram", output_image);

  // Evaluate
  auto result = session.Evaluate(binding, L"");

  // check the output video frame object by saving output image to disk
  std::wstring out_name = L"mel_spectrogram.jpg";
  winrt::Windows::Storage::StorageFolder folder = winrt::Windows::Storage::StorageFolder::GetFolderFromPathAsync(L"e:\\").get();
  winrt::Windows::Storage::StorageFile file = folder.CreateFileAsync(out_name, winrt::Windows::Storage::CreationCollisionOption::ReplaceExisting).get();
  winrt::Windows::Storage::Streams::IRandomAccessStream write_stream = file.OpenAsync(winrt::Windows::Storage::FileAccessMode::ReadWrite).get();
  winrt::Windows::Graphics::Imaging::BitmapEncoder encoder = winrt::Windows::Graphics::Imaging::BitmapEncoder::CreateAsync(winrt::Windows::Graphics::Imaging::BitmapEncoder::JpegEncoderId(), write_stream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(output_image.SoftwareBitmap());
  encoder.FlushAsync().get();


  /*auto final_tensor = result.Outputs().Lookup(L"Output.MelSpectrogram").as<TensorFloat>();
  auto final_ivv = final_tensor.GetAsVectorView();
  printf("Output.MelSpectrogram\n\n");
  for (uint32_t i = 0; i < final_ivv.Size(); i++) {
    printf("%f\n", final_ivv.GetAt(i));
  }*/
}

static void TestGemm() {
  printf("\nGemm\n");
  using namespace winml_experimental;
  using Operator = winml_experimental::LearningModelOperator;

  // TODO: OPERATOR IDL SHOULD CONTAIN IMPLEMENTED OPERATOR TYPES AND BE GENERATED BY THE BUILD
  // TODO: FREE DIMENSIONS ARE NOT IMPLEMENTED
  // TODO: ATTRIBUTES ARE NOT IMPLEMENTED, IE - operator.SetAttribute(L"att", attribute))
  // TODO: CONSTANT INPUTS ARE NOT IMPLEMENTED, IE - builder.Inputs().AddConstant(L"c", constant_y)
  // TODO: SHOULD LEARNINGMODELBUILDER::CREATE ACCEPT OPSET VERSION MAP?
  // TODO: MAPFEATUREDESCRIPTOR AND SEQUENCEFEATUREDESCRIPTOR CONSTRUCTORS AND ABI METHODS ARE NOT IMPLEMENTED
  //
  // GEMM C INPUT SHOULD BE OPTIONAL BUT IT REQUIRED BY ONNX AS THE IMPLEMENTATION IS NO UP TO DATE

  std::vector<int64_t> shape = {3, 3};
  std::vector<float> x =
  {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  };
  auto model =
    LearningModelBuilder::Create()
      .Inputs().Add(TensorFeatureDescriptor(L"a", L"the a input", TensorKind::Float, shape))
      .Inputs().Add(TensorFeatureDescriptor(L"b", L"the b input", TensorKind::Float, shape))
      .Inputs().Add(TensorFeatureDescriptor(L"c", L"the c input", TensorKind::Float, shape))
      .Outputs().Add(TensorFeatureDescriptor(L"y", L"the y output", TensorKind::Float, shape))
      .Operators().Add(Operator(L"Gemm", L"gemm0", L"").SetInput(L"A", L"a").SetInput(L"B", L"b").SetInput(L"C", L"c").SetOutput(L"Y", L"y"))
      .CreateModel();

}

static void TestModelBuilding() {
  TestDFT();
  {
    int64_t batch_size = 1;
    int64_t num_samples = 32;
    int64_t onesided_dft_length = static_cast<int64_t>(std::floor(num_samples / 2) + 1);
    int64_t n_mel_bins = 10;
    int64_t sampling_rate = 48000;
    TestSpectrogram(batch_size, num_samples, onesided_dft_length, n_mel_bins, sampling_rate);
  }

  TestWindowFunction(L"HannWindow");
  TestWindowFunction(L"HammingWindow");
  TestWindowFunction(L"BlackmanWindow");
  TestGemm();

  
  {
    int64_t sample_rate = 8192;
    float signal_duration_in_seconds = 5.f;

    //float frame_duration_in_seconds = .025f;
    //float frame_stride_in_seconds = .01f;

    int64_t batch_size = 1;
    int64_t signal_size = static_cast<int64_t>(sample_rate * signal_duration_in_seconds);
    int64_t dft_size = 256;  //static_cast<int64_t>(sample_rate * frame_duration_in_seconds);
    int64_t hop_size = 128;  //static_cast<int64_t>(sample_rate * frame_stride_in_seconds);
    TestSTFT(batch_size, signal_size, dft_size, hop_size);
  }

  {
    int64_t batch_size = 1;

    int64_t sample_rate = 8192;
    float signal_duration_in_seconds = 5.f;
    int64_t signal_size = static_cast<int64_t>(sample_rate * signal_duration_in_seconds);
    int64_t window_size = 512;
    int64_t dft_size = 512;
    int64_t hop_size = 128;
    int64_t n_mel_bins = 1024;

    TestMiddleCSpectrogram(batch_size, signal_size, dft_size, window_size, hop_size, n_mel_bins, sample_rate);
  }
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
    SetIntraOpNumThreads,
    TestModelBuilding
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
