// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"

#include <d3dx12.h>

#include "CommonDeviceHelpers.h"
#include "CustomOperatorProvider.h"
#include "filehelpers.h"
#include "robuffer.h"
#include "scenariotestscppwinrt.h"
#include "Windows.Graphics.DirectX.Direct3D11.interop.h"
#include "windows.ui.xaml.media.dxinterop.h"

#include <d2d1.h>
#include <d3d11.h>
#include <initguid.h>
#include <MemoryBuffer.h>
#include <iostream>
#if __has_include("dxcore.h")
#define ENABLE_DXCORE 1
#endif
#ifdef ENABLE_DXCORE
#include <dxcore.h>
#endif

// lame, but WinBase.h redefines this, which breaks winrt headers later
#ifdef GetCurrentTime
#undef GetCurrentTime
#endif

#include <winrt/windows.ui.xaml.h>
#include <winrt/windows.ui.xaml.automation.peers.h>
#include <winrt/windows.ui.xaml.controls.h>
#include <winrt/windows.ui.xaml.media.imaging.h>
#include <winrt/windows.ui.xaml.media.animation.h>

using namespace winml;
using namespace wfc;
using namespace wm;
using namespace wgi;
using namespace wgdx;
using namespace ws;
using namespace wss;
using namespace winrt::Windows::UI::Xaml::Media::Imaging;
using namespace Windows::Graphics::DirectX::Direct3D11;

static void ScenarioCppWinrtTestsClassSetup() {
  winrt::init_apartment();
#ifdef BUILD_INBOX
  winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
}

static void Sample1() {
  LearningModel model = nullptr;
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(filePath));
}

ILearningModelFeatureValue MakeTensor(const ITensorFeatureDescriptor& descriptor) {
  auto dataType = descriptor.TensorKind();
  std::vector<int64_t> shape;
  int64_t size = 1;
  for (auto&& dim : descriptor.Shape()) {
    if (dim == -1) dim = 1;
    shape.push_back(dim);
    size *= dim;
  }

  switch (dataType) {
    case TensorKind::Float: {
      std::vector<float> buffer;
      buffer.resize(static_cast<size_t>(size));
      auto ftv = TensorFloat::CreateFromIterable(shape, winrt::single_threaded_vector<float>(std::move(buffer)));
      return ftv;
    }
    default:
      winrt::throw_hresult(E_NOTIMPL);
      break;
  }
}

ILearningModelFeatureValue MakeImage(const IImageFeatureDescriptor& /*descriptor*/, wf::IInspectable data) {
  VideoFrame videoFrame = nullptr;
  if (data != nullptr) {
    SoftwareBitmap sb = nullptr;
    data.as(sb);
    videoFrame = VideoFrame::CreateWithSoftwareBitmap(sb);
  } else {
    SoftwareBitmap sb = SoftwareBitmap(BitmapPixelFormat::Bgra8, 28, 28);
    videoFrame = VideoFrame::CreateWithSoftwareBitmap(sb);
  }
  auto imageValue = ImageFeatureValue::CreateFromVideoFrame(videoFrame);
  return imageValue;
}

ILearningModelFeatureValue FeatureValueFromFeatureValueDescriptor(ILearningModelFeatureDescriptor descriptor, wf::IInspectable data = nullptr) {
  auto kind = descriptor.Kind();
  switch (kind) {
    case LearningModelFeatureKind::Image: {
      ImageFeatureDescriptor imageDescriptor = nullptr;
      descriptor.as(imageDescriptor);
      return MakeImage(imageDescriptor, data);
    }
    case LearningModelFeatureKind::Map:
      winrt::throw_hresult(E_NOTIMPL);
      break;
    case LearningModelFeatureKind::Sequence:
      winrt::throw_hresult(E_NOTIMPL);
      break;
    case LearningModelFeatureKind::Tensor: {
      TensorFeatureDescriptor tensorDescriptor = nullptr;
      descriptor.as(tensorDescriptor);
      return MakeTensor(tensorDescriptor);
    }
    default:
      winrt::throw_hresult(E_INVALIDARG);
      break;
  }
}

// helper method that populates a binding object with default data
static void BindFeatures(LearningModelBinding binding, IVectorView<ILearningModelFeatureDescriptor> features) {
  for (auto&& feature : features) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(feature);
    // set an actual buffer here. we're using uninitialized data for simplicity.
    binding.Bind(feature.Name(), featureValue);
  }
}

//! Scenario1 : Load , bind, eval a model using all the system defaults (easy path)
static void Scenario1LoadBindEvalDefault() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the default device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the input and the output buffers by name
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(input);
    // set an actual buffer here. we're using uninitialized data for simplicity.
    binding.Bind(input.Name(), featureValue);
  }
  // run eval
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

//! Scenario2: Load a model from stream
//          - winRT, and win32
static void Scenario2LoadModelFromStream() {
  // get a stream
  std::wstring path = FileHelpers::GetModulePath() + L"model.onnx";
  auto storageFile = StorageFile::GetFileFromPathAsync(path).get();

  // load the stream
  Streams::IRandomAccessStreamReference streamref;
  storageFile.as(streamref);

  // load a model
  LearningModel model = nullptr;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromStreamAsync(streamref).get());
  WINML_EXPECT_TRUE(model != nullptr);
}

//! Scenario3: pass a SoftwareBitmap into a model
static void Scenario3SoftwareBitmapInputBinding() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the default device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the input and the output buffers by name
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    // load the SoftwareBitmap
    SoftwareBitmap sb = FileHelpers::GetSoftwareBitmapFromFile(FileHelpers::GetModulePath() + L"fish.png");
    auto videoFrame = VideoFrame::CreateWithSoftwareBitmap(sb);
    auto imageValue = ImageFeatureValue::CreateFromVideoFrame(videoFrame);

    WINML_EXPECT_NO_THROW(binding.Bind(input.Name(), imageValue));
  }
  // run eval
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

//! Scenario5: run an async eval
wf::IAsyncOperation<LearningModelEvaluationResult> DoEvalAsync() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the default device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the input and the output buffers by name
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(input);
    // set an actual buffer here. we're using uninitialized data for simplicity.
    binding.Bind(input.Name(), featureValue);
  }
  // run eval async
  return session.EvaluateAsync(binding, L"");
}

static void Scenario5AsyncEval() {
  auto task = DoEvalAsync();

  while (task.Status() == wf::AsyncStatus::Started) {
    std::cout << "Waiting...\n";
    Sleep(30);
  }
  std::cout << "Done\n";
  WINML_EXPECT_NO_THROW(task.get());
}

//! Scenario6: use BindInputWithProperties - BitmapBounds, BitmapPixelFormat
// apparently this scenario is cut for rs5. - not cut, just rewprked. move props
// to the image value when that is checked in.
static void Scenario6BindWithProperties() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the default device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the input and the output buffers by name
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    SoftwareBitmap sb = SoftwareBitmap(BitmapPixelFormat::Bgra8, 224, 224);
    auto videoFrame = VideoFrame::CreateWithSoftwareBitmap(sb);
    auto imageValue = ImageFeatureValue::CreateFromVideoFrame(videoFrame);

    PropertySet propertySet;

    // make a BitmapBounds
    BitmapBounds bounds;
    bounds.X = 0;
    bounds.Y = 0;
    bounds.Height = 100;
    bounds.Width = 100;

    auto bitmapsBoundsProperty = wf::PropertyValue::CreateUInt32Array({bounds.X, bounds.Y, bounds.Width, bounds.Height});
    // insert it in the property set
    propertySet.Insert(L"BitmapBounds", bitmapsBoundsProperty);

    // make a BitmapPixelFormat
    BitmapPixelFormat bitmapPixelFormat = BitmapPixelFormat::Bgra8;
    // translate it to an int so it can be used as a PropertyValue;
    int intFromBitmapPixelFormat = static_cast<int>(bitmapPixelFormat);
    auto bitmapPixelFormatProperty = wf::PropertyValue::CreateInt32(intFromBitmapPixelFormat);
    // insert it in the property set
    propertySet.Insert(L"BitmapPixelFormat", bitmapPixelFormatProperty);

    // bind with properties
    WINML_EXPECT_NO_THROW(binding.Bind(input.Name(), imageValue, propertySet));
  }
  // run eval
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

//! Scenario7: run eval without creating a binding object
static void Scenario7EvalWithNoBind() {
  auto map = winrt::single_threaded_map<winrt::hstring, wf::IInspectable>();

  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the default device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  // enumerate feature descriptors and create features (but don't bind them)
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(input);
    map.Insert(input.Name(), featureValue);
  }
  // run eval
  WINML_EXPECT_NO_THROW(session.EvaluateFeaturesAsync(map, L"").get());
}

//! Scenario8: choose which device to run the model on - PreferredDeviceType, PreferredDevicePerformance, SetDeviceFromSurface, SetDevice
// create a session on the default device
static void Scenario8SetDeviceSampleDefault() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  LearningModelDevice anyDevice(LearningModelDeviceKind::Default);
  LearningModelSession anySession(model, anyDevice);
}

// create a session on the CPU device
static void Scenario8SetDeviceSampleCPU() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  LearningModelDevice cpuDevice(LearningModelDeviceKind::Cpu);
  LearningModelSession cpuSession(model, cpuDevice);
}

// create a session on the default DML device
static void Scenario8SetDeviceSampleDefaultDirectX() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  LearningModelDevice dmlDeviceDefault(LearningModelDeviceKind::DirectX);
  LearningModelSession dmlSessionDefault(model, dmlDeviceDefault);
}

// create a session on the DML device that provides best power
static void Scenario8SetDeviceSampleMinPower() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  LearningModelDevice dmlDeviceMinPower(LearningModelDeviceKind::DirectXMinPower);
  LearningModelSession dmlSessionMinPower(model, dmlDeviceMinPower);
}

// create a session on the DML device that provides best perf
static void Scenario8SetDeviceSampleMaxPerf() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  LearningModelDevice dmlDeviceMaxPerf(LearningModelDeviceKind::DirectXHighPerformance);
  LearningModelSession dmlSessionMaxPerf(model, dmlDeviceMaxPerf);
}

// create a session on the same device my camera is on
static void Scenario8SetDeviceSampleMyCameraDevice() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  auto devices = winrt::Windows::Devices::Enumeration::DeviceInformation::FindAllAsync(winrt::Windows::Devices::Enumeration::DeviceClass::VideoCapture).get();
  winrt::hstring deviceId;
  if (devices.Size() > 0) {
    auto device = devices.GetAt(0);
    deviceId = device.Id();
    auto deviceName = device.Name();
    auto enabled = device.IsEnabled();
    std::cout << "Found device " << deviceName.c_str() << ", enabled = " << enabled << "\n";
    wm::Capture::MediaCapture captureManager;
    wm::Capture::MediaCaptureInitializationSettings settings;
    settings.VideoDeviceId(deviceId);
    captureManager.InitializeAsync(settings).get();
    auto mediaCaptureSettings = captureManager.MediaCaptureSettings();
    auto direct3D11Device = mediaCaptureSettings.Direct3D11Device();
    LearningModelDevice dmlDeviceCamera = LearningModelDevice::CreateFromDirect3D11Device(direct3D11Device);
    LearningModelSession dmlSessionCamera(model, dmlDeviceCamera);
  } else {
    WINML_SKIP_TEST("Test skipped because video capture device is missing");
  }
}

// create a device from D3D11 Device
static void Scenario8SetDeviceSampleD3D11Device() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  winrt::com_ptr<ID3D11Device> pD3D11Device = nullptr;
  winrt::com_ptr<ID3D11DeviceContext> pContext = nullptr;
  D3D_FEATURE_LEVEL fl;
  HRESULT result = D3D11CreateDevice(
      nullptr, D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
      D3D11_SDK_VERSION, pD3D11Device.put(), &fl, pContext.put());
  if (FAILED(result)) {
    WINML_SKIP_TEST("Test skipped because d3d11 device is missing");
  }

  // get dxgiDevice from d3ddevice
  winrt::com_ptr<IDXGIDevice> pDxgiDevice;
  pD3D11Device.get()->QueryInterface<IDXGIDevice>(pDxgiDevice.put());

  winrt::com_ptr<::IInspectable> pInspectable;
  CreateDirect3D11DeviceFromDXGIDevice(pDxgiDevice.get(), pInspectable.put());

  LearningModelDevice device = LearningModelDevice::CreateFromDirect3D11Device(
      pInspectable.as<wgdx::Direct3D11::IDirect3DDevice>());
  LearningModelSession session(model, device);
}

// create a session on the a specific dx device that I chose some other way , note we have to use native interop here and pass a cmd queue
static void Scenario8SetDeviceSampleCustomCommandQueue() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  winrt::com_ptr<ID3D12Device> pD3D12Device = nullptr;
  CommonDeviceHelpers::AdapterEnumerationSupport support;
  if (FAILED(CommonDeviceHelpers::GetAdapterEnumerationSupport(&support))) {
    WINML_LOG_ERROR("Unable to load DXGI or DXCore");
    return;
  }
  HRESULT result = S_OK;
  if (support.has_dxgi) {
    WINML_EXPECT_NO_THROW(result = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(pD3D12Device.put())));
  }
#ifdef ENABLE_DXCORE
  if (support.has_dxgi == false) {
    winrt::com_ptr<IDXCoreAdapterFactory> spFactory;
    DXCoreCreateAdapterFactory(IID_PPV_ARGS(spFactory.put()));
    const GUID gpuFilter[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS};
    winrt::com_ptr<IDXCoreAdapterList> spAdapterList;
    spFactory->CreateAdapterList(1, gpuFilter, IID_PPV_ARGS(spAdapterList.put()));
    winrt::com_ptr<IDXCoreAdapter> spAdapter;
    WINML_EXPECT_NO_THROW(spAdapterList->GetAdapter(0, IID_PPV_ARGS(spAdapter.put())));
    ::IUnknown* pAdapter = spAdapter.get();
    WINML_EXPECT_NO_THROW(result = D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(pD3D12Device.put())));
  }
#endif

  if (FAILED(result)) {
    WINML_SKIP_TEST("Test skipped because d3d12 device is missing");
    return;
  }
  winrt::com_ptr<ID3D12CommandQueue> dxQueue = nullptr;
  D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
  commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  pD3D12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&dxQueue));
  auto factory = winrt::get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();

  winrt::com_ptr<::IUnknown> spUnk;
  factory->CreateFromD3D12CommandQueue(dxQueue.get(), spUnk.put());

  auto dmlDeviceCustom = spUnk.as<LearningModelDevice>();
  LearningModelSession dmlSessionCustom(model, dmlDeviceCustom);
}

//pass a Tensor in as an input GPU
static void Scenario9LoadBindEvalInputTensorGPU() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"fns-candy.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);

  winrt::com_ptr<ID3D12Device> pD3D12Device;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), pD3D12Device.put_void()));
  winrt::com_ptr<ID3D12CommandQueue> dxQueue;
  D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
  commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  pD3D12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue), dxQueue.put_void());
  auto devicefactory = winrt::get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
  auto tensorfactory = winrt::get_activation_factory<TensorFloat, ITensorStaticsNative>();

  winrt::com_ptr<::IUnknown> spUnk;
  WINML_EXPECT_NO_THROW(devicefactory->CreateFromD3D12CommandQueue(dxQueue.get(), spUnk.put()));

  LearningModelDevice dmlDeviceCustom = nullptr;
  WINML_EXPECT_NO_THROW(spUnk.as(dmlDeviceCustom));
  LearningModelSession dmlSessionCustom = nullptr;
  WINML_EXPECT_NO_THROW(dmlSessionCustom = LearningModelSession(model, dmlDeviceCustom));

  LearningModelBinding modelBinding(dmlSessionCustom);

  UINT64 bufferbytesize = 720 * 720 * 3 * sizeof(float);
  D3D12_HEAP_PROPERTIES heapProperties = {
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
      D3D12_MEMORY_POOL_UNKNOWN,
      0,
      0};
  D3D12_RESOURCE_DESC resourceDesc = {
      D3D12_RESOURCE_DIMENSION_BUFFER,
      0,
      bufferbytesize,
      1,
      1,
      1,
      DXGI_FORMAT_UNKNOWN,
      {1, 0},
      D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

  winrt::com_ptr<ID3D12Resource> pGPUResource = nullptr;
  pD3D12Device->CreateCommittedResource(
      &heapProperties,
      D3D12_HEAP_FLAG_NONE,
      &resourceDesc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      __uuidof(ID3D12Resource),
      pGPUResource.put_void());
  winrt::com_ptr<::IUnknown> spUnkTensor;
  TensorFloat input1imagetensor(nullptr);
  __int64 shape[4] = {1, 3, 720, 720};
  tensorfactory->CreateFromD3D12Resource(pGPUResource.get(), shape, 4, spUnkTensor.put());
  spUnkTensor.try_as(input1imagetensor);

  auto feature = model.InputFeatures().First();
  WINML_EXPECT_NO_THROW(modelBinding.Bind(feature.Current().Name(), input1imagetensor));

  auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto outputtensorshape = outputtensordescriptor.Shape();
  VideoFrame outputimage(
      BitmapPixelFormat::Rgba8,
      static_cast<int32_t>(outputtensorshape.GetAt(3)),
      static_cast<int32_t>(outputtensorshape.GetAt(2)));
  ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);

  WINML_EXPECT_NO_THROW(modelBinding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

  // Testing GetAsD3D12Resource
  winrt::com_ptr<ID3D12Resource> pReturnedResource;
  input1imagetensor.as<ITensorNative>()->GetD3D12Resource(pReturnedResource.put());
  WINML_EXPECT_EQUAL(pReturnedResource.get(), pGPUResource.get());

  // Evaluate the model
  winrt::hstring correlationId;
  dmlSessionCustom.EvaluateAsync(modelBinding, correlationId).get();
}

static void Scenario13SingleModelOnCPUandGPU() {
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  LearningModelSession cpuSession(model, LearningModelDevice(LearningModelDeviceKind::Cpu));
  LearningModelSession gpuSession(model, LearningModelDevice(LearningModelDeviceKind::DirectX));

  LearningModelBinding cpuBinding(cpuSession);
  LearningModelBinding gpuBinding(gpuSession);
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto cpuFeatureValue = FeatureValueFromFeatureValueDescriptor(input);
    cpuBinding.Bind(input.Name(), cpuFeatureValue);

    auto gpuFeatureValue = FeatureValueFromFeatureValueDescriptor(input);
    gpuBinding.Bind(input.Name(), gpuFeatureValue);
  }

  auto cpuTask = cpuSession.EvaluateAsync(cpuBinding, L"cpu");
  auto gpuTask = gpuSession.EvaluateAsync(gpuBinding, L"gpu");

  WINML_EXPECT_NO_THROW(cpuTask.get());
  WINML_EXPECT_NO_THROW(gpuTask.get());
}

// Validates when binding input image with free dimensions, the binding step is executed correctly.
static void Scenario11FreeDimensionsTensor() {
  std::wstring filePath = FileHelpers::GetModulePath() + L"free_dimensional_image_input.onnx";
  // load a model with expected input size: -1 x -1
  auto model = LearningModel::LoadFromFilePath(filePath);
  auto session = LearningModelSession(model);
  auto binding = LearningModelBinding(session);

  VideoFrame inputImage(BitmapPixelFormat::Rgba8, 1000, 1000);
  ImageFeatureValue inputimagetensor = ImageFeatureValue::CreateFromVideoFrame(inputImage);

  auto feature = model.InputFeatures().First();
  binding.Bind(feature.Current().Name(), inputimagetensor);
  feature.MoveNext();
  binding.Bind(feature.Current().Name(), inputimagetensor);

  session.Evaluate(binding, L"");
}

static void Scenario11FreeDimensionsImage() {
  std::wstring filePath = FileHelpers::GetModulePath() + L"free_dimensional_imageDes.onnx";
  // load a model with expected input size: -1 x -1
  auto model = LearningModel::LoadFromFilePath(filePath);
  auto session = LearningModelSession(model);
  auto binding = LearningModelBinding(session);

  VideoFrame inputImage(BitmapPixelFormat::Bgra8, 1000, 1000);
  ImageFeatureValue inputimagetensor = ImageFeatureValue::CreateFromVideoFrame(inputImage);

  auto feature = model.InputFeatures().First();
  ImageFeatureDescriptor imageDescriptor = nullptr;
  feature.Current().as(imageDescriptor);
  binding.Bind(feature.Current().Name(), inputimagetensor);

  feature.MoveNext();
  feature.Current().as(imageDescriptor);
  binding.Bind(feature.Current().Name(), inputimagetensor);

  session.Evaluate(binding, L"");
}

struct SwapChainEntry {
  LearningModelSession session;
  LearningModelBinding binding;
  wf::IAsyncOperation<LearningModelEvaluationResult> activetask;
  SwapChainEntry() : session(nullptr), binding(nullptr), activetask(nullptr) {}
};
void SubmitEval(LearningModel model, SwapChainEntry* sessionBindings, int swapchaindex) {
  if (sessionBindings[swapchaindex].activetask != nullptr) {
    //make sure the previously submitted work for this swapchain index is complete before reusing resources
    sessionBindings[swapchaindex].activetask.get();
  }
  // bind the input and the output buffers by name
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(input);
    // set an actual buffer here. we're using uninitialized data for simplicity.
    sessionBindings[swapchaindex].binding.Bind(input.Name(), featureValue);
  }
  // submit an eval and wait for it to finish submitting work
  sessionBindings[swapchaindex].activetask = sessionBindings[swapchaindex].session.EvaluateAsync(sessionBindings[swapchaindex].binding, L"0");
  // return without waiting for the submit to finish, setup the completion handler
}

//Scenario14:Load single model, run it mutliple times on a single gpu device using a fast swapchain pattern
static void Scenario14RunModelSwapchain() {
  const int swapchainentrycount = 3;
  SwapChainEntry sessionBindings[swapchainentrycount];

  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on gpu1
  LearningModelDevice dmlDevice = LearningModelDevice(LearningModelDeviceKind::DirectX);
  // create the swapchain style bindings to cycle through
  for (int i = 0; i < swapchainentrycount; i++) {
    sessionBindings[i].session = LearningModelSession(model, dmlDevice);
    sessionBindings[i].binding = LearningModelBinding(sessionBindings[i].session);
  }

  //submit 10 evaluations to 3 swapchain entries
  int swapchaindex = 0;
  for (int i = 0; i < 10; i++) {
    swapchaindex = swapchaindex % swapchainentrycount;
    SubmitEval(model, sessionBindings, (swapchaindex)++);
  }

  //wait for all work to be completed
  for (int i = 0; i < swapchainentrycount; i++) {
    if (sessionBindings[i].activetask != nullptr) {
      //make sure the previously submitted work for this swapchain index is compolete before resuing resources
      sessionBindings[i].activetask.get();
    }
  }
}
static void LoadBindEval_CustomOperator_CPU(const wchar_t* fileName) {
  auto customOperatorProvider = winrt::make<CustomOperatorProvider>();
  auto provider = customOperatorProvider.as<ILearningModelOperatorProvider>();

  LearningModel model = LearningModel::LoadFromFilePath(fileName, provider);
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  LearningModelBinding bindings(session);

  auto inputShape = std::vector<int64_t>{5};
  auto inputData = std::vector<float>{-50.f, -25.f, 0.f, 25.f, 50.f};
  auto inputValue =
      TensorFloat::CreateFromIterable(
          inputShape,
          winrt::single_threaded_vector<float>(std::move(inputData)).GetView());
  WINML_EXPECT_NO_THROW(bindings.Bind(L"X", inputValue));

  auto outputValue = TensorFloat::Create();
  WINML_EXPECT_NO_THROW(bindings.Bind(L"Y", outputValue));

  winrt::hstring correlationId;
  WINML_EXPECT_NO_THROW(session.Evaluate(bindings, correlationId));

  auto buffer = outputValue.GetAsVectorView();
  WINML_EXPECT_TRUE(buffer != nullptr);
}

//! Scenario17 : Control the dev diagnostics features of WinML Tracing
static void Scenario17DevDiagnostics() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the default device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the input and the output buffers by name
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(input);
    // set an actual buffer here. we're using uninitialized data for simplicity.
    binding.Bind(input.Name(), featureValue);
  }
  session.EvaluationProperties().Insert(L"EnableDebugOutput", nullptr);
  // run eval
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

/**
 * Custom Operator Tests are labeled as GPU tests because the DML code is interlaced with the custom op code
 * even though CPU custom ops shouldn't be dependent on GPU functionality.
 * These should be reclassed to ScenarioCppWinrt once the DML code is decoupled from the custom op code.
**/
// create a session that loads a model with a branch new operator, register the custom operator, and load/bind/eval
static void Scenario20aLoadBindEvalCustomOperatorCPU() {
  std::wstring filePath = FileHelpers::GetModulePath() + L"noisy_relu.onnx";
  LoadBindEval_CustomOperator_CPU(filePath.c_str());
}

// create a session that loads a model with an overridden operator, register the replacement custom operator, and load/bind/eval
static void Scenario20bLoadBindEvalReplacementCustomOperatorCPU() {
  std::wstring filePath = FileHelpers::GetModulePath() + L"relu.onnx";
  LoadBindEval_CustomOperator_CPU(filePath.c_str());
}

//! Scenario21: Load two models, set them up to run chained after one another on the same gpu hardware device
static void Scenario21RunModel2ChainZ() {
  // load a model, TODO: get a model that has an image descriptor
  std::wstring filePath = FileHelpers::GetModulePath() + L"fns-candy.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create both session on the default gpu
  LearningModelSession session1(model, LearningModelDevice(LearningModelDeviceKind::DirectX));
  LearningModelSession session2(model, LearningModelDevice(LearningModelDeviceKind::DirectX));
  // create both binding sets
  LearningModelBinding binding1(session1);
  LearningModelBinding binding2(session2);
  // get the input descriptor
  auto input = model.InputFeatures().GetAt(0);
  // load a SoftwareBitmap
  auto sb = FileHelpers::GetSoftwareBitmapFromFile(FileHelpers::GetModulePath() + L"fish_720.png");
  auto videoFrame = VideoFrame::CreateWithSoftwareBitmap(sb);
  // bind it
  binding1.Bind(input.Name(), videoFrame);
  // get the output descriptor
  auto output = model.OutputFeatures().GetAt(0);
  // create an empty output tensor since we don't want the first model to detensorize into an image.

  std::vector<int64_t> shape = {1, 3, 720, 720};
  auto outputValue = TensorFloat::Create(shape);  //   FeatureValueFromFeatureValueDescriptor(input, nullptr);
                                                  // now bind the(empty) output so we have a marker to chain with
  binding1.Bind(output.Name(), outputValue);
  // and leave the output unbound on the second model, we will fetch it later
  // run both models async
  WINML_EXPECT_NO_THROW(session1.EvaluateAsync(binding1, L""));

  // now bind that output to the next models input
  binding2.Bind(input.Name(), outputValue);

  //eval the second model
  auto session2AsyncOp = session2.EvaluateAsync(binding2, L"");

  // now get the output don't wait, queue up the next model
  auto finalOutput = session2AsyncOp.get().Outputs().First().Current().Value();
}

bool VerifyHelper(ImageFeatureValue actual, ImageFeatureValue expected) {
  auto softwareBitmapActual = actual.VideoFrame().SoftwareBitmap();
  auto softwareBitmapExpected = expected.VideoFrame().SoftwareBitmap();
  WINML_EXPECT_EQUAL(softwareBitmapActual.PixelHeight(), softwareBitmapExpected.PixelHeight());
  WINML_EXPECT_EQUAL(softwareBitmapActual.PixelWidth(), softwareBitmapExpected.PixelWidth());
  WINML_EXPECT_EQUAL(softwareBitmapActual.BitmapPixelFormat(), softwareBitmapExpected.BitmapPixelFormat());

  // 4 means 4 channels
  uint32_t size = 4 * softwareBitmapActual.PixelHeight() * softwareBitmapActual.PixelWidth();

  ws::Streams::Buffer actualOutputBuffer(size);
  ws::Streams::Buffer expectedOutputBuffer(size);

  softwareBitmapActual.CopyToBuffer(actualOutputBuffer);
  softwareBitmapExpected.CopyToBuffer(expectedOutputBuffer);

  byte* actualBytes;
  actualOutputBuffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>()->Buffer(&actualBytes);
  byte* expectedBytes;
  expectedOutputBuffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>()->Buffer(&expectedBytes);

  byte* pActualByte = actualBytes;
  byte* pExpectedByte = expectedBytes;

  // hard code, might need to be modified later.
  const float cMaxErrorRate = 0.06f;
  byte epsilon = 20;
  UINT errors = 0;
  for (uint32_t i = 0; i < size; i++, pActualByte++, pExpectedByte++) {
    // Only the check the first three channels, which are (B, G, R)
    if((i + 1) % 4 == 0) continue;
    auto diff = std::abs(*pActualByte - *pExpectedByte);
    if (diff > epsilon) {
      errors++;
    }
  }
  std::cout << "total errors is " << errors << "/" << size << ", errors rate is " << (float)errors / size << "\n";

  return ((float)errors / size < cMaxErrorRate);
}

static void Scenario22ImageBindingAsCPUTensor() {
  std::wstring modulePath = FileHelpers::GetModulePath();
  std::wstring inputImagePath = modulePath + L"fish_720.png";
  std::wstring bmImagePath = modulePath + L"bm_fish_720.jpg";
  std::wstring modelPath = modulePath + L"fns-candy.onnx";

  auto device = LearningModelDevice(LearningModelDeviceKind::Default);
  auto model = LearningModel::LoadFromFilePath(modelPath);
  auto session = LearningModelSession(model, device);
  auto binding = LearningModelBinding(session);

  SoftwareBitmap softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(inputImagePath);
  softwareBitmap = SoftwareBitmap::Convert(softwareBitmap, BitmapPixelFormat::Bgra8);

  // Put softwareBitmap into buffer
  BYTE* pData = nullptr;
  UINT32 size = 0;
  wgi::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(wgi::BitmapBufferAccessMode::Read));
  wf::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
  auto spByteAccess = reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
  spByteAccess->GetBuffer(&pData, &size);

  std::vector<int64_t> shape = {1, 3, softwareBitmap.PixelHeight(), softwareBitmap.PixelWidth()};
  float* pCPUTensor;
  uint32_t uCapacity;
  TensorFloat tf = TensorFloat::Create(shape);
  winrt::com_ptr<ITensorNative> itn = tf.as<ITensorNative>();
  itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUTensor), &uCapacity);

  uint32_t height = softwareBitmap.PixelHeight();
  uint32_t width = softwareBitmap.PixelWidth();
  for (UINT32 i = 0; i < size - 2; i += 4) {
    // loop condition is i < size - 2 to avoid potential for extending past the memory buffer
    UINT32 pixelInd = i / 4;
    pCPUTensor[pixelInd] = (float)pData[i];
    pCPUTensor[(height * width) + pixelInd] = (float)pData[i + 1];
    pCPUTensor[(height * width * 2) + pixelInd] = (float)pData[i + 2];
  }

  // Bind input
  binding.Bind(model.InputFeatures().First().Current().Name(), tf);

  // Bind output
  auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto outputtensorshape = outputtensordescriptor.Shape();
  VideoFrame outputimage(
      BitmapPixelFormat::Bgra8,
      static_cast<int32_t>(outputtensorshape.GetAt(3)),
      static_cast<int32_t>(outputtensorshape.GetAt(2)));
  ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);
  WINML_EXPECT_NO_THROW(binding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

  // Evaluate the model
  winrt::hstring correlationId;
  WINML_EXPECT_NO_THROW(session.EvaluateAsync(binding, correlationId).get());

  // Verify the output by comparing with the benchmark image
  SoftwareBitmap bm_softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(bmImagePath);
  bm_softwareBitmap = SoftwareBitmap::Convert(bm_softwareBitmap, BitmapPixelFormat::Bgra8);
  VideoFrame bm_videoFrame = VideoFrame::CreateWithSoftwareBitmap(bm_softwareBitmap);
  ImageFeatureValue bm_imagevalue = ImageFeatureValue::CreateFromVideoFrame(bm_videoFrame);
  WINML_EXPECT_TRUE(VerifyHelper(bm_imagevalue, outputTensor));

  // check the output video frame object by saving output image to disk
  std::wstring outputDataImageFileName = L"out_cpu_tensor_fish_720.jpg";
  StorageFolder currentfolder = StorageFolder::GetFolderFromPathAsync(modulePath).get();
  StorageFile outimagefile = currentfolder.CreateFileAsync(outputDataImageFileName, CreationCollisionOption::ReplaceExisting).get();
  IRandomAccessStream writestream = outimagefile.OpenAsync(FileAccessMode::ReadWrite).get();
  BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), writestream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(outputimage.SoftwareBitmap());
  encoder.FlushAsync().get();
}

static void Scenario22ImageBindingAsGPUTensor() {
  std::wstring modulePath = FileHelpers::GetModulePath();
  std::wstring inputImagePath = modulePath + L"fish_720.png";
  std::wstring bmImagePath = modulePath + L"bm_fish_720.jpg";
  std::wstring modelPath = modulePath + L"fns-candy.onnx";
  std::wstring outputDataImageFileName = L"out_gpu_tensor_fish_720.jpg";

  SoftwareBitmap softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(inputImagePath);
  softwareBitmap = SoftwareBitmap::Convert(softwareBitmap, BitmapPixelFormat::Bgra8);

  // Put softwareBitmap into cpu buffer
  BYTE* pData = nullptr;
  UINT32 size = 0;
  wgi::BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(wgi::BitmapBufferAccessMode::Read));
  wf::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
  auto spByteAccess = reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
  spByteAccess->GetBuffer(&pData, &size);

  std::vector<int64_t> shape = {1, 3, softwareBitmap.PixelHeight(), softwareBitmap.PixelWidth()};
  FLOAT* pCPUTensor;
  uint32_t uCapacity;

  // CPU tensorization
  TensorFloat tf = TensorFloat::Create(shape);
  winrt::com_ptr<ITensorNative> itn = tf.as<ITensorNative>();
  itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUTensor), &uCapacity);

  uint32_t height = softwareBitmap.PixelHeight();
  uint32_t width = softwareBitmap.PixelWidth();
  for (UINT32 i = 0; i < size - 2; i += 4) {
    // loop condition is i < size - 2 to avoid potential for extending past the memory buffer
    UINT32 pixelInd = i / 4;
    pCPUTensor[pixelInd] = (FLOAT)pData[i];
    pCPUTensor[(height * width) + pixelInd] = (FLOAT)pData[i + 1];
    pCPUTensor[(height * width * 2) + pixelInd] = (FLOAT)pData[i + 2];
  }

  // create the d3d device.
  winrt::com_ptr<ID3D12Device> pD3D12Device = nullptr;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&pD3D12Device)));

  // create the command queue.
  winrt::com_ptr<ID3D12CommandQueue> dxQueue = nullptr;
  D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
  commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  pD3D12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&dxQueue));
  auto devicefactory = winrt::get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
  auto tensorfactory = winrt::get_activation_factory<TensorFloat, ITensorStaticsNative>();
  winrt::com_ptr<::IUnknown> spUnk;
  devicefactory->CreateFromD3D12CommandQueue(dxQueue.get(), spUnk.put());

  LearningModel model(nullptr);
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(modelPath));
  LearningModelDevice dmlDeviceCustom = nullptr;
  WINML_EXPECT_NO_THROW(spUnk.as(dmlDeviceCustom));
  LearningModelSession dmlSessionCustom = nullptr;
  WINML_EXPECT_NO_THROW(dmlSessionCustom = LearningModelSession(model, dmlDeviceCustom));
  LearningModelBinding modelBinding = nullptr;
  WINML_EXPECT_NO_THROW(modelBinding = LearningModelBinding(dmlSessionCustom));

  // Create ID3D12GraphicsCommandList and Allocator
  D3D12_COMMAND_LIST_TYPE queuetype = dxQueue->GetDesc().Type;
  winrt::com_ptr<ID3D12CommandAllocator> alloctor;
  winrt::com_ptr<ID3D12GraphicsCommandList> cmdList;

  pD3D12Device->CreateCommandAllocator(
      queuetype,
      winrt::guid_of<ID3D12CommandAllocator>(),
      alloctor.put_void());

  pD3D12Device->CreateCommandList(
      0,
      queuetype,
      alloctor.get(),
      nullptr,
      winrt::guid_of<ID3D12CommandList>(),
      cmdList.put_void());

  // Create Committed Resource
  // 3 is number of channels we use. R G B without alpha.
  UINT64 bufferbytesize = 3 * sizeof(float) * softwareBitmap.PixelWidth() * softwareBitmap.PixelHeight();
  D3D12_HEAP_PROPERTIES heapProperties = {
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
      D3D12_MEMORY_POOL_UNKNOWN,
      0,
      0};
  D3D12_RESOURCE_DESC resourceDesc = {
      D3D12_RESOURCE_DIMENSION_BUFFER,
      0,
      bufferbytesize,
      1,
      1,
      1,
      DXGI_FORMAT_UNKNOWN,
      {1, 0},
      D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

  winrt::com_ptr<ID3D12Resource> pGPUResource = nullptr;
  winrt::com_ptr<ID3D12Resource> imageUploadHeap;
  pD3D12Device->CreateCommittedResource(
      &heapProperties,
      D3D12_HEAP_FLAG_NONE,
      &resourceDesc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      __uuidof(ID3D12Resource),
      pGPUResource.put_void());

  // Create the GPU upload buffer.
  CD3DX12_HEAP_PROPERTIES props(D3D12_HEAP_TYPE_UPLOAD);
  auto buffer = CD3DX12_RESOURCE_DESC::Buffer(bufferbytesize);
  WINML_EXPECT_NO_THROW(pD3D12Device->CreateCommittedResource(
      &props,
      D3D12_HEAP_FLAG_NONE,
      &buffer,
      D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr,
      __uuidof(ID3D12Resource),
      imageUploadHeap.put_void()));

  // Copy from Cpu to GPU
  D3D12_SUBRESOURCE_DATA CPUData = {};
  CPUData.pData = reinterpret_cast<BYTE*>(pCPUTensor);
  CPUData.RowPitch = static_cast<LONG_PTR>(bufferbytesize);
  CPUData.SlicePitch = static_cast<LONG_PTR>(bufferbytesize);
  UpdateSubresources(cmdList.get(), pGPUResource.get(), imageUploadHeap.get(), 0, 0, 1, &CPUData);

  // Close the command list and execute it to begin the initial GPU setup.
  WINML_EXPECT_NO_THROW(cmdList->Close());
  ID3D12CommandList* ppCommandLists[] = {cmdList.get()};
  dxQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

  // GPU tensorize
  winrt::com_ptr<::IUnknown> spUnkTensor;
  TensorFloat input1imagetensor(nullptr);
  __int64 shapes[4] = {1, 3, softwareBitmap.PixelWidth(), softwareBitmap.PixelHeight()};
  tensorfactory->CreateFromD3D12Resource(pGPUResource.get(), shapes, 4, spUnkTensor.put());
  spUnkTensor.try_as(input1imagetensor);

  auto feature = model.InputFeatures().First();
  WINML_EXPECT_NO_THROW(modelBinding.Bind(feature.Current().Name(), input1imagetensor));

  auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto outputtensorshape = outputtensordescriptor.Shape();
  VideoFrame outputimage(
      BitmapPixelFormat::Rgba8,
      static_cast<int32_t>(outputtensorshape.GetAt(3)),
      static_cast<int32_t>(outputtensorshape.GetAt(2)));
  ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);

  WINML_EXPECT_NO_THROW(modelBinding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

  // Evaluate the model
  winrt::hstring correlationId;
  dmlSessionCustom.EvaluateAsync(modelBinding, correlationId).get();

  // Verify the output by comparing with the benchmark image
  SoftwareBitmap bm_softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(bmImagePath);
  bm_softwareBitmap = SoftwareBitmap::Convert(bm_softwareBitmap, BitmapPixelFormat::Rgba8);
  VideoFrame bm_videoFrame = VideoFrame::CreateWithSoftwareBitmap(bm_softwareBitmap);
  ImageFeatureValue bm_imagevalue = ImageFeatureValue::CreateFromVideoFrame(bm_videoFrame);
  WINML_EXPECT_TRUE(VerifyHelper(bm_imagevalue, outputTensor));

  //check the output video frame object
  StorageFolder currentfolder = StorageFolder::GetFolderFromPathAsync(modulePath).get();
  StorageFile outimagefile = currentfolder.CreateFileAsync(outputDataImageFileName, CreationCollisionOption::ReplaceExisting).get();
  IRandomAccessStream writestream = outimagefile.OpenAsync(FileAccessMode::ReadWrite).get();
  BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), writestream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(outputimage.SoftwareBitmap());
  encoder.FlushAsync().get();
}

static void Scenario23NominalPixelRange() {
  std::wstring modulePath = FileHelpers::GetModulePath();
  std::wstring inputImagePath = modulePath + L"1080.jpg";

  // The following models have single op "add", with different metadata
  std::vector<std::wstring> modelPaths = {
    // Normalized_0_1 and image output
    modulePath + L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_0_1.onnx",
    // Normalized_1_1 and image output
    modulePath + L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_1_1.onnx"
  };

  for (uint32_t model_i = 0; model_i < modelPaths.size(); model_i++) {
    // load model and create session
    auto model = LearningModel::LoadFromFilePath(modelPaths[model_i]);
    auto session = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::DirectX));
    auto binding = LearningModelBinding(session);

    SoftwareBitmap softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(inputImagePath);
    auto videoFrame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
    auto imageValue = ImageFeatureValue::CreateFromVideoFrame(videoFrame);

    // Create Zero tensor
    auto inputShape = std::vector<int64_t>{ 1, 3, 1080, 1920 };
    auto inputData = std::vector<float>(3 * 1080 * 1920, 0);
    auto zeroValue =
      TensorFloat::CreateFromIterable(
        inputShape,
        winrt::single_threaded_vector<float>(std::move(inputData)).GetView());
    // bind inputs
    binding.Bind(L"input_39", imageValue);
    binding.Bind(L"input_40", zeroValue);

    VideoFrame outputimage(BitmapPixelFormat::Bgra8, 1920, 1080);
    ImageFeatureValue outputIfv = ImageFeatureValue::CreateFromVideoFrame(outputimage);
    binding.Bind(L"add_3", outputIfv);

    winrt::hstring correlationId;
    session.EvaluateAsync(binding, correlationId).get();

    WINML_EXPECT_TRUE(VerifyHelper(imageValue, outputIfv));
  }
}

static void QuantizedModels() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"onnxzoo_lotus_inception_v1-dq.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the default device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::Default));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the input and the output buffers by name
  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(input);
    // set an actual buffer here. we're using uninitialized data for simplicity.
    binding.Bind(input.Name(), featureValue);
  }
  // run eval
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, filePath));
}

static void MsftQuantizedModels() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"coreml_Resnet50_ImageNet-dq.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::DirectX));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the input and the output buffers by name

  std::wstring fullImagePath = FileHelpers::GetModulePath() + L"kitten_224.png";
  StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
  IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();

  auto inputs = model.InputFeatures();
  for (auto&& input : inputs) {
    auto featureValue = FeatureValueFromFeatureValueDescriptor(input, softwareBitmap);
    // set an actual buffer here. we're using uninitialized data for simplicity.
    binding.Bind(input.Name(), featureValue);
  }
  // run eval
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, filePath));
}

static void SyncVsAsync() {
  // create model, device and session
  LearningModel model = nullptr;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(FileHelpers::GetModulePath() + L"fns-candy.onnx"));

  LearningModelSession session = nullptr;
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::DirectX)));

  // create the binding
  LearningModelBinding modelBinding(session);

  // bind the input
  std::wstring fullImagePath = FileHelpers::GetModulePath() + L"fish_720.png";
  StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
  IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
  VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

  auto imagetensor = ImageFeatureValue::CreateFromVideoFrame(frame);
  auto inputFeatureDescriptor = model.InputFeatures().First();
  WINML_EXPECT_NO_THROW(modelBinding.Bind(inputFeatureDescriptor.Current().Name(), imagetensor));

  UINT N = 20;

  auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto outputtensorshape = outputtensordescriptor.Shape();
  VideoFrame outputimage(
      BitmapPixelFormat::Rgba8,
      static_cast<int32_t>(outputtensorshape.GetAt(3)),
      static_cast<int32_t>(outputtensorshape.GetAt(2)));
  ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);
  WINML_EXPECT_NO_THROW(modelBinding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

  // evaluate N times synchronously and time it
  auto startSync = std::chrono::high_resolution_clock::now();
  for (UINT i = 0; i < N; i++) {
    session.Evaluate(modelBinding, L"");
  }
  auto syncTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startSync);
  std::cout << "Synchronous time for " << N << " evaluations: " << syncTime.count() << " milliseconds\n";

  // evaluate N times Asynchronously and time it
  std::vector<wf::IAsyncOperation<LearningModelEvaluationResult>> tasks;
  std::vector<LearningModelBinding> bindings(N, nullptr);

  for (size_t i = 0; i < bindings.size(); i++) {
    bindings[i] = LearningModelBinding(session);
    bindings[i].Bind(inputFeatureDescriptor.Current().Name(), imagetensor);
    bindings[i].Bind(
        model.OutputFeatures().First().Current().Name(),
        VideoFrame(BitmapPixelFormat::Rgba8,
                   static_cast<int32_t>(outputtensorshape.GetAt(3)),
                   static_cast<int32_t>(outputtensorshape.GetAt(2))));
  }

  auto startAsync = std::chrono::high_resolution_clock::now();
  for (UINT i = 0; i < N; i++) {
    tasks.emplace_back(session.EvaluateAsync(bindings[i], L""));
  }
  // wait for them all to complete
  for (auto&& task : tasks) {
    task.get();
  }
  auto asyncTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startAsync);
  std::cout << "Asynchronous time for " << N << " evaluations: " << asyncTime.count() << " milliseconds\n";
}

static void CustomCommandQueueWithFence() {
  static const wchar_t* const modelFileName = L"fns-candy.onnx";
  static const wchar_t* const inputDataImageFileName = L"fish_720.png";

  winrt::com_ptr<ID3D12Device> d3d12Device;
  WINML_EXPECT_HRESULT_SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), d3d12Device.put_void()));

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

  winrt::com_ptr<ID3D12CommandQueue> queue;
  WINML_EXPECT_HRESULT_SUCCEEDED(d3d12Device->CreateCommandQueue(&queueDesc, __uuidof(ID3D12CommandQueue), queue.put_void()));

  winrt::com_ptr<ID3D12Fence> fence;
  WINML_EXPECT_HRESULT_SUCCEEDED(d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), fence.put_void()));

  auto devicefactory = winrt::get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();

  winrt::com_ptr<::IUnknown> learningModelDeviceUnknown;
  WINML_EXPECT_HRESULT_SUCCEEDED(devicefactory->CreateFromD3D12CommandQueue(queue.get(), learningModelDeviceUnknown.put()));

  LearningModelDevice device = nullptr;
  WINML_EXPECT_NO_THROW(learningModelDeviceUnknown.as(device));

  std::wstring modulePath = FileHelpers::GetModulePath();

  // WinML model creation
  std::wstring fullModelPath = modulePath + modelFileName;
  LearningModel model(nullptr);
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
  LearningModelSession modelSession = nullptr;
  WINML_EXPECT_NO_THROW(modelSession = LearningModelSession(model, device));
  LearningModelBinding modelBinding = nullptr;
  WINML_EXPECT_NO_THROW(modelBinding = LearningModelBinding(modelSession));

  std::wstring fullImagePath = modulePath + inputDataImageFileName;

  StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
  IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
  VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
  ImageFeatureValue input1imagetensor = ImageFeatureValue::CreateFromVideoFrame(frame);

  auto feature = model.InputFeatures().First();
  WINML_EXPECT_NO_THROW(modelBinding.Bind(feature.Current().Name(), input1imagetensor));

  auto outputtensordescriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto outputtensorshape = outputtensordescriptor.Shape();
  VideoFrame outputimage(
      BitmapPixelFormat::Rgba8,
      static_cast<int32_t>(outputtensorshape.GetAt(3)),
      static_cast<int32_t>(outputtensorshape.GetAt(2)));
  ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);

  WINML_EXPECT_NO_THROW(modelBinding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

  // Block the queue on the fence, evaluate the model, then queue a signal. The model evaluation should not complete
  // until after the wait is unblocked, and the signal should not complete until model evaluation does. This can
  // only be true if WinML executes the workload on the supplied queue (instead of using its own).

  WINML_EXPECT_HRESULT_SUCCEEDED(queue->Wait(fence.get(), 1));

  WINML_EXPECT_HRESULT_SUCCEEDED(queue->Signal(fence.get(), 2));

  winrt::hstring correlationId;
  wf::IAsyncOperation<LearningModelEvaluationResult> asyncOp;
  WINML_EXPECT_NO_THROW(asyncOp = modelSession.EvaluateAsync(modelBinding, correlationId));

  Sleep(1000);  // Give the model a chance to run (which it shouldn't if everything is working correctly)

  // Because we haven't unblocked the wait yet, model evaluation must not have completed (nor the fence signal)
  WINML_EXPECT_NOT_EQUAL(asyncOp.Status(), wf::AsyncStatus::Completed);
  WINML_EXPECT_EQUAL(fence->GetCompletedValue(), 0);

  // Unblock the queue
  WINML_EXPECT_HRESULT_SUCCEEDED(fence->Signal(1));

  // Wait for model evaluation to complete
  asyncOp.get();

  // The fence must be signaled by now (because model evaluation has completed)
  WINML_EXPECT_EQUAL(fence->GetCompletedValue(), 2);
}

static void ReuseVideoFrame() {
  std::wstring modulePath = FileHelpers::GetModulePath();
  std::wstring inputImagePath = modulePath + L"fish_720.png";
  std::wstring bmImagePath = modulePath + L"bm_fish_720.jpg";
  std::wstring modelPath = modulePath + L"fns-candy.onnx";

  std::vector<LearningModelDeviceKind> deviceKinds = {LearningModelDeviceKind::Cpu, LearningModelDeviceKind::DirectX};
  std::vector<std::string> videoFrameSources;
  CommonDeviceHelpers::AdapterEnumerationSupport support;
  CommonDeviceHelpers::GetAdapterEnumerationSupport(&support);
  if (support.has_dxgi) {
    videoFrameSources = {"SoftwareBitmap", "Direct3DSurface"};
  } else {
    videoFrameSources = {"SoftwareBitmap"};
  }

  for (auto deviceKind : deviceKinds) {
    auto device = LearningModelDevice(deviceKind);
    auto model = LearningModel::LoadFromFilePath(modelPath);
    auto session = LearningModelSession(model, device);
    auto binding = LearningModelBinding(session);
    for (auto videoFrameSource : videoFrameSources) {
      VideoFrame reuseVideoFrame = nullptr;
      if (videoFrameSource == "SoftwareBitmap") {
        reuseVideoFrame = VideoFrame::CreateWithSoftwareBitmap(SoftwareBitmap(BitmapPixelFormat::Bgra8, 720, 720));
      } else {
        reuseVideoFrame = VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized, 720, 720);
      }
      for (uint32_t i = 0; i < 3; ++i) {
        SoftwareBitmap softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(inputImagePath);
        VideoFrame videoFrame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
        // reuse video frame
        videoFrame.CopyToAsync(reuseVideoFrame).get();

        // bind input
        binding.Bind(model.InputFeatures().First().Current().Name(), reuseVideoFrame);

        // bind output
        VideoFrame outputimage(BitmapPixelFormat::Bgra8, 720, 720);
        ImageFeatureValue outputTensor = ImageFeatureValue::CreateFromVideoFrame(outputimage);
        WINML_EXPECT_NO_THROW(binding.Bind(model.OutputFeatures().First().Current().Name(), outputTensor));

        // evaluate
        winrt::hstring correlationId;
        WINML_EXPECT_NO_THROW(session.EvaluateAsync(binding, correlationId).get());

        // verify result
        SoftwareBitmap bm_softwareBitmap = FileHelpers::GetSoftwareBitmapFromFile(bmImagePath);
        bm_softwareBitmap = SoftwareBitmap::Convert(bm_softwareBitmap, BitmapPixelFormat::Bgra8);
        VideoFrame bm_videoFrame = VideoFrame::CreateWithSoftwareBitmap(bm_softwareBitmap);
        ImageFeatureValue bm_imagevalue = ImageFeatureValue::CreateFromVideoFrame(bm_videoFrame);
        WINML_EXPECT_TRUE(VerifyHelper(bm_imagevalue, outputTensor));
      }
    }
  }
}
static void EncryptedStream() {
  // get a stream
  std::wstring path = FileHelpers::GetModulePath() + L"model.onnx";
  auto storageFile = StorageFile::GetFileFromPathAsync(path).get();
  auto fileBuffer = ws::FileIO::ReadBufferAsync(storageFile).get();

  // encrypt
  auto algorithmName = winrt::Windows::Security::Cryptography::Core::SymmetricAlgorithmNames::AesCbcPkcs7();
  auto algorithm = winrt::Windows::Security::Cryptography::Core::SymmetricKeyAlgorithmProvider::OpenAlgorithm(algorithmName);
  uint32_t keyLength = 32;
  auto keyBuffer = winrt::Windows::Security::Cryptography::CryptographicBuffer::GenerateRandom(keyLength);
  auto key = algorithm.CreateSymmetricKey(keyBuffer);
  auto iv = winrt::Windows::Security::Cryptography::CryptographicBuffer::GenerateRandom(algorithm.BlockLength());
  auto encryptedBuffer = winrt::Windows::Security::Cryptography::Core::CryptographicEngine::Encrypt(key, fileBuffer, iv);

  // verify loading the encrypted stream fails appropriately.
  auto encryptedStream = InMemoryRandomAccessStream();
  encryptedStream.WriteAsync(encryptedBuffer).get();
  WINML_EXPECT_THROW_SPECIFIC(LearningModel::LoadFromStream(RandomAccessStreamReference::CreateFromStream(encryptedStream)),
                              winrt::hresult_error,
                              [](const winrt::hresult_error& e) -> bool {
                                return e.code() == E_INVALIDARG;
                              });

  // now decrypt
  auto decryptedBuffer = winrt::Windows::Security::Cryptography::Core::CryptographicEngine::Decrypt(key, encryptedBuffer, iv);
  auto decryptedStream = InMemoryRandomAccessStream();
  decryptedStream.WriteAsync(decryptedBuffer).get();

  // load!
  LearningModel model = nullptr;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromStream(RandomAccessStreamReference::CreateFromStream(decryptedStream)));
  LearningModelSession session = nullptr;
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model));
}

static void DeviceLostRecovery() {
  // load a model
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a session on the DirectX device
  LearningModelSession session(model, LearningModelDevice(LearningModelDeviceKind::DirectX));
  // create a binding set
  LearningModelBinding binding(session);
  // bind the inputs
  BindFeatures(binding, model.InputFeatures());

  // force device lost here
  {
    winrt::com_ptr<ID3D12Device5> d3d12Device;
    D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device5), d3d12Device.put_void());
    d3d12Device->RemoveDevice();
  }

  // evaluate should fail
  try {
    session.Evaluate(binding, L"");
    WINML_LOG_ERROR("Evaluate should fail after removing the device");
  } catch (...) {
  }

  // remove all references to the device by resetting the session and binding.
  session = nullptr;
  binding = nullptr;

  // create new session and binding and try again!
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::DirectX)));
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindFeatures(binding, model.InputFeatures());
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

static void D2DInterop() {
  // load a model (model.onnx == squeezenet[1,3,224,224])
  std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
  LearningModel model = LearningModel::LoadFromFilePath(filePath);
  // create a dx12 device
  winrt::com_ptr<ID3D12Device1> device = nullptr;
  WINML_EXPECT_HRESULT_SUCCEEDED(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device1), device.put_void()));
  // now create a command queue from it
  winrt::com_ptr<ID3D12CommandQueue> commandQueue = nullptr;
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  WINML_EXPECT_HRESULT_SUCCEEDED(device->CreateCommandQueue(&queueDesc, winrt::guid_of<ID3D12CommandQueue>(), commandQueue.put_void()));
  // create a winml learning device based on that dx12 queue
  auto factory = winrt::get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
  winrt::com_ptr<::IUnknown> spUnk;
  WINML_EXPECT_HRESULT_SUCCEEDED(factory->CreateFromD3D12CommandQueue(commandQueue.get(), spUnk.put()));
  auto learningDevice = spUnk.as<LearningModelDevice>();
  // create a winml session from that dx device
  LearningModelSession session(model, learningDevice);
  // now lets try and do some XAML/d2d on that same device, first prealloc a VideoFrame
  VideoFrame frame = VideoFrame::CreateAsDirect3D11SurfaceBacked(
      DirectXPixelFormat::B8G8R8A8UIntNormalized,
      224,
      224,
      session.Device().Direct3D11Device());
  // create a D2D factory
  D2D1_FACTORY_OPTIONS options = {};
  winrt::com_ptr<ID2D1Factory> d2dFactory;
  WINML_EXPECT_HRESULT_SUCCEEDED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(ID2D1Factory), &options, d2dFactory.put_void()));
  // grab the dxgi surface back from our video frame
  winrt::com_ptr<IDXGISurface> dxgiSurface;
  winrt::com_ptr<IDirect3DDxgiInterfaceAccess> dxgiInterfaceAccess = frame.Direct3DSurface().as<IDirect3DDxgiInterfaceAccess>();
  WINML_EXPECT_HRESULT_SUCCEEDED(dxgiInterfaceAccess->GetInterface(__uuidof(IDXGISurface), dxgiSurface.put_void()));
  // and try and use our surface to create a render targer
  winrt::com_ptr<ID2D1RenderTarget> renderTarget;
  D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties();
  props.pixelFormat = D2D1::PixelFormat(
      DXGI_FORMAT_B8G8R8A8_UNORM,
      D2D1_ALPHA_MODE_IGNORE);
  WINML_EXPECT_HRESULT_SUCCEEDED(d2dFactory->CreateDxgiSurfaceRenderTarget(
      dxgiSurface.get(),
      props,
      renderTarget.put()));
}

static void BindMultipleCPUBuffersAsInputs(LearningModelDeviceKind kind) {
  std::wstring module_path = FileHelpers::GetModulePath();
  std::wstring model_path = module_path + L"fns-candy.onnx";

  // init session
  auto device = LearningModelDevice(kind);
  auto model = LearningModel::LoadFromFilePath(model_path);
  auto session = LearningModelSession(model, device);
  auto binding = LearningModelBinding(session);

  // Load input
  std::wstring input_image_path = module_path + L"fish_720.png";
  std::wstring image_path = module_path + L"bm_fish_720.jpg";
  auto software_bitmap = FileHelpers::GetSoftwareBitmapFromFile(input_image_path);
  auto bgra8_bitmap = SoftwareBitmap::Convert(software_bitmap, BitmapPixelFormat::Bgra8);

  uint32_t height = bgra8_bitmap.PixelHeight();
  uint32_t width = bgra8_bitmap.PixelWidth();
  uint32_t frame_size = height * width * sizeof(float);

  // Declare raw pointers
  UINT32 size = 0;
  BYTE* data = nullptr;
  float* red_data = nullptr;
  float* green_data = nullptr;
  float* blue_data = nullptr;

  // Get memory buffers
  wgi::BitmapBuffer bitmap(bgra8_bitmap.LockBuffer(wgi::BitmapBufferAccessMode::Read));
  wf::MemoryBuffer red(frame_size);
  wf::MemoryBuffer green(frame_size);
  wf::MemoryBuffer blue(frame_size);

  // Create references
  auto bitmap_reference = bitmap.CreateReference();
  auto red_reference = red.CreateReference();
  auto green_reference = green.CreateReference();
  auto blue_reference = blue.CreateReference();

  // Get byte access objects
  auto bitmap_byteaccess = bitmap_reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
  auto red_byteaccess = red_reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
  auto green_byteaccess = green_reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
  auto blue_byteaccess = blue_reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();

  // Get raw buffers
  bitmap_byteaccess->GetBuffer(&data, &size);
  red_byteaccess->GetBuffer(reinterpret_cast<BYTE**>(&red_data), &frame_size);
  green_byteaccess->GetBuffer(reinterpret_cast<BYTE**>(&green_data), &frame_size);
  blue_byteaccess->GetBuffer(reinterpret_cast<BYTE**>(&blue_data), &frame_size);

  for (UINT32 i = 0; i < size - 2 && i / 4 < frame_size; i += 4) {
    // loop condition is i < size - 2 to avoid potential for extending past the memory buffer
    UINT32 pixelInd = i / 4;
    red_data[pixelInd] = (float)data[i];
    green_data[pixelInd] = (float)data[i + 1];
    blue_data[pixelInd] = (float)data[i + 2];
  }

  auto buffers = winrt::single_threaded_vector<wss::IBuffer>();
  buffers.Append(wss::Buffer::CreateCopyFromMemoryBuffer(red));
  buffers.Append(wss::Buffer::CreateCopyFromMemoryBuffer(green));
  buffers.Append(wss::Buffer::CreateCopyFromMemoryBuffer(blue));
  // second batch
  buffers.Append(wss::Buffer::CreateCopyFromMemoryBuffer(red));
  buffers.Append(wss::Buffer::CreateCopyFromMemoryBuffer(green));
  buffers.Append(wss::Buffer::CreateCopyFromMemoryBuffer(blue));
  
  // Bind input
  binding.Bind(model.InputFeatures().First().Current().Name(), buffers);

  // Bind output
  auto output_descriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto output_shape = output_descriptor.Shape();
  VideoFrame outputimage1(
      BitmapPixelFormat::Bgra8,
      static_cast<int32_t>(output_shape.GetAt(3)),
      static_cast<int32_t>(output_shape.GetAt(2)));
  VideoFrame outputimage2(
      BitmapPixelFormat::Bgra8,
      static_cast<int32_t>(output_shape.GetAt(3)),
      static_cast<int32_t>(output_shape.GetAt(2)));

  auto output_frames = winrt::single_threaded_vector<wm::VideoFrame>();
  output_frames.Append(outputimage1);
  output_frames.Append(outputimage2);
  WINML_EXPECT_NO_THROW(binding.Bind(model.OutputFeatures().First().Current().Name(), output_frames));

  // Evaluate the model
  winrt::hstring correlationId;
  WINML_EXPECT_NO_THROW(session.EvaluateAsync(binding, correlationId).get());

  // Verify the output by comparing with the benchmark image
  SoftwareBitmap benchmark_output_bitmap = FileHelpers::GetSoftwareBitmapFromFile(image_path);
  benchmark_output_bitmap = SoftwareBitmap::Convert(benchmark_output_bitmap, BitmapPixelFormat::Bgra8);
  VideoFrame benchmark_output_frame = VideoFrame::CreateWithSoftwareBitmap(benchmark_output_bitmap);
  ImageFeatureValue benchmark_output_featurevalue = ImageFeatureValue::CreateFromVideoFrame(benchmark_output_frame);
  WINML_EXPECT_TRUE(VerifyHelper(benchmark_output_featurevalue, ImageFeatureValue::CreateFromVideoFrame(outputimage1)));
  WINML_EXPECT_TRUE(VerifyHelper(benchmark_output_featurevalue, ImageFeatureValue::CreateFromVideoFrame(outputimage2)));

  // check the output video frame object by saving output image to disk
  std::wstring output_filename = L"out_cpu_tensor_fish_720.jpg";
  StorageFolder current_folder = StorageFolder::GetFolderFromPathAsync(module_path).get();
  StorageFile output_file = current_folder.CreateFileAsync(output_filename, CreationCollisionOption::ReplaceExisting).get();
  IRandomAccessStream output_stream = output_file.OpenAsync(FileAccessMode::ReadWrite).get();
  BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), output_stream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(outputimage2.SoftwareBitmap());
  encoder.FlushAsync().get();
}

static void BindMultipleCPUBuffersInputsOnCpu() {
  BindMultipleCPUBuffersAsInputs(LearningModelDeviceKind::Cpu);
}

static void BindMultipleCPUBuffersInputsOnGpu() {
  BindMultipleCPUBuffersAsInputs(LearningModelDeviceKind::DirectX);
}

static void BindMultipleCPUBuffersAsOutputs(LearningModelDeviceKind kind) {
  std::wstring modulePath = FileHelpers::GetModulePath();
  std::wstring inputImagePath = modulePath + L"fish_720.png";
  std::wstring bmImagePath = modulePath + L"bm_fish_720.jpg";
  std::wstring modelPath = modulePath + L"fns-candy.onnx";

  auto device = LearningModelDevice(kind);
  auto model = LearningModel::LoadFromFilePath(modelPath);
  auto session = LearningModelSession(model, device);
  auto binding = LearningModelBinding(session);

  auto software_bitmap = FileHelpers::GetSoftwareBitmapFromFile(inputImagePath);
  auto video_frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);

  // Bind input
  binding.Bind(model.InputFeatures().First().Current().Name(), video_frame);

  // Bind output
  uint32_t height = software_bitmap.PixelHeight();
  uint32_t width = software_bitmap.PixelWidth();
  uint32_t channel_frame_size = height * width * sizeof(float);

  wf::MemoryBuffer red(channel_frame_size);
  wf::MemoryBuffer green(channel_frame_size);
  wf::MemoryBuffer blue(channel_frame_size);

  auto output_descriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto output_shape = output_descriptor.Shape();

  auto red_buffer = wss::Buffer::CreateCopyFromMemoryBuffer(red);
  auto green_buffer = wss::Buffer::CreateCopyFromMemoryBuffer(green);
  auto blue_buffer = wss::Buffer::CreateCopyFromMemoryBuffer(blue);

  auto output_frames = winrt::single_threaded_vector<wss::IBuffer>();
  output_frames.Append(red_buffer);
  output_frames.Append(green_buffer);
  output_frames.Append(blue_buffer);
  WINML_EXPECT_NO_THROW(binding.Bind(model.OutputFeatures().First().Current().Name(), output_frames));

  // Evaluate the model
  winrt::hstring correlationId;
  WINML_EXPECT_NO_THROW(session.EvaluateAsync(binding, correlationId).get());

  auto output_bitmap = SoftwareBitmap(BitmapPixelFormat::Bgra8, 720, 720);
  float* red_bytes;
  float* green_bytes;
  float* blue_bytes;
  red_buffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>()->Buffer(reinterpret_cast<byte**>(&red_bytes));
  green_buffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>()->Buffer(reinterpret_cast<byte**>(&green_bytes));
  blue_buffer.try_as<::Windows::Storage::Streams::IBufferByteAccess>()->Buffer(reinterpret_cast<byte**>(&blue_bytes));
  
  // Verify the output by comparing with the benchmark image
  SoftwareBitmap benchmark_bitmap = FileHelpers::GetSoftwareBitmapFromFile(bmImagePath);
  benchmark_bitmap = SoftwareBitmap::Convert(benchmark_bitmap, BitmapPixelFormat::Bgra8);

  BYTE* benchmark_data = nullptr;
  UINT32 benchmark_size = 0;
  wgi::BitmapBuffer benchmark_bitmap_buffer(benchmark_bitmap.LockBuffer(wgi::BitmapBufferAccessMode::Read));
  wf::IMemoryBufferReference benchmark_reference = benchmark_bitmap_buffer.CreateReference();
  auto benchmark_byte_access = benchmark_reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
  benchmark_byte_access->GetBuffer(&benchmark_data, &benchmark_size);
  
  // hard code, might need to be modified later.
  const float cMaxErrorRate = 0.06f;
  byte epsilon = 20;
  UINT errors = 0;
  for (UINT32 i = 0; i < height * width; i ++) {
    if (std::abs(red_bytes[i] - benchmark_data[i * 4]) > epsilon) {
      errors++;
    }
    if (std::abs(green_bytes[i] - benchmark_data[i * 4 + 1]) > epsilon) {
      errors++;
    }
    if (std::abs(blue_bytes[i] - benchmark_data[i * 4 + 2]) > epsilon) {
      errors++;
    }
  }
  auto total_size = height * width * 3;
  std::cout << "total errors is " << errors << "/" << total_size << ", errors rate is " << (float)errors / total_size << "\n";

  WINML_EXPECT_TRUE((float)errors / total_size < cMaxErrorRate);


  // check the output video frame object by saving output image to disk
  std::wstring outputDataImageFileName = L"out_cpu_tensor_fish_720.jpg";
  StorageFolder currentfolder = StorageFolder::GetFolderFromPathAsync(modulePath).get();
  StorageFile outimagefile = currentfolder.CreateFileAsync(outputDataImageFileName, CreationCollisionOption::ReplaceExisting).get();
  IRandomAccessStream writestream = outimagefile.OpenAsync(FileAccessMode::ReadWrite).get();
  BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), writestream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(output_bitmap);
  encoder.FlushAsync().get();
}

static void BindMultipleCPUBuffersOutputsOnCpu() {
  BindMultipleCPUBuffersAsOutputs(LearningModelDeviceKind::Cpu);
}

static void BindMultipleCPUBuffersOutputsOnGpu() {
  BindMultipleCPUBuffersAsOutputs(LearningModelDeviceKind::DirectX);
}

const ScenarioTestsApi& getapi() {
  static ScenarioTestsApi api =
      {
          ScenarioCppWinrtTestsClassSetup,
          Sample1,
          Scenario1LoadBindEvalDefault,
          Scenario2LoadModelFromStream,
          Scenario5AsyncEval,
          Scenario7EvalWithNoBind,
          Scenario8SetDeviceSampleDefault,
          Scenario8SetDeviceSampleCPU,
          Scenario17DevDiagnostics,
          Scenario22ImageBindingAsCPUTensor,
          Scenario23NominalPixelRange,
          QuantizedModels,
          EncryptedStream,
          Scenario3SoftwareBitmapInputBinding,
          Scenario6BindWithProperties,
          Scenario8SetDeviceSampleDefaultDirectX,
          Scenario8SetDeviceSampleMinPower,
          Scenario8SetDeviceSampleMaxPerf,
          Scenario8SetDeviceSampleMyCameraDevice,
          Scenario8SetDeviceSampleCustomCommandQueue,
          Scenario9LoadBindEvalInputTensorGPU,
          Scenario13SingleModelOnCPUandGPU,
          Scenario11FreeDimensionsTensor,
          Scenario11FreeDimensionsImage,
          Scenario14RunModelSwapchain,
          Scenario20aLoadBindEvalCustomOperatorCPU,
          Scenario20bLoadBindEvalReplacementCustomOperatorCPU,
          Scenario21RunModel2ChainZ,
          Scenario22ImageBindingAsGPUTensor,
          MsftQuantizedModels,
          SyncVsAsync,
          CustomCommandQueueWithFence,
          ReuseVideoFrame,
          DeviceLostRecovery,
          Scenario8SetDeviceSampleD3D11Device,
          D2DInterop,
          BindMultipleCPUBuffersInputsOnCpu,
          BindMultipleCPUBuffersInputsOnGpu,
          BindMultipleCPUBuffersOutputsOnCpu,
          BindMultipleCPUBuffersOutputsOnGpu,
      };

  if (SkipGpuTests()) {
    api.Scenario6BindWithProperties = SkipTest;
    api.Scenario8SetDeviceSampleDefaultDirectX = SkipTest;
    api.Scenario8SetDeviceSampleMinPower = SkipTest;
    api.Scenario8SetDeviceSampleMaxPerf = SkipTest;
    api.Scenario8SetDeviceSampleCustomCommandQueue = SkipTest;
    api.Scenario9LoadBindEvalInputTensorGPU = SkipTest;
    api.Scenario13SingleModelOnCPUandGPU = SkipTest;
    api.Scenario11FreeDimensionsTensor = SkipTest;
    api.Scenario11FreeDimensionsImage = SkipTest;
    api.Scenario14RunModelSwapchain = SkipTest;
    api.Scenario20aLoadBindEvalCustomOperatorCPU = SkipTest;
    api.Scenario20bLoadBindEvalReplacementCustomOperatorCPU = SkipTest;
    api.Scenario21RunModel2ChainZ = SkipTest;
    api.Scenario22ImageBindingAsGPUTensor = SkipTest;
    api.Scenario23NominalPixelRange = SkipTest;
    api.MsftQuantizedModels = SkipTest;
    api.SyncVsAsync = SkipTest;
    api.CustomCommandQueueWithFence = SkipTest;
    api.ReuseVideoFrame = SkipTest;
    api.DeviceLostRecovery = SkipTest;
    api.Scenario8SetDeviceSampleD3D11Device = SkipTest;
    api.D2DInterop = SkipTest;
    api.BindMultipleCPUBuffersInputsOnGpu = SkipTest;
    api.BindMultipleCPUBuffersOutputsOnGpu = SkipTest;
  }

  if (RuntimeParameterExists(L"EdgeCore")) {
    api.Scenario8SetDeviceSampleMyCameraDevice = SkipTest;
    api.Scenario8SetDeviceSampleD3D11Device = SkipTest;
    api.D2DInterop = SkipTest;
  }

  if (RuntimeParameterExists(L"noVideoFrameTests")) {
    api.Scenario1LoadBindEvalDefault = SkipTest;
    api.Scenario3SoftwareBitmapInputBinding = SkipTest;
    api.Scenario5AsyncEval = SkipTest;
    api.Scenario6BindWithProperties = SkipTest;
    api.Scenario7EvalWithNoBind = SkipTest;
    api.Scenario9LoadBindEvalInputTensorGPU = SkipTest;
    api.Scenario11FreeDimensionsTensor = SkipTest;
    api.Scenario11FreeDimensionsImage = SkipTest;
    api.Scenario13SingleModelOnCPUandGPU = SkipTest;
    api.Scenario14RunModelSwapchain = SkipTest;
    api.Scenario17DevDiagnostics = SkipTest;
    api.Scenario21RunModel2ChainZ = SkipTest;
    api.Scenario22ImageBindingAsCPUTensor = SkipTest;
    api.Scenario22ImageBindingAsGPUTensor = SkipTest;
    api.Scenario23NominalPixelRange = SkipTest;
    api.CustomCommandQueueWithFence = SkipTest;
    api.ReuseVideoFrame = SkipTest;
    api.D2DInterop = SkipTest;
    api.DeviceLostRecovery = SkipTest;
    api.QuantizedModels = SkipTest;
    api.MsftQuantizedModels = SkipTest;
    api.BindMultipleCPUBuffersInputsOnCpu = SkipTest;
    api.BindMultipleCPUBuffersInputsOnGpu = SkipTest;
    api.BindMultipleCPUBuffersOutputsOnCpu = SkipTest;
    api.BindMultipleCPUBuffersOutputsOnGpu = SkipTest;
  }
  if (RuntimeParameterExists(L"noIDXGIFactory6Tests")) {
    api.Scenario8SetDeviceSampleMinPower = SkipTest;
    api.Scenario8SetDeviceSampleMaxPerf = SkipTest;
  }
  if (RuntimeParameterExists(L"noID3D12Device5Tests")) {
    api.DeviceLostRecovery = SkipTest;
  }
  return api;
}
