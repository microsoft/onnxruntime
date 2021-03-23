// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"
#include "RawApiTestsGpu.h"
#include "RawApiHelpers.h"

#include <d3d11.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <d3d11on12.h>
#include <d3d11_3.h>

namespace ml = Microsoft::AI::MachineLearning;

enum class DeviceType
{
    CPU,
    DirectX,
    D3D11Device,
    D3D12CommandQueue,
    DirectXHighPerformance,
    DirectXMinPower,
    Last
};


ml::learning_model_device CreateDevice(DeviceType deviceType)
{
    switch (deviceType)
    {
    case DeviceType::CPU:
        return ml::learning_model_device();
    case DeviceType::DirectX:
        return ml::gpu::directx_device(ml::gpu::directx_device_kind::directx);
    case DeviceType::DirectXHighPerformance:
        return ml::gpu::directx_device(ml::gpu::directx_device_kind::directx_high_power);
    case DeviceType::DirectXMinPower:
        return ml::gpu::directx_device(ml::gpu::directx_device_kind::directx_min_power);
    case DeviceType::D3D11Device:
    {
        Microsoft::WRL::ComPtr<ID3D11Device> d3d11Device;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> d3d11DeviceContext;
        D3D_FEATURE_LEVEL d3dFeatureLevel;
        auto result = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            nullptr,
            0,
            D3D11_SDK_VERSION,
            d3d11Device.GetAddressOf(),
            &d3dFeatureLevel,
            d3d11DeviceContext.GetAddressOf()
        );
        if (FAILED(result))
        {
            printf("Failed to create d3d11 device");
            exit(3);
        }

        Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
        d3d11Device.Get()->QueryInterface<IDXGIDevice>(dxgiDevice.GetAddressOf());

        Microsoft::WRL::ComPtr<IInspectable> inspectable;
        CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), inspectable.GetAddressOf());

        Microsoft::WRL::ComPtr<ABI::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice> direct3dDevice;
        inspectable.As(&direct3dDevice);

        return ml::gpu::directx_device(direct3dDevice.Get());
    }
    case DeviceType::D3D12CommandQueue:
    {
        Microsoft::WRL::ComPtr<ID3D12Device> d3d12Device;
        auto result = D3D12CreateDevice(
            nullptr,
            D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_12_0,
            __uuidof(ID3D12Device),
            reinterpret_cast<void**>(d3d12Device.GetAddressOf()));
        if (FAILED(result))
        {
            printf("Failed to create d3d12 device");
            exit(3);
        }
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        d3d12Device->CreateCommandQueue(
            &commandQueueDesc,
            __uuidof(ID3D12CommandQueue),
            reinterpret_cast<void**>(queue.GetAddressOf()));

        return ml::gpu::directx_device(queue.Get());
    }
    default:
      return ml::learning_model_device();
    }
}

static void RawApiTestsGpuApiTestsClassSetup() {
  WINML_EXPECT_HRESULT_SUCCEEDED(RoInitialize(RO_INIT_TYPE::RO_INIT_SINGLETHREADED));
}

static void CreateDirectXDevice() {
  WINML_EXPECT_NO_THROW(CreateDevice(DeviceType::DirectX));
}

static void CreateD3D11DeviceDevice() {
  WINML_EXPECT_NO_THROW(CreateDevice(DeviceType::D3D11Device));
}

static void CreateD3D12CommandQueueDevice() {
  WINML_EXPECT_NO_THROW(CreateDevice(DeviceType::D3D12CommandQueue));
}

static void CreateDirectXHighPerformanceDevice() {
  WINML_EXPECT_NO_THROW(CreateDevice(DeviceType::DirectXHighPerformance));
}

static void CreateDirectXMinPowerDevice() {
  WINML_EXPECT_NO_THROW(CreateDevice(DeviceType::DirectXMinPower));
}

static void Evaluate() {
  std::wstring model_path = L"model.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));

  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>(CreateDevice(DeviceType::DirectX)));

  RunOnDevice(*model.get(), *device.get(), InputStrategy::CopyInputs);

  WINML_EXPECT_NO_THROW(model.reset());
}

static void EvaluateNoInputCopy() {
  std::wstring model_path = L"model.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));

  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>(CreateDevice(DeviceType::DirectX)));

  RunOnDevice(*model.get(), *device.get(), InputStrategy::BindAsReference);

  WINML_EXPECT_NO_THROW(model.reset());
}

static void EvaluateManyBuffers() {
  std::wstring model_path = L"model.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));

  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>(CreateDevice(DeviceType::DirectX)));

  RunOnDevice(*model.get(), *device.get(), InputStrategy::BindWithMultipleReferences);

  WINML_EXPECT_NO_THROW(model.reset());
}

const RawApiTestsGpuApi& getapi() {
  static RawApiTestsGpuApi api = {
      RawApiTestsGpuApiTestsClassSetup,
      CreateDirectXDevice,
      CreateD3D11DeviceDevice,
      CreateD3D12CommandQueueDevice,
      CreateDirectXHighPerformanceDevice,
      CreateDirectXMinPowerDevice,
      Evaluate,
      EvaluateNoInputCopy,
      EvaluateManyBuffers
  };

  if (SkipGpuTests()) {
    api.CreateDirectXDevice = SkipTest;
    api.CreateD3D11DeviceDevice = SkipTest;
    api.CreateD3D12CommandQueueDevice = SkipTest;
    api.CreateDirectXHighPerformanceDevice = SkipTest;
    api.CreateDirectXMinPowerDevice = SkipTest;
    api.Evaluate = SkipTest;
    api.EvaluateNoInputCopy = SkipTest;
    api.EvaluateManyBuffers = SkipTest;
  }
  return api;
}