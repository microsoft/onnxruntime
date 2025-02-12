// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#define INITGUID
#include <guiddef.h>
#include <directx/dxcore.h>
#undef INITGUID

#include "directx/d3d12.h"

#include <DirectML.h>
#ifndef _GAMING_XBOX
#include <dxgi1_4.h>
#endif

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include <wil/wrl.h>
#include <wil/result.h>

#include "core/providers/dml/dml_provider_factory.h"
#include "core/providers/dml/dml_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/allocator_adapters.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"
#include "DmlExecutionProvider/src/ErrorHandling.h"
#include "DmlExecutionProvider/src/GraphicsUnknownHelper.h"
#include "DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/platform/env.h"
#include "core/providers/dml/dml_session_options_config_keys.h"
#include "core/providers/dml/DmlExecutionProvider/src/ExecutionContext.h"

namespace onnxruntime {

static bool ConfigValueIsTrue(std::string&& config_value)
{
  std::transform(config_value.begin(), config_value.end(), config_value.begin(), [](char ch) { return static_cast<char>(std::tolower(ch)); });
  return config_value == "true" || config_value == "1";
}

struct DMLProviderFactory : IExecutionProviderFactory {
  DMLProviderFactory(
    const ConfigOptions& config_options,
    IDMLDevice* dml_device,
    ID3D12CommandQueue* cmd_queue,
    bool disable_metacommands,
    bool python_api
    )
    : dml_device_(dml_device),
      cmd_queue_(cmd_queue),
      metacommands_enabled_(!disable_metacommands),
      python_api_(python_api) {
    graph_capture_enabled_ = ConfigValueIsTrue(config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableGraphCapture, "0"));
    cpu_sync_spinning_enabled_ = ConfigValueIsTrue(config_options.GetConfigOrDefault(kOrtSessionOptionsConfigEnableCpuSyncSpinning, "0"));
    disable_memory_arena_ = ConfigValueIsTrue(config_options.GetConfigOrDefault(kOrtSessionOptionsConfigDisableMemoryArena, "0"));
  }

  ~DMLProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  void SetMetacommandsEnabled(bool metacommands_enabled);

 private:
  ComPtr<IDMLDevice> dml_device_{};
  ComPtr<ID3D12CommandQueue> cmd_queue_{};
  bool metacommands_enabled_ = true;
  bool graph_capture_enabled_ = false;
  bool cpu_sync_spinning_enabled_ = false;
  bool disable_memory_arena_ = false;
  bool python_api_ = false;
};

std::unique_ptr<IExecutionProvider> DMLProviderFactory::CreateProvider() {
  ComPtr<Dml::ExecutionContext> execution_context;

  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(cmd_queue_->GetDevice(IID_PPV_ARGS(&d3d12_device)));

  if (python_api_) {
    uint32_t execution_context_ptr_size = gsl::narrow_cast<uint32_t>(sizeof(execution_context.GetAddressOf()));

    // First, check if an I/O binding API that was used before this session or another session has already created a queue
    if (FAILED(d3d12_device->GetPrivateData(dml_execution_context_guid, &execution_context_ptr_size, execution_context.GetAddressOf()))) {
      execution_context = wil::MakeOrThrow<Dml::ExecutionContext>(d3d12_device.Get(), dml_device_.Get(), cmd_queue_.Get(), true, true);
      ORT_THROW_IF_FAILED(d3d12_device->SetPrivateDataInterface(dml_execution_context_guid, execution_context.Get()));
    }
  } else {
    execution_context = wil::MakeOrThrow<Dml::ExecutionContext>(d3d12_device.Get(), dml_device_.Get(), cmd_queue_.Get(), cpu_sync_spinning_enabled_, false);
  }

  auto provider = Dml::CreateExecutionProvider(dml_device_.Get(), execution_context.Get(), metacommands_enabled_, graph_capture_enabled_, cpu_sync_spinning_enabled_, disable_memory_arena_);
  return provider;
}

void DMLProviderFactory::SetMetacommandsEnabled(bool metacommands_enabled) {
  metacommands_enabled_ = metacommands_enabled;
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_DML(const ConfigOptions& config_options,
                                                                              IDMLDevice* dml_device,
                                                                              ID3D12CommandQueue* cmd_queue,
                                                                              bool disable_metacommands,
                                                                              bool python_api) {
#ifndef _GAMING_XBOX
  // Validate that the D3D12 devices match between DML and the command queue. This specifically asks for IUnknown in
  // order to be able to compare the pointers for COM object identity.
  ComPtr<IUnknown> d3d12_device_0;
  ComPtr<IUnknown> d3d12_device_1;
  ORT_THROW_IF_FAILED(dml_device->GetParentDevice(IID_PPV_ARGS(&d3d12_device_0)));
  ORT_THROW_IF_FAILED(cmd_queue->GetDevice(IID_PPV_ARGS(&d3d12_device_1)));

  if (d3d12_device_0 != d3d12_device_1) {
    ORT_THROW_HR(E_INVALIDARG);
  }
#endif

  ComPtr<ID3D12Device> d3d12_device;
  ORT_THROW_IF_FAILED(dml_device->GetParentDevice(IID_GRAPHICS_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));
  const Env& env = Env::Default();
  auto luid = d3d12_device->GetAdapterLuid();
  env.GetTelemetryProvider().LogExecutionProviderEvent(&luid);
  return std::make_shared<onnxruntime::DMLProviderFactory>(config_options, dml_device, cmd_queue, disable_metacommands, python_api);
}

void DmlConfigureProviderFactoryMetacommandsEnabled(IExecutionProviderFactory* factory, bool metacommandsEnabled) {
  auto dml_provider_factory = static_cast<DMLProviderFactory*>(factory);
  dml_provider_factory->SetMetacommandsEnabled(metacommandsEnabled);
}


bool IsSoftwareAdapter(IDXGIAdapter1* adapter) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    // see here for documentation on filtering WARP adapter:
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    auto isBasicRenderDriverVendorId = desc.VendorId == 0x1414;
    auto isBasicRenderDriverDeviceId = desc.DeviceId == 0x8c;
    auto isSoftwareAdapter = desc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;

    return isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
}

static bool IsHardwareAdapter(IDXCoreAdapter* adapter) {
  bool is_hardware = false;
  THROW_IF_FAILED(adapter->GetProperty(
    DXCoreAdapterProperty::IsHardware,
    &is_hardware));
  return is_hardware;
}

static bool IsGPU(IDXCoreAdapter* compute_adapter) {
  // Only considering hardware adapters
  if (!IsHardwareAdapter(compute_adapter)) {
    return false;
  }
  return compute_adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS);
}

static bool IsNPU(IDXCoreAdapter* compute_adapter) {
  // Only considering hardware adapters
  if (!IsHardwareAdapter(compute_adapter)) {
    return false;
  }
  return !(compute_adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS));
}

enum class DeviceType { GPU, NPU, BadDevice };

static DeviceType FilterAdapterTypeQuery(IDXCoreAdapter* adapter, OrtDmlDeviceFilter filter) {
  auto allow_gpus = (filter & OrtDmlDeviceFilter::Gpu) == OrtDmlDeviceFilter::Gpu;
  if (IsGPU(adapter) && allow_gpus) {
    return DeviceType::GPU;
  }

  auto allow_npus = (filter & OrtDmlDeviceFilter::Npu) == OrtDmlDeviceFilter::Npu;
  if (IsNPU(adapter) && allow_npus) {
    return DeviceType::NPU;
  }

  return DeviceType::BadDevice;
}

// Struct for holding each adapter
struct AdapterInfo {
  ComPtr<IDXCoreAdapter> Adapter;
  DeviceType Type; // GPU or NPU
};

static ComPtr<IDXCoreAdapterList> EnumerateDXCoreAdapters(IDXCoreAdapterFactory* adapter_factory) {
  ComPtr<IDXCoreAdapterList> adapter_list;

  // TODO: use_dxcore_workload_enumeration should be determined by QI
  // When DXCore APIs are available QI for relevant enumeration interfaces
  constexpr bool use_dxcore_workload_enumeration = false;
  if (!use_dxcore_workload_enumeration) {
    ORT_THROW_IF_FAILED(
      adapter_factory->CreateAdapterList(1,
        &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML,
        adapter_list.GetAddressOf()));

    if (adapter_list->GetAdapterCount() == 0)
    {
        ORT_THROW_IF_FAILED(adapter_factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, adapter_list.GetAddressOf()));
    }
  }

  return adapter_list;
}

static void SortDXCoreAdaptersByPreference(
  IDXCoreAdapterList* adapter_list,
  OrtDmlPerformancePreference preference) {
  if (adapter_list->GetAdapterCount() <= 1) {
    return;
  }

  // DML prefers the HighPerformance adapter by default
  std::array<DXCoreAdapterPreference, 1> adapter_list_preferences = {
    DXCoreAdapterPreference::HighPerformance
  };

  // If callers specify minimum power change the DXCore sort policy
  // NOTE DXCoreAdapterPrefernce does not apply to mixed adapter lists - only to GPU lists
  if (preference == OrtDmlPerformancePreference::MinimumPower) {
    adapter_list_preferences[0] = DXCoreAdapterPreference::MinimumPower;
  }

  ORT_THROW_IF_FAILED(adapter_list->Sort(
    static_cast<uint32_t>(adapter_list_preferences.size()),
    adapter_list_preferences.data()));
}

static std::vector<AdapterInfo> FilterDXCoreAdapters(
  IDXCoreAdapterList* adapter_list,
  OrtDmlDeviceFilter filter) {
  auto adapter_infos = std::vector<AdapterInfo>();
  const uint32_t count = adapter_list->GetAdapterCount();
  for (uint32_t i = 0; i < count; ++i) {
    ComPtr<IDXCoreAdapter> candidate_adapter;
    ORT_THROW_IF_FAILED(adapter_list->GetAdapter(i, candidate_adapter.GetAddressOf()));

    // Add the adapters that are valid based on the device filter (GPU, NPU, or Both)
    auto adapter_type = FilterAdapterTypeQuery(candidate_adapter.Get(), filter);
    if (adapter_type != DeviceType::BadDevice) {
      adapter_infos.push_back(AdapterInfo{candidate_adapter, adapter_type});
    }
  }

  return adapter_infos;
}

static void SortHeterogenousDXCoreAdapterList(
  std::vector<AdapterInfo>& adapter_infos,
  OrtDmlDeviceFilter filter,
  OrtDmlPerformancePreference preference) {
  if (adapter_infos.size() <= 1) {
    return;
  }

  // When considering both GPUs and NPUs sort them by performance preference
  // of Default (Gpus first), HighPerformance (GPUs first), or LowPower (NPUs first)
  auto keep_npus = (filter & OrtDmlDeviceFilter::Npu) == OrtDmlDeviceFilter::Npu;
  auto only_npus =  filter == OrtDmlDeviceFilter::Npu;
  if (!keep_npus || only_npus) {
    return;
  }

  struct SortingPolicy {
    // default is false because GPUs are considered higher priority in
    // a mixed adapter environment
    bool npus_first_ = false;

    SortingPolicy(bool npus_first = false) : npus_first_(npus_first) { }

    bool operator()(const AdapterInfo& a, const AdapterInfo& b) {
      return npus_first_ ? a.Type < b.Type : a.Type > b.Type;
    }
  };

  auto npus_first = (preference == OrtDmlPerformancePreference::MinimumPower);
  auto policy = SortingPolicy(npus_first);
  std::sort(adapter_infos.begin(), adapter_infos.end(), policy);
}

std::shared_ptr<IExecutionProviderFactory> DMLProviderFactoryCreator::CreateFromDeviceOptions(
    const ConfigOptions& config_options,
    const OrtDmlDeviceOptions* device_options,
    bool disable_metacommands,
    bool python_api) {
  auto default_device_options = OrtDmlDeviceOptions { Default, Gpu };
  if (device_options == nullptr) {
    device_options = &default_device_options;
  }

  OrtDmlPerformancePreference preference = device_options->Preference;
  OrtDmlDeviceFilter filter = device_options->Filter;

  // Create DXCore Adapter Factory
  ComPtr<IDXCoreAdapterFactory> adapter_factory;
  ORT_THROW_IF_FAILED(::DXCoreCreateAdapterFactory(adapter_factory.GetAddressOf()));

  // Get all DML compatible DXCore adapters
  ComPtr<IDXCoreAdapterList> adapter_list;
  adapter_list = EnumerateDXCoreAdapters(adapter_factory.Get());

  if (adapter_list->GetAdapterCount() == 0) {
    ORT_THROW("No GPUs or NPUs detected.");
  }

  // Sort the adapter list to honor DXCore hardware ordering
  SortDXCoreAdaptersByPreference(adapter_list.Get(), preference);

  // TODO: use_dxcore_workload_enumeration should be determined by QI
  // When DXCore APIs are available QI for relevant enumeration interfaces
  constexpr bool use_dxcore_workload_enumeration = false;

  std::vector<AdapterInfo> adapter_infos;
  if (!use_dxcore_workload_enumeration) {
    // Filter all DXCore adapters to hardware type specified by the device filter
    adapter_infos = FilterDXCoreAdapters(adapter_list.Get(), filter);
    if (adapter_infos.size() == 0) {
      ORT_THROW("No devices detected that match the filter criteria.");
    }
  }

  // DXCore Sort ignores NPUs. When both GPUs and NPUs are present, manually sort them.
  SortHeterogenousDXCoreAdapterList(adapter_infos, filter, preference);

  // Extract just the adapters
  auto adapters = std::vector<ComPtr<IDXCoreAdapter>>(adapter_infos.size());
  std::transform(
    adapter_infos.begin(), adapter_infos.end(),
    adapters.begin(),
    [](auto& a){ return a.Adapter; });

  return onnxruntime::DMLProviderFactoryCreator::CreateFromAdapterList(config_options, std::move(adapters), disable_metacommands, python_api);
}

static std::optional<OrtDmlPerformancePreference> ParsePerformancePreference(const ProviderOptions& provider_options) {
  static const std::string PerformancePreference = "performance_preference";
  static const std::string Default = "default";
  static const std::string HighPerformance = "high_performance";
  static const std::string MinimumPower = "minimum_power";

  auto preference_it = provider_options.find(PerformancePreference);
  if (preference_it != provider_options.end()) {
    if (preference_it->second == Default) {
      return OrtDmlPerformancePreference::Default;
    }

    if (preference_it->second == HighPerformance) {
      return OrtDmlPerformancePreference::HighPerformance;
    }

    if (preference_it->second == MinimumPower) {
      return OrtDmlPerformancePreference::MinimumPower;
    }

    ORT_THROW("Invalid PerformancePreference provided for DirectML EP device selection.");
  }

  return {};
}

static std::optional<OrtDmlDeviceFilter> ParseFilter(const ProviderOptions& provider_options) {
  static const std::string Filter = "device_filter";
  static const std::string Any = "any";
  static const std::string Gpu = "gpu";
  static const std::string Npu = "npu";

  auto preference_it = provider_options.find(Filter);
  if (preference_it != provider_options.end()) {
    if (preference_it->second == Gpu) {
      return OrtDmlDeviceFilter::Gpu;
    }

    if (preference_it->second == Any) {
      return OrtDmlDeviceFilter::Any;
    }
    if (preference_it->second == Npu) {
      return OrtDmlDeviceFilter::Npu;
    }

    ORT_THROW("Invalid Filter provided for DirectML EP device selection.");
  }

  return {};
}

static std::optional<int> ParseDeviceId(const ProviderOptions& provider_options) {
  static const std::string DeviceId = "device_id";

  auto preference_it = provider_options.find(DeviceId);
  if (preference_it != provider_options.end()) {
     if (!preference_it->second.empty()) {
       return std::stoi(preference_it->second);
     }
  }

  return {};
}

static bool ParseBoolean(const ProviderOptions& provider_options, const std::string& key) {
  auto preference_it = provider_options.find(key);
  if (preference_it != provider_options.end() && !preference_it->second.empty()) {
      if (preference_it->second == "True" || preference_it->second == "true") {
        return true;
      } else if (preference_it->second == "False" || preference_it->second == "false") {
        return false;
      } else {
        ORT_THROW("[ERROR] [DirectML] The value for the key '" + key + "' should be 'True' or 'False'. Default value is 'False'.\n");
      }
  }

  return false;
}

std::shared_ptr<IExecutionProviderFactory> DMLProviderFactoryCreator::CreateFromProviderOptions(
    const ConfigOptions& config_options,
    const ProviderOptions& provider_options,
    bool python_api) {

  bool disable_metacommands = ParseBoolean(provider_options, "disable_metacommands");
  bool skip_software_device_check = false;
  auto device_id = ParseDeviceId(provider_options);

  if (device_id.has_value())
  {
    return onnxruntime::DMLProviderFactoryCreator::Create(config_options, device_id.value(), skip_software_device_check, disable_metacommands, python_api);
  }

  auto preference = ParsePerformancePreference(provider_options);
  auto filter = ParseFilter(provider_options);

  // If no preference/filters are specified then create with default preference/filters.
  if (!preference.has_value() && !filter.has_value()) {
    return onnxruntime::DMLProviderFactoryCreator::CreateFromDeviceOptions(config_options, nullptr, disable_metacommands, python_api);
  }

  if (!preference.has_value()) {
    preference = OrtDmlPerformancePreference::Default;
  }

  if (!filter.has_value()) {
    filter = OrtDmlDeviceFilter::Gpu;
  }

  OrtDmlDeviceOptions device_options;
  device_options.Preference = preference.value();
  device_options.Filter = filter.value();
  return onnxruntime::DMLProviderFactoryCreator::CreateFromDeviceOptions(config_options, &device_options, disable_metacommands, python_api);
}

Microsoft::WRL::ComPtr<ID3D12Device> DMLProviderFactoryCreator::CreateD3D12Device(
  int device_id,
  bool skip_software_device_check) {
#ifdef _GAMING_XBOX
    ComPtr<ID3D12Device> d3d12_device;
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
    params.Version = D3D12_SDK_VERSION;
    params.GraphicsCommandQueueRingSizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
    params.GraphicsScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
    params.ComputeScratchMemorySizeBytes = static_cast<UINT>(D3D12XBOX_DEFAULT_SIZE_BYTES);
    ORT_THROW_IF_FAILED(D3D12XboxCreateDevice(nullptr, &params, IID_GRAPHICS_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));
#else
    ComPtr<IDXGIFactory4> dxgi_factory;
    ORT_THROW_IF_FAILED(CreateDXGIFactory2(0, IID_GRAPHICS_PPV_ARGS(dxgi_factory.ReleaseAndGetAddressOf())));

    ComPtr<IDXGIAdapter1> adapter;
    ORT_THROW_IF_FAILED(dxgi_factory->EnumAdapters1(device_id, &adapter));

    // Disallow using DML with the software adapter (Microsoft Basic Display Adapter) because CPU evaluations are much
    // faster. Some scenarios though call for EP initialization without this check (as execution will not actually occur
    // anyway) such as operation kernel registry enumeration for documentation purposes.
    if (!skip_software_device_check)
    {
      ORT_THROW_HR_IF(ERROR_GRAPHICS_INVALID_DISPLAY_ADAPTER, IsSoftwareAdapter(adapter.Get()));
    }

    ComPtr<ID3D12Device> d3d12_device;
    ORT_THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_GRAPHICS_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));
#endif

  return d3d12_device;
}

Microsoft::WRL::ComPtr<IDMLDevice> DMLProviderFactoryCreator::CreateDMLDevice(ID3D12Device* d3d12_device) {
  DML_CREATE_DEVICE_FLAGS flags = DML_CREATE_DEVICE_FLAG_NONE;

  // In debug builds, enable the DML debug layer if the D3D12 debug layer is also enabled
#if _DEBUG && !_GAMING_XBOX
  Microsoft::WRL::ComPtr<ID3D12DebugDevice> debug_device;
  (void)d3d12_device->QueryInterface(IID_PPV_ARGS(&debug_device));  // ignore failure
  const bool is_d3d12_debug_layer_enabled = (debug_device != nullptr);

  if (is_d3d12_debug_layer_enabled) {
    flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
  }
#endif

  Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
  ORT_THROW_IF_FAILED(DMLCreateDevice1(
      d3d12_device,
      flags,
      DML_FEATURE_LEVEL_5_0,
      IID_PPV_ARGS(&dml_device)));

  return dml_device;
}

static D3D12_COMMAND_LIST_TYPE CalculateCommandListType(ID3D12Device* d3d12_device) {
  D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels = {};

  D3D_FEATURE_LEVEL feature_levels_list[] = {
  #ifndef _GAMING_XBOX
      D3D_FEATURE_LEVEL_1_0_GENERIC,
  #endif
      D3D_FEATURE_LEVEL_1_0_CORE,
      D3D_FEATURE_LEVEL_11_0,
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_12_0,
      D3D_FEATURE_LEVEL_12_1
  };

  feature_levels.NumFeatureLevels = ARRAYSIZE(feature_levels_list);
  feature_levels.pFeatureLevelsRequested = feature_levels_list;
  ORT_THROW_IF_FAILED(d3d12_device->CheckFeatureSupport(
      D3D12_FEATURE_FEATURE_LEVELS,
      &feature_levels,
      sizeof(feature_levels)
      ));

  auto use_compute_command_list = (feature_levels.MaxSupportedFeatureLevel <= D3D_FEATURE_LEVEL_1_0_CORE);

  if (use_compute_command_list)
  {
    return D3D12_COMMAND_LIST_TYPE_COMPUTE;
  }

  return D3D12_COMMAND_LIST_TYPE_DIRECT;
}

std::shared_ptr<IExecutionProviderFactory> CreateDMLDeviceAndProviderFactory(
  const ConfigOptions& config_options,
  ID3D12Device* d3d12_device,
  bool disable_metacommands,
  bool python_api = false) {
  D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
  cmd_queue_desc.Type = CalculateCommandListType(d3d12_device);
  cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

  ComPtr<ID3D12CommandQueue> cmd_queue;
  ORT_THROW_IF_FAILED(d3d12_device->CreateCommandQueue(&cmd_queue_desc, IID_GRAPHICS_PPV_ARGS(cmd_queue.ReleaseAndGetAddressOf())));

  ComPtr<IDMLDevice> dml_device;
  if (python_api) {
    uint32_t dml_device_ptr_size = gsl::narrow_cast<uint32_t>(sizeof(dml_device.GetAddressOf()));

    if (FAILED(d3d12_device->GetPrivateData(dml_device_guid, &dml_device_ptr_size, dml_device.GetAddressOf()))) {
      dml_device = onnxruntime::DMLProviderFactoryCreator::CreateDMLDevice(d3d12_device);
      ORT_THROW_IF_FAILED(d3d12_device->SetPrivateDataInterface(dml_device_guid, dml_device.Get()));
    }
  } else {
    dml_device = onnxruntime::DMLProviderFactoryCreator::CreateDMLDevice(d3d12_device);
  }

  return CreateExecutionProviderFactory_DML(config_options, dml_device.Get(), cmd_queue.Get(), disable_metacommands, python_api);
}

std::shared_ptr<IExecutionProviderFactory> DMLProviderFactoryCreator::Create(
    const ConfigOptions& config_options,
    int device_id,
    bool skip_software_device_check,
    bool disable_metacommands,
    bool python_api) {
  ComPtr<ID3D12Device> d3d12_device = CreateD3D12Device(device_id, skip_software_device_check);
  return CreateDMLDeviceAndProviderFactory(config_options, d3d12_device.Get(), disable_metacommands, python_api);
}

std::shared_ptr<IExecutionProviderFactory> DMLProviderFactoryCreator::CreateFromAdapterList(
    const ConfigOptions& config_options,
    std::vector<ComPtr<IDXCoreAdapter>>&& adapters,
    bool disable_metacommands,
    bool python_api) {
  // Choose the first device from the list since it's the highest priority
  auto adapter = adapters[0];

  auto feature_level = D3D_FEATURE_LEVEL_11_0;
  if (IsNPU(adapter.Get())) {
    feature_level = D3D_FEATURE_LEVEL_1_0_GENERIC;
  }

  // Create D3D12 Device from DXCore Adapter
  ComPtr<ID3D12Device> d3d12_device;
  if (feature_level == D3D_FEATURE_LEVEL_1_0_GENERIC) {
      // Attempt to create a D3D_FEATURE_LEVEL_1_0_CORE device first, in case the device supports this
      // feature level and the D3D runtime does not support D3D_FEATURE_LEVEL_1_0_GENERIC
      HRESULT hrUnused = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_1_0_CORE, IID_GRAPHICS_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf()));
  }

  if (!d3d12_device) {
    ORT_THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), feature_level, IID_GRAPHICS_PPV_ARGS(d3d12_device.ReleaseAndGetAddressOf())));
  }

  return CreateDMLDeviceAndProviderFactory(config_options, d3d12_device.Get(), disable_metacommands, python_api);
}

}  // namespace onnxruntime

// [[deprecated]]
// This export should be deprecated.
// The OrtSessionOptionsAppendExecutionProvider_DML export on the OrtDmlApi should be used instead.
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id) {
API_IMPL_BEGIN
  options->provider_factories.push_back(onnxruntime::DMLProviderFactoryCreator::Create(options->value.config_options, device_id, false, false, false));
API_IMPL_END
  return nullptr;
}

// [[deprecated]]
// This export should be deprecated.
// The OrtSessionOptionsAppendExecutionProvider_DML1 export on the OrtDmlApi should be used instead.
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProviderEx_DML, _In_ OrtSessionOptions* options,
                    _In_ IDMLDevice* dml_device, _In_ ID3D12CommandQueue* cmd_queue) {
API_IMPL_BEGIN
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_DML(options->value.config_options,
                                                                                        dml_device,
                                                                                        cmd_queue,
                                                                                        false,
                                                                                        false));
API_IMPL_END
  return nullptr;
}

ORT_API_STATUS_IMPL(CreateGPUAllocationFromD3DResource, _In_ ID3D12Resource* d3d_resource, _Out_ void** dml_resource) {
  API_IMPL_BEGIN
#ifdef USE_DML
  *dml_resource = Dml::CreateGPUAllocationFromD3DResource(d3d_resource);
#else
  *dml_resource = nullptr;
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(FreeGPUAllocation, _In_ void* ptr) {
  API_IMPL_BEGIN
#ifdef USE_DML
  Dml::FreeGPUAllocation(ptr);
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_DML2, _In_ OrtSessionOptions* options, OrtDmlDeviceOptions* device_options) {
API_IMPL_BEGIN
#ifdef USE_DML
  auto factory = onnxruntime::DMLProviderFactoryCreator::CreateFromDeviceOptions(options->value.config_options, device_options, false, false);
  // return the create function for a dxcore device
  options->provider_factories.push_back(factory);
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(GetD3D12ResourceFromAllocation, _In_ OrtAllocator* ort_allocator, _In_ void* allocation, _Out_ ID3D12Resource** d3d_resource) {
  API_IMPL_BEGIN
#ifdef USE_DML
  auto wrapping_allocator = static_cast<onnxruntime::OrtAllocatorImplWrappingIAllocator*>(ort_allocator);
  auto allocator = wrapping_allocator->GetWrappedIAllocator();
  if (!allocator) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "No requested allocator available");
  }
  *d3d_resource = Dml::GetD3D12ResourceFromAllocation(allocator.get(), allocation);
  (*d3d_resource)->AddRef();
#else
  *d3d_resource = nullptr;
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

static constexpr OrtDmlApi ort_dml_api_10_to_x = {
  &OrtSessionOptionsAppendExecutionProvider_DML,
  &OrtSessionOptionsAppendExecutionProviderEx_DML,
  &CreateGPUAllocationFromD3DResource,
  &FreeGPUAllocation,
  &GetD3D12ResourceFromAllocation,
  &OrtSessionOptionsAppendExecutionProvider_DML2,
};

const OrtDmlApi* GetOrtDmlApi(_In_ uint32_t /*version*/) NO_EXCEPTION {
#ifdef USE_DML
  return &ort_dml_api_10_to_x;
#else
    return nullptr;
#endif
}
