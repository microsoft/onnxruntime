#include "wsl_device_discovery.h"
#include <dlfcn.h>
#include <wsl/winadapter.h>
#include <directx/dxcore.h>
#include "core/common/common.h"
#include "core/common/logging/logging.h"

#include <filesystem>

namespace fs = std::filesystem;

namespace onnxruntime {
namespace {
Status ErrorCodeToStatus(const std::error_code& ec) {
  if (!ec) {
    return Status::OK();
  }

  return Status{common::StatusCategory::ONNXRUNTIME, common::StatusCode::FAIL,
                MakeString("Error: std::error_code with category name: ", ec.category().name(),
                           ", value: ", ec.value(), ", message: ", ec.message())};
}
//// NTSTATUS values
// #define STATUS_SUCCESS ((NTSTATUS)0x00000000L)
// #define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)

// using NTSTATUS = LONG;
// using D3DKMT_HANDLE = UINT;

//// Adapter state flags for D3DKMTEnumAdapters3
// enum D3DKMT_ADAPTERSTATE : UINT {
// D3DKMT_ADAPTERSTATE_NONE = 0,
//};

//// D3DKMT_ADAPTERTYPE - describes the type of adapter
// struct D3DKMT_ADAPTERTYPE {
// union {
// struct {
// UINT RenderSupported : 1;
// UINT DisplaySupported : 1;
// UINT SoftwareDevice : 1;
// UINT PostDevice : 1;
// UINT HybridDiscrete : 1;
// UINT HybridIntegrated : 1;
// UINT IndirectDisplayDevice : 1;
// UINT Paravirtualized : 1;
// UINT ACGSupported : 1;
// UINT SupportSetTimingsFromVidPn : 1;
// UINT Detachable : 1;
// UINT ComputeOnly : 1;
// UINT Prototype : 1;
// UINT RuntimePowerManagement : 1;
// UINT Reserved : 18;
//};
// UINT Value;
//};
//};

//// D3DKMT_ADAPTERINFO - adapter information returned by D3DKMTEnumAdapters3
// struct D3DKMT_ADAPTERINFO {
// D3DKMT_HANDLE hAdapter;
// LUID AdapterLuid;
// ULONG NumOfSources;
// BOOL bPrecisePresentRegionsPreferred;
//};

//// D3DKMT_ENUMADAPTERS2 - input/output structure for D3DKMTEnumAdapters2
// struct D3DKMT_ENUMADAPTERS2 {
// ULONG NumAdapters;
// D3DKMT_ADAPTERINFO* pAdapters;
//};

//// D3DKMT_ENUMADAPTERS3 - input/output structure for D3DKMTEnumAdapters3
// struct D3DKMT_ENUMADAPTERS3 {
// D3DKMT_ADAPTERSTATE Filter;
// ULONG NumAdapters;
// D3DKMT_ADAPTERINFO* pAdapters;
//};

//// D3DKMT_QUERYADAPTERINFO command types
// enum KMTQUERYADAPTERINFOTYPE : UINT {
// KMTQAITYPE_UMDRIVERPRIVATE = 0,
// KMTQAITYPE_UMDRIVERNAME = 1,
// KMTQAITYPE_UMOPENGLINFO = 2,
// KMTQAITYPE_GETSEGMENTSIZE = 3,
// KMTQAITYPE_ADAPTERGUID = 4,
// KMTQAITYPE_FLIPQUEUEINFO = 5,
// KMTQAITYPE_ADAPTERADDRESS = 6,
// KMTQAITYPE_SETWORKINGSETINFO = 7,
// KMTQAITYPE_ADAPTERREGISTRYINFO = 8,
// KMTQAITYPE_CURRENTDISPLAYMODE = 9,
// KMTQAITYPE_MODELIST = 10,
// KMTQAITYPE_CHECKDRIVERUPDATESTATUS = 11,
// KMTQAITYPE_VIRTUALADDRESSINFO = 12,
// KMTQAITYPE_DRIVERVERSION = 13,
// KMTQAITYPE_ADAPTERTYPE = 15,
// KMTQAITYPE_WDDM_1_2_CAPS = 24,
// KMTQAITYPE_WDDM_1_3_CAPS = 35,
// KMTQAITYPE_WDDM_2_0_CAPS = 44,
// KMTQAITYPE_PHYSICALADAPTERCOUNT = 48,
// KMTQAITYPE_PHYSICALADAPTERDEVICEIDS = 49,
// KMTQAITYPE_WDDM_2_7_CAPS = 70,
//};

//// D3DKMT_QUERYADAPTERINFO - query adapter info structure
// struct D3DKMT_QUERYADAPTERINFO {
// D3DKMT_HANDLE hAdapter;
// KMTQUERYADAPTERINFOTYPE Type;
// void* pPrivateDriverData;
// UINT PrivateDriverDataSize;
//};

//// D3DKMT_SEGMENTSIZEINFO - segment size information
// struct D3DKMT_SEGMENTSIZEINFO {
// ULONGLONG DedicatedVideoMemorySize;
// ULONGLONG DedicatedSystemMemorySize;
// ULONGLONG SharedSystemMemorySize;
//};

//// D3DKMT_ADAPTERADDRESS - PCI address info
// struct D3DKMT_ADAPTERADDRESS {
// UINT BusNumber;
// UINT DeviceNumber;
// UINT FunctionNumber;
//};

//// D3DKMT_ADAPTERREGISTRYINFO - registry info
// struct D3DKMT_ADAPTERREGISTRYINFO {
// WCHAR AdapterString[260];
// WCHAR BiosString[260];
// WCHAR DacType[260];
// WCHAR ChipType[260];
//};

//// D3DKMT_CLOSEADAPTER - close adapter structure
// struct D3DKMT_CLOSEADAPTER {
// D3DKMT_HANDLE hAdapter;
//};

// using PFN_D3DKMTEnumAdapters2 = NTSTATUS (*)(D3DKMT_ENUMADAPTERS2*);
// using PFN_D3DKMTEnumAdapters3 = NTSTATUS (*)(D3DKMT_ENUMADAPTERS3*);
// using PFN_D3DKMTQueryAdapterInfo = NTSTATUS (*)(const D3DKMT_QUERYADAPTERINFO*);
// using PFN_D3DKMTCloseAdapter = NTSTATUS (*)(const D3DKMT_CLOSEADAPTER*);
// using PFN_DXCoreCreateAdapterFactory = HRESULT (*)(const GUID& riid, void** ppvFactory);

// class DxCoreLib {
// public:
// PFN_D3DKMTEnumAdapters2 D3DKMTEnumAdapters2;
// PFN_D3DKMTQueryAdapterInfo D3DKMTQueryAdapterInfo;
// PFN_D3DKMTCloseAdapter D3DKMTCloseAdapter;
// PFN_DXCoreCreateAdapterFactory DXCoreCreateAdapterFactory;
// Status load() {
// m_handle = dlopen("libdxcore.so", RTLD_LAZY);
// if (!m_handle) {
// return Status{common::StatusCategory::ONNXRUNTIME, common::StatusCode::FAIL, "Could not dlopen libdxcore.so on WSL"};
//}
// D3DKMTEnumAdapters2 = (decltype(D3DKMTEnumAdapters2))dlsym(m_handle, "D3DKMTEnumAdapters2");
// D3DKMTQueryAdapterInfo = (decltype(D3DKMTQueryAdapterInfo))dlsym(m_handle, "D3DKMTQueryAdapterInfo");
// D3DKMTCloseAdapter = (decltype(D3DKMTCloseAdapter))dlsym(m_handle, "D3DKMTCloseAdapter");
// if (!D3DKMTEnumAdapters2 || !D3DKMTQueryAdapterInfo || !D3DKMTCloseAdapter) {
// return Status{common::StatusCategory::ONNXRUNTIME, common::StatusCode::FAIL, "Could not get function pointers from libdxcore.so on WSL"};
//}
// return Status::OK();
//}

// private:
// void* m_handle;
//};
}  // namespace

Status
DetectGpuIfWsl(std::vector<OrtHardwareDevice>& gpu_devices_out) {
  std::error_code error_code{};
  auto status = Status::OK();

  const fs::path dxg_path = "/dev/dxg";
  const bool dxg_exists = fs::exists(dxg_path, error_code);
  ORT_RETURN_IF_ERROR(ErrorCodeToStatus(error_code));
  const bool is_wsl_with_graphics = dxg_exists && fs::is_character_file(dxg_path, error_code);
  ORT_RETURN_IF_ERROR(ErrorCodeToStatus(error_code));
  if (!is_wsl_with_graphics) {
    return Status::OK();
  }
  LOGS_DEFAULT(INFO) << "Detected GPU enabled WSL: /dev/dxg is present";

  ORT_RETURN_IF_ERROR(DetectGpuNvml(gpu_devices_out));

  return Status::OK();

  // DxCoreLib dxcore_lib;
  // ORT_RETURN_IF_ERROR(dxcore_lib.load());

  // IDXCoreAdapterFactory* factory = nullptr;
  // IDXCoreAdapterList* adapterList = nullptr;
  // HRESULT hr;

  // hr = dxcore_lib.DXCoreCreateAdapterFactory(IID_IDXCoreAdapterFactory, reinterpret_cast<void**>(&factory));
  // if (FAILED(hr)) {
  // }

  // hr = factory->CreateAdapterList(1,
  //&DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS,
  // IID_IDXCoreAdapterList,
  // reinterpret_cast<void**>(&adapterList));

  // auto adapterCount = adapterList->GetAdapterCount();
  // for (uint32_t i = 0; i < adapterCount; i++) {
  // IDXCoreAdapter* adapter = nullptr;

  // hr = adapterList->GetAdapter(i, IID_IDXCoreAdapter, reinterpret_cast<void**>(&adapter));
  // if (FAILED(hr)) {
  // }

  // if (!adapter->IsValid()) {
  // }

  //// Get LUID (Instance Locally Unique Identifier)
  // if (adapter->IsPropertySupported(DXCoreAdapterProperty::InstanceLuid)) {
  // LUID luid = {};
  // hr = adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, sizeof(luid), &luid);
  // if (SUCCEEDED(hr)) {
  //// printf("  LUID: %08X-%08X\n", static_cast<unsigned>(luid.HighPart), luid.LowPart);
  //// collectedLuids.push_back(luid);
  //} else {
  //// print_error("  GetProperty(InstanceLuid)", hr);
  //}
  //} else {
  //// printf("  LUID: [property not supported]\n");
  //}

  //// Get driver description
  // if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription)) {
  // size_t descSize = 0;
  // hr = adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &descSize);
  // if (SUCCEEDED(hr) && descSize > 0) {
  // char* description = static_cast<char*>(malloc(descSize));
  // if (description) {
  // hr = adapter->GetProperty(DXCoreAdapterProperty::DriverDescription,
  // descSize, description);
  // if (SUCCEEDED(hr)) {
  // printf("  Driver: %s\n", description);
  //}
  // free(description);
  //}
  //}
  //}

  //// Get Hardware ID
  // if (adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareID)) {
  // DXCoreHardwareID hwid = {};
  // hr = adapter->GetProperty(DXCoreAdapterProperty::HardwareID, sizeof(hwid), &hwid);
  // if (SUCCEEDED(hr)) {
  // printf("  PCI ID: VEN_%04X DEV_%04X SUBSYS_%08X REV_%02X\n",
  // hwid.vendorID, hwid.deviceID, hwid.subSysID, hwid.revision);
  //}
  //}

  //// Get dedicated adapter memory
  // if (adapter->IsPropertySupported(DXCoreAdapterProperty::DedicatedAdapterMemory)) {
  // uint64_t mem = 0;
  // hr = adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory, sizeof(mem), &mem);
  // if (SUCCEEDED(hr) && mem > 0) {
  // printf("  Dedicated VRAM: %lu MB\n", static_cast<unsigned long>(mem / (1024 * 1024)));
  //}
  //}

  //// Check if it's a hardware adapter
  // if (adapter->IsPropertySupported(DXCoreAdapterProperty::IsHardware)) {
  // bool isHardware = false;
  // hr = adapter->GetProperty(DXCoreAdapterProperty::IsHardware, sizeof(isHardware), &isHardware);
  // if (SUCCEEDED(hr)) {
  // printf("  Hardware: %s\n", isHardware ? "Yes" : "No (software renderer)");
  //}
  //}

  //// Check if it's an integrated GPU
  // if (adapter->IsPropertySupported(DXCoreAdapterProperty::IsIntegrated)) {
  // bool isIntegrated = false;
  // hr = adapter->GetProperty(DXCoreAdapterProperty::IsIntegrated, sizeof(isIntegrated), &isIntegrated);
  // if (SUCCEEDED(hr)) {
  // printf("  Integrated: %s\n", isIntegrated ? "Yes" : "No (discrete GPU)");
  //}
  //}

  // printf("--------------------------------------------------------------------------------\n");
  // adapter->Release();
  //}

  // D3DKMT_ENUMADAPTERS2 enum_adapters2{};
  // if (!NT_SUCCESS(dxcore_lib.D3DKMTEnumAdapters2(&enum_adapters2))) {
  // }
  // std::vector<D3DKMT_ADAPTERINFO> adapters2(enum_adapters2.NumAdapters);
  // if (!NT_SUCCESS(dxcore_lib.D3DKMTEnumAdapters2(&enum_adapters2))) {
  // }
  // for (const auto& adapter : adapters2) {
  //// auto virtualized_luid = adapter.AdapterLuid;
  // D3DKMT_ADAPTERADDRESS pci_address{};
  // D3DKMT_QUERYADAPTERINFO query_info{};

  // query_info.Type = KMTQAITYPE_ADAPTERADDRESS;
  // query_info.hAdapter = adapter.hAdapter;
  // query_info.pPrivateDriverData = &pci_address;
  // query_info.PrivateDriverDataSize = sizeof(pci_address);
  // dxcore_lib.D3DKMTQueryAdapterInfo(&query_info);
  //}

  //// There are different virtualised devices as PCI devices at
  //// dxcore can give use their properties.
  ////
  //// TODO

  ////// dxcore devices are not usable for CUDA though since they have different LUIDs and PCI bus ids
  ////// We use the presence of libnvidia-ml.so.1 as an indication that Nvidia GPUs are present
  //// ORT_RETURN_IF_ERROR(DetectGpuNvml(gpu_devices_out));
  // return status;
}

Status DetectGpuNvml(std::vector<OrtHardwareDevice>& gpu_devices_out) {
  void* nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (!nvml_handle) {
    // if nvml is not found is normal if no Nvidia driver is installed
    // we just report back 0 detected NVML devices
    return Status::OK();
  }
  typedef struct nvmlPciInfo_st {
    char busIdLegacy[16];      //!< The legacy tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;       //!< The PCI domain on which the device's bus resides, 0 to 0xffffffff
    unsigned int bus;          //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;       //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;  //!< The combined 16-bit device id and 16-bit vendor id

    // Added in NVML 2.285 API
    unsigned int pciSubSystemId;  //!< The 32-bit Sub System Device ID

    char busId[32];  //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
  } nvmlPciInfo_t;
  int (*PFN_nvmlInit_v2)();
  int (*PFN_nvmlShutdown)();
  int (*PFN_nvmlDeviceGetCount_v2)(unsigned int*);
  int (*PFN_nvmlDeviceGetHandleByIndex_v2)(unsigned int index, void* device);
  int (*PFN_nvmlDeviceGetPciInfo_v3)(void* device, nvmlPciInfo_t* pci);
  PFN_nvmlInit_v2 = (decltype(PFN_nvmlInit_v2))dlsym(nvml_handle, "nvmlInit_v2");
  PFN_nvmlShutdown = (decltype(PFN_nvmlShutdown))dlsym(nvml_handle, "nvmlShutdown");
  PFN_nvmlDeviceGetCount_v2 = (decltype(PFN_nvmlDeviceGetCount_v2))dlsym(nvml_handle, "nvmlDeviceGetCount_v2");
  PFN_nvmlDeviceGetHandleByIndex_v2 = (decltype(PFN_nvmlDeviceGetHandleByIndex_v2))dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex_v2");
  PFN_nvmlDeviceGetPciInfo_v3 = (decltype(PFN_nvmlDeviceGetPciInfo_v3))dlsym(nvml_handle, "nvmlDeviceGetPciInfo_v3");
  if (!PFN_nvmlInit_v2 || !PFN_nvmlShutdown || !PFN_nvmlDeviceGetCount_v2 || !PFN_nvmlDeviceGetHandleByIndex_v2 || !PFN_nvmlDeviceGetPciInfo_v3) {
    return Status{common::StatusCategory::ONNXRUNTIME, common::StatusCode::FAIL, "Failed to load NVML functions"};
  }
  int err{};
  unsigned int num_gpus{};
  err = PFN_nvmlInit_v2();
  if (err) {
    goto error;
  }
  err = PFN_nvmlDeviceGetCount_v2(&num_gpus);
  if (err) {
    goto error;
  }
  for (unsigned int i = 0; i < num_gpus; i++) {
    nvmlPciInfo_t info{};
    void* dev_handle{};
    err = PFN_nvmlDeviceGetHandleByIndex_v2(i, &dev_handle);
    if (err) {
      goto error;
    }
    err = PFN_nvmlDeviceGetPciInfo_v3(dev_handle, &info);
    if (err) {
      goto error;
    }
    auto gpu_device = OrtHardwareDevice{OrtHardwareDeviceType_GPU, 0x10de, info.pciDeviceId, "NVIDIA", {}};
    gpu_device.metadata.Add("pci_bus_id", info.busId);
    gpu_devices_out.push_back(gpu_device);
  }

  PFN_nvmlShutdown();
  dlclose(nvml_handle);
  return Status::OK();
error:
  PFN_nvmlShutdown();
  dlclose(nvml_handle);
  return Status{common::StatusCategory::ONNXRUNTIME, common::StatusCode::FAIL, "NVML error"};
}
}  // namespace onnxruntime
