// ---------------------------------------------------------------------------
// Model package sample.
//
// Opens a model package, lets ONNX Runtime select the best variant for the
// available execution providers, creates a session, and runs one inference for
// every component in the package.
//
// This file uses the ORT C++ API (onnxruntime_cxx_api.h) plus the small
// model-package wrappers in model_package_cxx.h, so consuming a package reads as:
//
//     Ort::ModelPackage pkg{package_root};
//     auto component = pkg.SelectComponent(env, name, session_options);
//     Ort::Session   session = component.CreateSession(env);
//
// Build (see cpp/README.md), then:
//     model_package_sample <onnxruntime.dll> <package_root> [openvino_plugin.dll]
//
// For each component the CPU variant is run, and the OpenVINO NPU variant too when
// an OpenVINO plugin is supplied and an NPU is present.
// ---------------------------------------------------------------------------
#include <windows.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// onnxruntime.dll is loaded dynamically at runtime (no import library), so the
// C++ API must be pointed at the loaded OrtApi via Ort::InitApi() before use.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "model_package_cxx.h"

namespace {

std::wstring Widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n, 0);
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
  if (!w.empty() && w.back() == 0) w.pop_back();
  return w;
}

// Load onnxruntime.dll dynamically and point the C++ API at it (no import library
// needed). Its directory is added to the search path so onnxruntime_providers_shared.dll
// next to it is found.
void InitOrtFromDll(const std::string& ort_dll) {
  std::wstring dll_w = Widen(ort_dll);
  size_t slash = dll_w.find_last_of(L"\\/");
  if (slash != std::wstring::npos) SetDllDirectoryW(dll_w.substr(0, slash).c_str());

  HMODULE h = LoadLibraryW(dll_w.c_str());
  if (h == nullptr) throw std::runtime_error("LoadLibrary failed for " + ort_dll);
  auto get_api_base = reinterpret_cast<const OrtApiBase*(ORT_API_CALL*)()>(GetProcAddress(h, "OrtGetApiBase"));
  if (get_api_base == nullptr) throw std::runtime_error("OrtGetApiBase not found in " + ort_dll);
  const OrtApiBase* api_base = get_api_base();
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) throw std::runtime_error("onnxruntime.dll is too old for this header (ORT_API_VERSION).");
  Ort::InitApi(api);
  std::cout << "ORT version: " << api_base->GetVersionString() << std::endl;
}

// The first OrtEpDevice for `ep_name` whose hardware type matches `type`.
std::optional<Ort::ConstEpDevice> FindDevice(const std::vector<Ort::ConstEpDevice>& devices,
                                             const char* ep_name, OrtHardwareDeviceType type) {
  for (const auto& d : devices) {
    if (std::string(d.EpName()) == ep_name && d.Device().Type() == type) return d;
  }
  return std::nullopt;
}

// Run one inference with an all-0.5 input on the session's first input; print stats.
void RunOnce(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto in_name = session.GetInputNameAllocated(0, allocator);
  auto out_name = session.GetOutputNameAllocated(0, allocator);

  auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  int64_t count = 1;
  for (auto& d : shape) {
    if (d < 0) d = 1;  // shapes are static in this sample; guard just in case
    count *= d;
  }
  std::vector<float> input(static_cast<size_t>(count), 0.5f);

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                            shape.data(), shape.size());

  const char* in_names[] = {in_name.get()};
  const char* out_names[] = {out_name.get()};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

  const float* out = outputs[0].GetTensorData<float>();
  size_t out_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  double absmean = 0.0;
  bool all_zero = true;
  for (size_t i = 0; i < out_count; ++i) {
    absmean += out[i] < 0 ? -out[i] : out[i];
    if (out[i] != 0.0f) all_zero = false;
  }
  absmean /= (out_count ? out_count : 1);

  std::cout << "      input dims=[";
  for (size_t i = 0; i < shape.size(); ++i) std::cout << shape[i] << (i + 1 < shape.size() ? "," : "");
  std::cout << "] output elems=" << out_count << " all_zero=" << (all_zero ? "true" : "false")
            << " absmean=" << absmean << std::endl;
}

// Select + run one component for the given EP device (nullopt => default CPU selection).
void RunComponent(Ort::Env& env, Ort::ModelPackage& pkg, const std::string& component,
                  const char* label, const std::optional<Ort::ConstEpDevice>& device) {
  std::cout << "\n[component=" << component << " target=" << label << "]" << std::endl;

  Ort::SessionOptions session_options;
  if (device.has_value()) {
    session_options.AppendExecutionProvider_V2(env, {*device}, {});
  }
  // else: no EP appended -> the selector picks the CPU variant.

  try {
    Ort::ModelPackageComponent comp = pkg.SelectComponent(env, component, session_options);
    std::cout << "  selected variant: " << comp.SelectedVariantName() << std::endl;
    Ort::Session session = comp.CreateSession(env);
    std::cout << "  session created" << std::endl;
    RunOnce(session);
  } catch (const Ort::Exception& e) {
    std::cout << "  not available: " << e.what() << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: model_package_sample <onnxruntime.dll> <package_root> [openvino_plugin.dll]\n";
    return 1;
  }
  const std::string ort_dll = argv[1];
  const std::string package_root = argv[2];
  const std::string ov_plugin = (argc >= 4) ? argv[3] : std::string();

  try {
    InitOrtFromDll(ort_dll);

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mp_sample"};

    // Register the OpenVINO EP (optional) and look for an NPU device.
    std::optional<Ort::ConstEpDevice> ov_npu;
    if (!ov_plugin.empty()) {
      env.RegisterExecutionProviderLibrary("OpenVINOExecutionProvider", Widen(ov_plugin));
      ov_npu = FindDevice(env.GetEpDevices(), "OpenVINOExecutionProvider", OrtHardwareDeviceType_NPU);
      std::cout << "OpenVINO EP registered; NPU device " << (ov_npu ? "found" : "NOT found") << std::endl;
    } else {
      std::cout << "No OpenVINO plugin supplied; running CPU variants only." << std::endl;
    }

    Ort::ModelPackage pkg{Widen(package_root)};
    for (const std::string& component : pkg.ComponentNames()) {
      RunComponent(env, pkg, component, "cpu", std::nullopt);
      if (ov_npu) RunComponent(env, pkg, component, "ov", ov_npu);
    }

    std::cout << "\nDONE" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 2;
  }
}
