#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

int wmain(int argument_count, wchar_t** arguments) {
  try {
    if (argument_count != 2) {
      throw std::invalid_argument("Pass the absolute path to the CUDA plug-in DLL");
    }
    std::cout << "ORT_VERSION=" << OrtGetApiBase()->GetVersionString() << std::endl;
    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "arm64-cuda-plugin-build-smoke");
    environment.RegisterExecutionProviderLibrary("cuda_plugin", arguments[1]);
    std::vector<Ort::ConstEpDevice> cuda_devices;
    for (const auto& device : environment.GetEpDevices()) {
      if (std::string(device.EpName()) == "CUDAExecutionProvider") {
        cuda_devices.push_back(device);
      }
    }
    if (cuda_devices.empty()) {
      throw std::runtime_error("The CUDA plug-in did not enumerate any CUDA devices");
    }
    std::cout << "ORT_CUDA_PLUGIN_DEVICE_COUNT=" << cuda_devices.size() << std::endl;
    {
      Ort::SessionOptions options;
      options.AppendExecutionProvider_V2(environment, cuda_devices, std::unordered_map<std::string, std::string>{});
    }
    environment.UnregisterExecutionProviderLibrary("cuda_plugin");
    std::cout << "ORT_CUDA_PLUGIN_REGISTRATION_SUCCEEDED" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}