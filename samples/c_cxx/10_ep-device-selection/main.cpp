#include <onnxruntime/core/graph/constants.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_run_options_config_keys.h>
#include <onnxruntime/core/session/onnxruntime_session_options_config_keys.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <regex>
#include <vector>

#include "argparsing.h"
#include "lodepng/lodepng.h"
#include "utils.h"

#if ORT_API_VERSION < 23
#error "Onnx runtime header too old. Version >=1.23.0 assumed"
#endif

using OrtFileString = std::basic_string<ORTCHAR_T>;

static OrtFileString toOrtFileString(const std::filesystem::path& path) {
  std::string string(path.string());
  return {string.begin(), string.end()};
}

#define PROVIDER_LIB_PAIR(NAME) \
  std::pair { NAME, DLL_NAME("onnxruntime_providers_" NAME) }

static void register_execution_providers(Ort::Env& env) {
  // clang-format off
  std::array provider_libraries{
      PROVIDER_LIB_PAIR("nv_tensorrt_rtx"),
      PROVIDER_LIB_PAIR("cuda"),
      PROVIDER_LIB_PAIR("openvino"),
      PROVIDER_LIB_PAIR("qnn"),
      PROVIDER_LIB_PAIR("cann"),
  };
  // clang-format on

  for (auto& [registration_name, dll] : provider_libraries) {
    auto providers_library = get_executable_path().parent_path() / dll;
    if (!std::filesystem::is_regular_file(providers_library)) {
      LOG("{} does not exist! Skipping execution provider", providers_library.string());
      continue;
    }
    try {
      env.RegisterExecutionProviderLibrary(registration_name, toOrtFileString(providers_library));
    } catch (std::exception& ex) {
      LOG("Failed to register {}! Skipping execution provider", providers_library.string());
    }
  }
}

Ort::ConstMemoryInfo match_common_memory_info(const Ort::Session& input_session, const Ort::Session& output_session) {
  auto input_infos = input_session.GetMemoryInfoForOutputs();
  auto output_infos = output_session.GetMemoryInfoForInputs();

  // First try to find a common non-CPU allocator
  for (auto& in : input_infos) {
    for (auto& out : output_infos) {
      if (in == out && in.GetDeviceType() != OrtMemoryInfoDeviceType_CPU &&
          in.GetDeviceMemoryType() == OrtDeviceMemoryType_DEFAULT) {
        return in;
      }
    }
  }
  // If impossible then also allow to fall back to CPU
  for (auto& in : input_infos) {
    for (auto& out : output_infos) {
      if (in == out) {
        return in;
      }
    }
  }
  THROW_ERROR("Could not find a common allocator");
}

static Ort::SessionOptions create_session_options(Ort::Env& env, const Opts& opts) {
  std::vector<Ort::ConstEpDevice> selected_devices;
  auto ep_devices = env.GetEpDevices();
  LOG("{} devices found", ep_devices.size());
  for (auto& device : ep_devices) {
    auto metadata = device.Device().Metadata();
    // LUID can be used on Windows platform to match EpDevices with
    // IDXGIAdapter in case an application already has a device selection
    // logic based on `IDXGIAdapter`s
    auto luid = metadata.GetValue("LUID");
    LOG("Vendor: {}, EpName: {}, DeviceId: 0x{:x}, LUID: {}", device.EpVendor(), device.EpName(),
        device.Device().DeviceId(), luid ? luid : "<unavailable>");
    if (to_uppercase(opts.select_vendor) == device.Device().Vendor()) {
      selected_devices.push_back(device);
    }
    if (to_uppercase(opts.select_ep) == device.EpName()) {
      selected_devices.push_back(device);
    }
  }

  Ort::SessionOptions so;
  if (!selected_devices.empty()) {
    Ort::KeyValuePairs ep_options;
    // Select EP for manually selected devices
    so.AppendExecutionProvider_V2(env, selected_devices, ep_options);
  }

  so.SetEpSelectionPolicy(opts.ep_device_policy);
  return so;
}

static Ort::Session create_session(Ort::Env& env, std::filesystem::path& model_file,
                                   const Ort::SessionOptions& session_options) {
  if (!std::filesystem::is_regular_file(model_file)) {
    THROW_ERROR("Model file \"{}\" does not exist!", model_file.string());
  }

  Ort::Session session(env, toOrtFileString(model_file).c_str(), session_options);
  return session;
}

auto main(int argc, char** argv) -> int {
  try {
    Opts opts = parse_args(argc, argv);

    auto api = Ort::GetApi();
    auto version_string = Ort::GetVersionString();
    auto build_info = api.GetBuildInfoString();

    LOG("Hello from ONNX runtime version: {} (build info {})\n", version_string, build_info);

    // Setup ORT environment
    auto env = Ort::Env(ORT_LOGGING_LEVEL_WARNING);
    register_execution_providers(env);
    // Create session options for ORT environment according to command line
    // parameters
    auto session_options = create_session_options(env, opts);

    // Load a ONNX files
    std::string model_file = MODEL_FILE;
    auto model_path = get_executable_path().parent_path() / MODEL_FILE;  // defined via CMAKE
    auto model_context_file = std::regex_replace(model_file, std::regex(".onnx$"), "_ctx.onnx");
    auto model_context_path = get_executable_path().parent_path() / model_context_file;
    bool use_model_context = std::filesystem::is_regular_file(model_context_path);
    auto load_path = use_model_context ? model_context_path : model_path;

    // Prepare inputs
    uint8_t* image{};
    DEFER(image, free(image));
    uint32_t width{};
    uint32_t height{};
    auto error = lodepng_decode32_file(&image, &width, &height, opts.input_image.c_str());
    if (error) {
      LOG("Failed to load image \"{}\"", opts.input_image);
      return EXIT_FAILURE;
    }
    LOG("Loaded image \"{}\" with size {}x{}", opts.input_image, width, height);

    CHECK_ORT(api.AddFreeDimensionOverrideByName(session_options, "N", 1));
    CHECK_ORT(api.AddFreeDimensionOverrideByName(session_options, "W", width));
    CHECK_ORT(api.AddFreeDimensionOverrideByName(session_options, "H", height));
    if (!use_model_context) {
      session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, opts.enableEpContext ? "1" : "0");
    }

    auto infer_session = create_session(env, load_path, session_options);

    Ort::AllocatorWithDefaultOptions cpu_allocator;
    std::array input_shape{int64_t(1), int64_t(height), int64_t(width), int64_t(4)};
    auto output_shape = infer_session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    // This allocates input and output on CPU. This is probably not what you want when doing multiple inferences on a GPU
    // See the sample (20_devicetensors-datatransfers) how to device memory and device synchronization streams efficiently
    Ort::Value input_value = Ort::Value::CreateTensor<uint8_t>(cpu_allocator.GetInfo(), image, width * height * 4,
                                                               input_shape.data(), input_shape.size());
    Ort::Value output_value =
        Ort::Value::CreateTensor<uint8_t>(cpu_allocator, output_shape.data(), output_shape.size());

    Ort::IoBinding inference_binding(infer_session);
    inference_binding.BindInput("input", input_value);
    inference_binding.BindOutput("depth", output_value);

    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    infer_session.Run(run_options, inference_binding);
    inference_binding.SynchronizeOutputs();

    lodepng_encode_file(opts.output_image.c_str(), output_value.GetTensorData<uint8_t>(), output_shape[2],
                        output_shape[1], LCT_GREY, 8);

  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
