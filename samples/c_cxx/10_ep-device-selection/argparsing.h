#pragma once

#include "utils.h"

#include <onnxruntime/core/graph/constants.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

struct Opts {
  std::string input_image;
  std::string output_image;
  std::string select_vendor;
  std::string select_ep;
  bool enableEpContext{true};
  OrtExecutionProviderDevicePolicy ep_device_policy =
      OrtExecutionProviderDevicePolicy_PREFER_GPU;
};

struct ArgumentSpec {
  const char *name;
  const char *short_name;
  const char *help;
  int num_args;
  std::function<bool(int)> lambda;
};

static Opts parse_args(int argc, char **argv) {
  using namespace std::string_view_literals;
  Opts opts;
  auto arg_specs = std::array{
      // clang-format off
    ArgumentSpec{
      "--input", "-i", "Path to input image (*.png)", 1, [&](int i) {
        opts.input_image = argv[i + 1];
        if (opts.input_image.starts_with("-")) {
          LOG("Path to input image can't start with -: \"{}\"",
                 opts.input_image.c_str());
          return false;
        }
        return true;
      }},
    ArgumentSpec{
      "--output", "-o", "Path where to save output image (*.png)", 1, [&](int i) {
        opts.output_image = argv[i + 1];
        if (opts.output_image.starts_with("-")) {
          LOG("Path to output image can't start with -: \"{}\"",
                 opts.output_image.c_str());
          return false;
        }
        return true;
      }},
    ArgumentSpec{
      "--select-vendor", "-f", "Select device of provided vendor.", 1, [&](int i) {
        opts.select_vendor = argv[i + 1];
        if (opts.select_vendor.starts_with("-")) {
          LOG("Vendor can't start with -: \"{}\"",
                 opts.select_vendor.c_str());
          return false;
        }
        return true;
      }},
    ArgumentSpec{
      "--select-ep", "-f", "Select devices that support a specific execution provider. "
        "See https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/graph/constants.h for EP names."
        , 1, [&](int i) {
        opts.select_vendor = argv[i + 1];
        if (opts.select_vendor.starts_with("-")) {
          LOG("Execution provider can't start with -: \"{}\"",
                 opts.select_vendor.c_str());
          return false;
        }
        return true;
      }},
    ArgumentSpec{
      "--ep-device-policy", "-p", "Set a EP device policy: e.g. prefer-cpu, prefer-gpu, prefer-npu, max-performance, max-efficiency, min-overall-power", 1, [&](int i) {
        if(argv[i+1] == "prefer-cpu"sv) {
          opts.ep_device_policy = OrtExecutionProviderDevicePolicy_PREFER_CPU;
        } else if(argv[i+1] == "prefer-gpu"sv) {
          opts.ep_device_policy = OrtExecutionProviderDevicePolicy_PREFER_GPU;
        } else if(argv[i+1] == "prefer-npu"sv) {
          opts.ep_device_policy = OrtExecutionProviderDevicePolicy_PREFER_NPU;
        } else if(argv[i+1] == "max-performance"sv) {
          opts.ep_device_policy = OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE;
        } else if(argv[i+1] == "max-efficiency"sv) {
          opts.ep_device_policy = OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY;
        } else if(argv[i+1] == "min-overall-power"sv) {
          opts.ep_device_policy = OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER;
        } else {
          LOG("Invalid execution provider policy: \"{}\"! Choose among prefer-cpu, prefer-gpu, prefer-npu, max-performance, max-efficiency, min-overall-power", argv[i+1]);
          return false;
        }
        return true;
      }}
      // clang-format on
  };
  auto print_usage = [&] {
    LOG("");
    LOG("Usage:");
    LOG("{} <options>", argv[0]);
    for (auto &spec : arg_specs) {
      if (spec.short_name) {
        LOG("\t{} {}    {}", spec.name, spec.short_name, spec.help);
      } else {
        LOG("\t{}    {}", spec.name, spec.help);
      }
    }
  };
  for (int i = 1; i < argc; i++) {
    bool arg_found = false;
    for (auto &spec : arg_specs) {
      if (std::strcmp(spec.name, argv[i]) == 0 ||
          (spec.short_name && std::strcmp(spec.short_name, argv[i]) == 0)) {
        if (i + spec.num_args < argc) {
          bool ok = spec.lambda(i);
          if (!ok) {
            LOG("Failed to parse arguments for {}!", spec.name);
            exit(EXIT_FAILURE);
          }
          arg_found = true;
          i += spec.num_args;
          break;
        } else {
          LOG("Not enough arguments for {} specified!", spec.name);
          exit(EXIT_FAILURE);
        }
      }
    }
    if (!arg_found) {
      auto arg = argv[i];
      LOG("Unknown argument: {}", arg);
      print_usage();
      exit(EXIT_FAILURE);
    }
  }
  if (opts.input_image.empty()) {
    opts.input_image = (get_executable_path().parent_path() / "Input.png").string();
  }
  if (opts.output_image.empty()) {
    opts.output_image = (get_executable_path().parent_path() / "output.png").string();
  }
  if (!std::filesystem::is_regular_file(opts.input_image)) {
    LOG("Please make sure that provided input image path exists: \"{}\"!",
        opts.input_image.c_str());
    print_usage();
    exit(EXIT_FAILURE);
  }
  if (!std::filesystem::is_directory(
          std::filesystem::path(opts.output_image).parent_path())) {
    LOG("Please make sure that the parent directory of the provided output "
        "path exists: \"{}\"!",
        opts.output_image.c_str());
    print_usage();
    exit(EXIT_FAILURE);
  }
  return opts;
}
