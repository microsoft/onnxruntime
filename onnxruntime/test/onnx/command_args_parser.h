// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/platform/path_lib.h"
#include "utils/utils.h"

namespace onnxruntime {
namespace test {
struct TestConfig {
  // if this var is not empty, only run the tests with name in this list
  std::vector<std::basic_string<PATH_CHAR_TYPE>> whitelisted_test_cases;
  int concurrent_session_runs = utils::GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  int repeat_count = 1;
  bool inference_mode = false;
  int p_models = utils::GetNumCpuCores();
  bool enable_cuda = false;
  bool enable_dnnl = false;
  bool enable_openvino = false;
  bool enable_tensorrt = false;
  bool enable_mem_pattern = true;
  bool enable_qnn = false;
  bool enable_nnapi = false;
  bool enable_vsinpu = false;
  bool enable_coreml = false;
  bool enable_snpe = false;
  bool enable_dml = false;
  bool enable_acl = false;
  bool enable_armnn = false;
  bool enable_rocm = false;
  bool enable_migraphx = false;
  bool enable_webgpu = false;
  bool enable_xnnpack = false;
  bool override_tolerance = false;
  double atol = 1e-5;
  double rtol = 1e-5;
  int device_id = 0;
  GraphOptimizationLevel graph_optimization_level = ORT_ENABLE_ALL;
  bool user_graph_optimization_level_set = false;
  bool set_denormal_as_zero = false;
  std::basic_string<ORTCHAR_T> ep_runtime_config_string;
  std::unordered_map<std::string, std::string> session_config_entries;
  std::string provider_name = "cpu";

  bool verbose_logging_required = false;
  bool ep_context_enable = false;
  bool disable_ep_context_embed_mode = false;
  bool pause = false;

  std::basic_string<ORTCHAR_T> plugin_ep_names_and_libs;
  std::vector<std::string> registered_plugin_eps;
  std::vector<std::string> plugin_ep_list;
  std::string selected_ep_device_indices;
  bool list_available_ep_devices = false;

  std::vector<std::basic_string<PATH_CHAR_TYPE>> data_dirs;
};

class CommandLineParser {
 public:
  static bool ParseArguments(TestConfig& test_config, int argc, ORTCHAR_T* argv[]);
};

}  // namespace test
}  // namespace onnxruntime
