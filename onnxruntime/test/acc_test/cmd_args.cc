#include "cmd_args.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <cassert>
#include <iostream>
#include <ostream>
#include <unordered_set>
#include <sstream>
#include <string_view>

struct CmdArgs {
  CmdArgs(int argc, char** argv) noexcept : argc_(argc), argv_(argv), index_(0) {}

  [[nodiscard]] bool HasNext() const { return index_ < argc_; }

  [[nodiscard]] std::string_view GetNext() {
    assert(HasNext());
    return argv_[index_++];
  }

  [[nodiscard]] std::string_view PeekNext() {
    assert(HasNext());
    return argv_[index_];
  }

 private:
  int argc_;
  char** argv_;
  int index_;
};

static void PrintUsage(std::ostream& stream, std::string_view prog_name) {
  stream << "Usage: " << prog_name << " [OPTIONS]"
         << std::endl;
  stream << "OPTIONS:" << std::endl;
  stream << "    -h/--help                   Print this help message" << std::endl;
  stream << "    -t/--test_dir               Path to test directory with models and inputs/outputs" << std::endl;
  stream << "    -l/--load_expected_outputs  Load expected outputs from raw output_<index>.raw files" << std::endl;
  stream << "    -s/--save_expected_outputs  Save outputs from baseline model on CPU EP to disk" << std::endl;
  stream << "    -e/--execution_provider     The execution provider to test (e.g., qnn)" << std::endl;
  stream << "    -o/--output_file            The output file into which to save accuracy results" << std::endl;
  stream << "    -a/--expected_accuracy_file The file containing expected accuracy results" << std::endl
         << std::endl;
}

static bool ParseQnnRuntimeOptions(std::string ep_config_string,
                                   std::unordered_map<std::string, std::string>& qnn_options) {
  std::istringstream ss(ep_config_string);
  std::string token;

  while (ss >> token) {
    if (token == "") {
      continue;
    }
    std::string_view token_sv(token);

    auto pos = token_sv.find("|");
    if (pos == std::string_view::npos || pos == 0 || pos == token_sv.length()) {
      std::cerr << "Use a '|' to separate the key and value for the run-time option you are trying to use." << std::endl;
      return false;
    }

    std::string_view key(token_sv.substr(0, pos));
    std::string_view value(token_sv.substr(pos + 1));

    if (key == "backend_path") {
      if (value.empty()) {
        std::cerr << "[ERROR]: Please provide the QNN backend path." << std::endl;
        return false;
      }
    } else if (key == "qnn_context_cache_enable") {
      if (value != "1") {
        std::cerr << "[ERROR]: Set to 1 to enable qnn_context_cache_enable." << std::endl;
        return false;
      }
    } else if (key == "qnn_context_cache_path") {
      // no validation
    } else if (key == "profiling_level") {
      std::unordered_set<std::string_view> supported_profiling_level = {"off", "basic", "detailed"};
      if (supported_profiling_level.find(value) == supported_profiling_level.end()) {
        std::cerr << "[ERROR]: Supported profiling_level: off, basic, detailed" << std::endl;
        return false;
      }
    } else if (key == "rpc_control_latency" || key == "vtcm_mb") {
      // no validation
    } else if (key == "htp_performance_mode") {
      std::unordered_set<std::string_view> supported_htp_perf_mode = {"burst", "balanced", "default", "high_performance",
                                                                      "high_power_saver", "low_balanced", "low_power_saver",
                                                                      "power_saver", "sustained_high_performance"};
      if (supported_htp_perf_mode.find(value) == supported_htp_perf_mode.end()) {
        std::ostringstream str_stream;
        std::copy(supported_htp_perf_mode.begin(), supported_htp_perf_mode.end(),
                  std::ostream_iterator<std::string_view>(str_stream, ","));
        std::string str = str_stream.str();
        std::cerr << "[ERROR]: Supported htp_performance_mode: " << str << std::endl;
        return false;
      }
    } else if (key == "qnn_saver_path") {
      // no validation
    } else if (key == "htp_graph_finalization_optimization_mode") {
      std::unordered_set<std::string_view> supported_htp_graph_final_opt_modes = {"0", "1", "2", "3"};
      if (supported_htp_graph_final_opt_modes.find(value) == supported_htp_graph_final_opt_modes.end()) {
        std::ostringstream str_stream;
        std::copy(supported_htp_graph_final_opt_modes.begin(), supported_htp_graph_final_opt_modes.end(),
                  std::ostream_iterator<std::string_view>(str_stream, ","));
        std::string str = str_stream.str();
        std::cerr << "[ERROR]: Wrong value for htp_graph_finalization_optimization_mode. select from: " << str << std::endl;
        return false;
      }
    } else if (key == "qnn_context_priority") {
      std::unordered_set<std::string_view> supported_qnn_context_priority = {"low", "normal", "normal_high", "high"};
      if (supported_qnn_context_priority.find(value) == supported_qnn_context_priority.end()) {
        std::cerr << "[ERROR]: Supported qnn_context_priority: low, normal, normal_high, high" << std::endl;
        return false;
      }
    } else {
      std::cerr << R"([ERROR]: Wrong key type entered. Choose from options: ['backend_path', 'qnn_context_cache_enable',
'qnn_context_cache_path', 'profiling_level', 'rpc_control_latency', 'vtcm_mb', 'htp_performance_mode',
'qnn_saver_path', 'htp_graph_finalization_optimization_mode', 'qnn_context_priority'])"
                << std::endl;
      return false;
    }

    qnn_options.insert(std::make_pair(std::string(key), std::string(value)));
  }

  return true;
}

static bool ParseQnnArgs(AppArgs& app_args, CmdArgs& cmd_args) {
  if (!cmd_args.HasNext()) {
    std::cerr << "[ERROR]: Must specify at least a QNN backend path." << std::endl;
    return false;
  }

  std::string_view args = cmd_args.GetNext();
  std::unordered_map<std::string, std::string> qnn_options;

  if (!ParseQnnRuntimeOptions(std::string(args), qnn_options)) {
    return false;
  }

  auto backend_iter = qnn_options.find("backend_path");
  if (backend_iter == qnn_options.end()) {
    std::cerr << "[ERROR]: Must provide a backend_path for the QNN execution provider." << std::endl;
    return false;
  }

  app_args.session_options.AppendExecutionProvider("QNN", qnn_options);
  app_args.session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // TODO: Parse config entries
  app_args.uses_qdq_model = backend_iter->second.rfind("QnnHtp") != std::string::npos;
  app_args.supports_multithread_inference = false;  // TODO: Work on enabling multi-threaded inference.
  return true;
}

static bool GetValidPath(std::string_view prog_name, std::string_view provided_path, bool is_dir,
                         std::filesystem::path& valid_path) {
  std::filesystem::path path = provided_path;
  std::error_code error_code;

  if (!std::filesystem::exists(path, error_code)) {
    std::cerr << "[ERROR]: Invalid path " << provided_path << ": "
              << error_code.message() << std::endl
              << std::endl;
    return false;
  }

  std::error_code abs_error_code;
  std::filesystem::path abs_path = std::filesystem::absolute(path, abs_error_code);
  if (abs_error_code) {
    std::cerr << "[ERROR]: Invalid path: " << abs_error_code.message() << std::endl
              << std::endl;
    return false;
  }

  if (is_dir && !std::filesystem::is_directory(abs_path)) {
    std::cerr << "[ERROR]: " << provided_path << " is not a directory" << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  if (!is_dir && !std::filesystem::is_regular_file(abs_path)) {
    std::cerr << "[ERROR]: " << provided_path << " is not a regular file" << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  valid_path = std::move(abs_path);

  return true;
}

bool ParseCmdLineArgs(AppArgs& app_args, int argc, char** argv) {
  CmdArgs cmd_args(argc, argv);
  std::string_view prog_name = cmd_args.GetNext();

  // Parse command-line arguments.
  while (cmd_args.HasNext()) {
    std::string_view arg = cmd_args.GetNext();

    if (arg == "-h" || arg == "--help") {
      PrintUsage(std::cout, prog_name);
      return true;
    } else if (arg == "-t" || arg == "--test_dir") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      arg = cmd_args.GetNext();
      if (!GetValidPath(prog_name, arg, true, app_args.test_dir)) {
        return false;
      }
    } else if (arg == "-o" || arg == "--output_file") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      app_args.output_file = cmd_args.GetNext();
    } else if (arg == "-a" || arg == "--expected_accuracy_file") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      arg = cmd_args.GetNext();
      if (!GetValidPath(prog_name, arg, false, app_args.expected_accuracy_file)) {
        return false;
      }
    } else if (arg == "-e" || arg == "--execution_provider") {
      if (!cmd_args.HasNext()) {
        std::cerr << "[ERROR]: Must provide an argument after the " << arg << " option" << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      arg = cmd_args.GetNext();
      if (arg == "qnn") {
        if (!ParseQnnArgs(app_args, cmd_args)) {
          return false;
        }
      } else {
        std::cerr << "[ERROR]: Unsupported execution provider: " << arg << std::endl;
        PrintUsage(std::cerr, prog_name);
        return false;
      }

      app_args.execution_provider = arg;
    } else if (arg == "-s" || arg == "--save_expected_outputs") {
      app_args.save_expected_outputs_to_disk = true;
    } else if (arg == "-l" || arg == "--load_expected_outputs") {
      app_args.load_expected_outputs_from_disk = true;
    } else {
      std::cerr << "[ERROR]: unknown command-line argument `" << arg << "`" << std::endl
                << std::endl;
      PrintUsage(std::cerr, prog_name);
      return false;
    }
  }

  //
  // Final argument validation:
  //

  if (app_args.test_dir.empty()) {
    std::cerr << "[ERROR]: Must provide a test directory using the -t/--test_dir option." << std::endl
              << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  if (app_args.execution_provider.empty()) {
    std::cerr << "[ERROR]: Must provide an execution provider using the -e/--execution_provider option." << std::endl
              << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  if (app_args.load_expected_outputs_from_disk && app_args.save_expected_outputs_to_disk) {
    std::cerr << "[ERROR]: Cannot enable both -s/--save_expected_outputs and -l/--load_expected_outputs" << std::endl
              << std::endl;
    PrintUsage(std::cerr, prog_name);
    return false;
  }

  return true;
}