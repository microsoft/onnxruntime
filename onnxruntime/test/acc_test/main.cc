// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdlib>  // std::abort
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#include "cmd_args.h"
#include "model_io_utils.h"
#include "data_loader.h"
#include "acc_task.h"
#include "task_thread_pool.h"

static std::vector<std::filesystem::path> GetSortedDatasetPaths(const std::filesystem::path& model_dir) {
  std::vector<std::filesystem::path> dataset_paths;
  const char* dataset_prefix = "test_data_set_";

  for (const auto& entry : std::filesystem::directory_iterator{model_dir}) {
    std::filesystem::path entry_path = entry.path();
    std::string entry_filename = entry_path.filename().string();

    if (std::filesystem::is_directory(entry_path) && entry_filename.rfind(dataset_prefix, 0) == 0) {
      dataset_paths.push_back(std::move(entry_path));
    }
  }

  auto cmp_indexed_paths = [dataset_prefix](const std::filesystem::path& a,
                                            const std::filesystem::path& b) -> bool {
    const int64_t a_index = GetFileIndexSuffix(a.filename().string(), dataset_prefix);
    const int64_t b_index = GetFileIndexSuffix(b.filename().string(), dataset_prefix);
    return a_index < b_index;
  };

  std::sort(dataset_paths.begin(), dataset_paths.end(), cmp_indexed_paths);

  return dataset_paths;
}

static bool GetExpectedOutputsFromModel(Ort::Env& env,
                                        TaskThreadPool& pool,
                                        const AppArgs& args,
                                        const std::filesystem::path& model_path,
                                        const std::vector<std::filesystem::path>& dataset_paths,
                                        std::vector<std::unique_ptr<char[]>>& all_inputs,
                                        std::vector<std::unique_ptr<char[]>>& all_outputs) {
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  Ort::Session f32_cpu_sess(env, model_path.c_str(), session_options);
  ModelIOInfo model_io_info;

  if (!ModelIOInfo::Init(model_io_info, f32_cpu_sess.GetConst())) {
    std::cerr << "[ERROR]: Failed to query model I/O information." << std::endl;
    return false;
  }

  if (!acctest::LoadIODataFromDisk(dataset_paths, model_io_info.inputs, "input_", all_inputs)) {
    std::cerr << "[ERROR]: Failed to load test inputs for model directory " << model_path.parent_path() << std::endl;
    return false;
  }

  const size_t num_datasets = dataset_paths.size();
  std::vector<Task> tasks;
  tasks.reserve(num_datasets);

  const size_t total_input_data_size = model_io_info.GetTotalInputSize();
  const size_t total_output_data_size = model_io_info.GetTotalOutputSize();

  all_outputs.reserve(num_datasets);

  for (size_t i = 0; i < num_datasets; i++) {
    all_outputs.emplace_back(std::make_unique<char[]>(total_output_data_size));

    Task task = Task::CreateInferenceTask(f32_cpu_sess, model_io_info,
                                          Span<const char>(all_inputs[i].get(), total_input_data_size),
                                          Span<char>(all_outputs.back().get(), total_output_data_size));
    tasks.push_back(std::move(task));
  }

  pool.CompleteTasks(tasks);

  if (args.save_expected_outputs_to_disk) {
    // Write outputs to disk: output_0.raw, output_1.raw, ...
    for (size_t dataset_index = 0; dataset_index < num_datasets; dataset_index++) {
      const std::filesystem::path& dataset_dir = dataset_paths[dataset_index];
      Span<const char> dataset_output(all_outputs[dataset_index].get(), total_output_data_size);
      const std::vector<IOInfo>& output_infos = model_io_info.outputs;
      const size_t num_outputs = output_infos.size();

      for (size_t buf_offset = 0, i = 0; i < num_outputs; buf_offset += output_infos[i].total_data_size, i++) {
        std::ostringstream oss;
        oss << "output_" << i << ".raw";

        std::filesystem::path output_filepath = dataset_dir / oss.str();
        std::ofstream ofs(output_filepath, std::ios::binary);

        assert(buf_offset < dataset_output.size());
        ofs.write(&dataset_output[buf_offset], output_infos[i].total_data_size);
      }
    }
  }
  return true;
}

static bool RunTestModel(Ort::Env& env,
                         TaskThreadPool& pool,
                         const std::filesystem::path& model_path,
                         const std::vector<std::filesystem::path>& dataset_paths,
                         const Ort::SessionOptions& session_options,
                         std::vector<std::unique_ptr<char[]>>& all_inputs,
                         std::vector<std::unique_ptr<char[]>>& all_outputs,
                         std::vector<std::vector<AccMetrics>>& test_accuracy_results) {
  Ort::Session session(env, model_path.c_str(), session_options);
  ModelIOInfo model_io_info;

  if (!ModelIOInfo::Init(model_io_info, session.GetConst())) {
    std::cerr << "[ERROR]: Failed to query model I/O information "
              << "for model " << model_path << std::endl;
    return false;
  }

  const size_t num_datasets = dataset_paths.size();

  if (all_inputs.empty()) {
    if (!acctest::LoadIODataFromDisk(dataset_paths, model_io_info.inputs, "input_", all_inputs)) {
      std::cerr << "[ERROR]: Failed to load test inputs for model directory "
                << model_path.parent_path() << std::endl;
      return false;
    }
  }

  if (all_outputs.empty()) {
    if (!acctest::LoadIODataFromDisk(dataset_paths, model_io_info.outputs, "output_", all_outputs)) {
      std::cerr << "[ERROR]: Failed to load test outputs for model directory "
                << model_path.parent_path() << std::endl;
      return false;
    }
  }

  assert(all_inputs.size() == num_datasets);
  assert(all_outputs.size() == num_datasets);

  std::vector<Task> tasks;
  tasks.reserve(num_datasets);

  test_accuracy_results.resize(num_datasets, std::vector<AccMetrics>(model_io_info.outputs.size()));

  const size_t total_input_data_size = model_io_info.GetTotalInputSize();
  const size_t total_output_data_size = model_io_info.GetTotalOutputSize();

  for (size_t i = 0; i < num_datasets; i++) {
    Task task = Task::CreateAccuracyCheckTask(session, model_io_info,
                                              Span<const char>(all_inputs[i].get(), total_input_data_size),
                                              Span<const char>(all_outputs[i].get(), total_output_data_size),
                                              Span<AccMetrics>(test_accuracy_results[i]));
    tasks.push_back(std::move(task));
  }

  pool.CompleteTasks(tasks);
  return true;
}

static void PrintAccuracyResults(const std::vector<std::vector<AccMetrics>>& test_accuracy_results,
                                 const std::vector<std::filesystem::path>& dataset_paths,
                                 const std::filesystem::directory_entry& model_dir,
                                 const std::string& output_file,
                                 std::unordered_map<std::string, size_t>& test_name_to_acc_result_index) {
  assert(test_accuracy_results.size() == dataset_paths.size());
  std::ostringstream oss;
  for (size_t i = 0; i < test_accuracy_results.size(); i++) {
    const std::filesystem::path& test_path = dataset_paths[i];
    const std::vector<AccMetrics>& metrics = test_accuracy_results[i];
    std::string key = model_dir.path().filename().string() + "/" + test_path.filename().string();
    test_name_to_acc_result_index[key] = i;

    oss << key << ",";
    for (size_t j = 0; j < metrics.size(); j++) {
      oss << std::setprecision(std::numeric_limits<double>::max_digits10) << metrics[j].snr;
      if (j < metrics.size() - 1) {
        oss << ",";
      }
    }
    oss << std::endl;
  }

  if (output_file.empty()) {
    std::cout << std::endl;
    std::cout << "Accuracy Results:" << std::endl;
    std::cout << "=================" << std::endl;
    std::cout << oss.str() << std::endl;
  } else {
    std::ofstream out_fs(output_file);
    out_fs << oss.str();
    out_fs.close();
  }
}

static bool CompareAccuracyWithExpectedValues(const std::filesystem::path& expected_accuracy_file,
                                              const std::vector<std::vector<AccMetrics>>& test_accuracy_results,
                                              const std::unordered_map<std::string, size_t>& test_name_to_acc_result_index,
                                              size_t& total_tests,
                                              size_t& total_failed_tests) {
  std::cout << std::endl;
  std::cout << "[INFO]: Comparing accuracy with " << expected_accuracy_file.filename().string() << std::endl;
  std::cout << "===============================================" << std::endl;
  std::ifstream in_fs(expected_accuracy_file);
  constexpr size_t N = 512;
  std::array<char, N> tmp_buf = {};

  while (in_fs.getline(&tmp_buf[0], tmp_buf.size())) {
    std::istringstream iss(tmp_buf.data());
    if (!iss.getline(&tmp_buf[0], tmp_buf.size(), ',')) {
      std::cerr << "[ERROR]: Failed to parse expected accuracy file " << expected_accuracy_file << std::endl;
      return false;
    }

    std::string key(tmp_buf.data());
    auto it = test_name_to_acc_result_index.find(key);
    if (it == test_name_to_acc_result_index.end()) {
      std::cerr << "[ERROR]: " << key << " was not a test that was run.";
      return false;
    }

    std::vector<double> expected_values;
    while (iss.getline(&tmp_buf[0], tmp_buf.size(), ',')) {
      expected_values.push_back(std::stod(tmp_buf.data()));
    }

    const std::vector<AccMetrics>& actual_output_metrics = test_accuracy_results[it->second];
    if (actual_output_metrics.size() != expected_values.size()) {
      std::cerr << "[ERROR]: test " << key << " does not have the expected number of outputs.";
      return false;
    }

    std::ostringstream oss;
    bool passed = true;
    for (size_t i = 0; i < expected_values.size(); i++) {
      const auto& metrics = actual_output_metrics[i];

      if (!(expected_values[i] - metrics.snr <= EPSILON_DBL)) {
        passed = false;
        oss << "\tOutput " << i << " SNR decreased: expected "
            << std::setprecision(std::numeric_limits<double>::max_digits10) << expected_values[i] << ", actual "
            << metrics.snr << std::endl;
      }
    }

    std::cout << "[INFO]: Checking if " << key << " degraded ... ";
    if (passed) {
      std::cout << "PASSED" << std::endl;
    } else {
      std::cout << "FAILED" << std::endl;
      std::cout << oss.str() << std::endl;
      total_failed_tests += 1;
    }
    total_tests += 1;
  }

  return true;
}

int main(int argc, char** argv) {
  try {
    AppArgs args;
    if (!ParseCmdLineArgs(args, argc, argv)) {
      return 1;
    }

    Ort::Env env;

    constexpr size_t num_pool_threads = 3;
    TaskThreadPool pool(num_pool_threads);
    TaskThreadPool dummy_pool(0);  // For EPs that only support single-threaded inference (e.g., QNN).
    size_t total_tests = 0;
    size_t total_failed_tests = 0;

    for (const std::filesystem::directory_entry& model_dir : std::filesystem::directory_iterator{args.test_dir}) {
      const std::filesystem::path& model_dir_path = model_dir.path();
      const std::vector<std::filesystem::path> dataset_paths = GetSortedDatasetPaths(model_dir_path);

      if (dataset_paths.empty()) {
        continue;  // Nothing to test.
      }

      std::filesystem::path base_model_path = model_dir_path / "model.onnx";
      std::filesystem::path ep_model_path;

      // Some EPs will need to use a QDQ model instead of the the original model.
      if (args.uses_qdq_model) {
        std::filesystem::path qdq_model_path = model_dir_path / "model.qdq.onnx";

        if (!std::filesystem::is_regular_file(qdq_model_path)) {
          std::cerr << "[ERROR]: Execution provider '" << args.execution_provider
                    << "' requires a QDQ model." << std::endl;
          return 1;
        }
        ep_model_path = std::move(qdq_model_path);
      } else {
        ep_model_path = base_model_path;
      }

      std::vector<std::unique_ptr<char[]>> all_inputs;
      std::vector<std::unique_ptr<char[]>> all_outputs;

      // Load expected outputs from base model running on CPU EP (unless user wants to use outputs from disk).
      if (!args.load_expected_outputs_from_disk) {
        if (!std::filesystem::is_regular_file(base_model_path)) {
          std::cerr << "[ERROR]: Cannot find ONNX model " << base_model_path
                    << " from which to get expected outputs." << std::endl;
          return 1;
        }

        if (!GetExpectedOutputsFromModel(env, pool, args, base_model_path, dataset_paths, all_inputs, all_outputs)) {
          return 1;
        }
      }

      // Run accuracy measurements with the EP under test.
      std::vector<std::vector<AccMetrics>> test_accuracy_results;
      TaskThreadPool& ep_pool = args.supports_multithread_inference ? pool : dummy_pool;
      if (!RunTestModel(env, ep_pool, ep_model_path, dataset_paths, args.session_options,
                        all_inputs, all_outputs, test_accuracy_results)) {
        return 1;
      }

      // Print the accuracy results to file or stdout.
      std::unordered_map<std::string, size_t> test_name_to_acc_result_index;
      PrintAccuracyResults(test_accuracy_results,
                           dataset_paths,
                           model_dir,
                           args.output_file,
                           test_name_to_acc_result_index);

      if (!args.expected_accuracy_file.empty()) {
        if (!CompareAccuracyWithExpectedValues(args.expected_accuracy_file, test_accuracy_results,
                                               test_name_to_acc_result_index, total_tests, total_failed_tests)) {
          return 1;
        }
      }
    }

    if (!args.expected_accuracy_file.empty()) {
      const size_t total_tests_passed = total_tests - total_failed_tests;
      std::cout << std::endl
                << "[INFO]: " << total_tests_passed << "/" << total_tests << " tests passed." << std::endl
                << "[INFO]: " << total_failed_tests << "/" << total_tests << " tests failed." << std::endl;
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "[ORT_QNN_APP EXCEPTION]: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
