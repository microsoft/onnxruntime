// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "src/mutator.h"
#include "testlog.h"
#include "OnnxPrediction.h"
#include "onnxruntime_session_options_config_keys.h"

#include <type_traits>

using user_options = struct
{
  bool write_model;
  bool verbose;
  bool stress;
  bool is_ort;
};

void predict(onnx::ModelProto& model_proto, unsigned int seed, Ort::Env& env) {
  // Create object for prediction
  //
  OnnxPrediction predict(model_proto, env);

  // Give predict a function to generate the data
  // to run prediction on.
  //
  predict.SetupInput(GenerateDataForInputTypeTensor, seed);

  // Run the prediction on the data
  //
  predict.RunInference();

  // View the output
  //
  predict.PrintOutputValues();
}

void mutateModelTest(onnx::ModelProto& model_proto,
                     std::wstring mutatedModelDirName,
                     user_options opt,
                     Ort::Env& env,
                     unsigned int seed = 0) {
  // Used to initialize all random engines
  //
  std::wstring modelPrefix = L"/ReproMutateModel_";
  if (seed == 0) {
    seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    modelPrefix = L"/MutateModel_";
  }

  if (opt.stress) {
    Logger::testLog.enable();
  }

  Logger::testLog << L"Mutate test seed: " << seed << Logger::endl;
  opt.stress ? Logger::testLog.disable() : Logger::testLog.enable();

  // Create mutator
  //
  protobuf_mutator::Mutator mutator;

  // Mutate model
  //
  Logger::testLog << L"Model Successfully Initialized" << Logger::endl;
  mutator.Seed(seed);
  mutator.Mutate(&model_proto, model_proto.ByteSizeLong());

  if (opt.write_model) {
    // Create file to store model
    //
    std::wstringstream mutateModelName;
    mutateModelName << mutatedModelDirName << modelPrefix << seed << L".onnx";
    auto mutateModelFileName = mutateModelName.str();

    // Log the model to a file
    //
    std::ofstream outStream(mutateModelFileName);
    model_proto.SerializeToOstream(&outStream);
    Logger::testLog << "Mutated Model Written to file: " << mutateModelFileName << Logger::endl;

    // Flush the buffer to ensure the
    // mutated model info for reproduction
    // purposes.
    //
    outStream << std::flush;
  }

  // Flush any logs before prediction
  //
  Logger::testLog.flush();

  // run prediction on model
  //
  predict(model_proto, seed, env);

  // print out all output before next test
  //
  Logger::testLog.flush();
}

void printUsage() {
  std::cout << "Not enough command line arguments\n";
  std::cout << "usage:\n"
            << "\t\tFor testing: test.exe /t [options] onnx_model_file test_timeout test_time_scale\n"
            << "\t\tFor repro/debugging: test.exe /r onnx_model_file seed\n"
            << "\n\nonnx_model_file: Unmutated onnx model file\n"
            << "options: /m - output mutated models /v - verbose logging /s - stress test"
            << "test_time_scale: h|m|s\n"
            << "test_timeout: Time to run the test in hrs\n"
            << "seed: The seed that generated the mutated model. This value is the decimal digit part of the mutated model name (or can be found in the logs)\n"
            << "\n";
}

enum class timeScale : char {
  Hrs = 'h',
  Min = 'm',
  Sec = 's'
};

struct runtimeOpt {
  std::wstring model_file_name{};
  std::wstring mutate_model_dir_name{};
  Logger::ccstream err_stream_buf{};
  Logger::wcstream werr_stream_buf{};
  bool repo_mode{false};
  int test_time_out{0};
  unsigned int seed{0};
  timeScale scale{timeScale::Sec};
  user_options user_opt{false, false, false, false};
};

int processCommandLine(int argc, char* argv[], runtimeOpt& opt) {
  if (argc <= 1) {
    printUsage();
    return 2;
  } else {
    bool isTest = std::string{argv[1]} == "/t";
    bool isRepo = std::string{argv[1]} == "/r";

    if (isRepo) {
      opt.repo_mode = true;
      opt.mutate_model_dir_name = L"./repromodel";
      std::filesystem::path mutate_model_dir{opt.mutate_model_dir_name};
      if (!std::filesystem::exists(mutate_model_dir)) {
        std::filesystem::create_directory(mutate_model_dir);
      }

      opt.model_file_name = Logger::towstr(argv[2]);
      Logger::testLog << L"Repo Model file: " << opt.model_file_name << Logger::endl;

      // Get seed
      //
      std::stringstream parser{argv[3]};
      parser >> opt.seed;
      if (parser.bad()) {
        throw std::exception("Could not parse seed from command line");
      }

      std::wcout << L"seed: " << opt.seed << L"\n";
    } else if (isTest) {
      int index{argc};
      index--;

      // Parse right to left
      //
      std::stringstream parser;
      char desired_scale;
      parser << argv[index--];
      parser >> desired_scale;

      if (parser.bad()) {
        throw std::exception("Could not parse the time scale from the command line");
      }

      opt.scale = static_cast<timeScale>(std::tolower(desired_scale));
      switch (opt.scale) {
        case timeScale::Hrs:
        case timeScale::Min:
        case timeScale::Sec:
          break;
        default:
          throw std::exception("Could not parse the time scale from the command line");
      }

      parser << argv[index--];
      parser >> opt.test_time_out;
      if (parser.bad()) {
        throw std::exception("Could not parse the time value from the command line");
      }

      Logger::testLog << L"Running Test for: " << opt.test_time_out << desired_scale << Logger::endl;
      opt.model_file_name = Logger::towstr(argv[index--]);
      Logger::testLog << L"Model file: " << opt.model_file_name << Logger::endl;
      std::filesystem::path model_file_namePath{opt.model_file_name};
      if (!std::filesystem::exists(model_file_namePath)) {
        throw std::exception("Cannot find model file");
      }

      // process options
      //
      while (index > 0) {
        auto option{std::string{argv[index]}};
        if (option == "/m") {
          opt.user_opt.write_model = true;
        } else if (option == "/v") {
          opt.user_opt.verbose = true;
        } else if (option == "/s") {
          opt.user_opt.stress = true;
        } else if (option == "/f") {
          opt.user_opt.is_ort = true;
        }
        index--;
      }

      if (opt.user_opt.stress) {
        std::cerr.rdbuf(&opt.err_stream_buf);
        std::wcerr.rdbuf(&opt.werr_stream_buf);
        opt.user_opt.write_model = false;
        opt.user_opt.verbose = false;
        Logger::testLog.disable();
        Logger::testLog.minLog();
      }

      // create directory for mutated model output
      //
      if (opt.user_opt.write_model) {
        opt.mutate_model_dir_name = L"./mutatemodel";
        std::filesystem::path mutate_model_dir{opt.mutate_model_dir_name};
        if (!std::filesystem::exists(mutate_model_dir)) {
          std::filesystem::create_directory(mutate_model_dir);
        }
      }
    } else {
      printUsage();
      return 2;
    }
  }

  return 0;
}

struct RunStats {
  size_t num_ort_exception;
  size_t num_std_exception;
  size_t num_unknown_exception;
  size_t num_successful_runs;
  size_t iteration;
  int status;
};

static void fuzz_handle_exception(struct RunStats& run_stats) {
  try {
    throw;
  } catch (const Ort::Exception& ortException) {
    run_stats.num_ort_exception++;
    Logger::testLog << L"onnx runtime exception: " << ortException.what() << Logger::endl;
    Logger::testLog << "Failed Test iteration: " << run_stats.iteration++ << Logger::endl;
  } catch (const std::exception& e) {
    run_stats.num_std_exception++;
    Logger::testLog << L"standard exception: " << e.what() << Logger::endl;
    Logger::testLog << "Failed Test iteration: " << run_stats.iteration++ << Logger::endl;
    run_stats.status = 1;
  } catch (...) {
    run_stats.num_unknown_exception++;
    Logger::testLog << L"unknown exception: " << Logger::endl;
    Logger::testLog << "Failed Test iteration: " << run_stats.iteration++ << Logger::endl;
    run_stats.status = 1;
    throw;
  }
}

int main(int argc, char* argv[]) {
  Ort::Env env;
  // Enable telemetry events
  //
  env.EnableTelemetryEvents();
  struct RunStats run_stats {};
  runtimeOpt opt{};
  user_options& user_opt{opt.user_opt};
  Logger::wcstream& werr_stream_buf{opt.werr_stream_buf};
  try {
    // Initialize the runtime options
    //
    auto canContinue{processCommandLine(argc, argv, opt) == 0};

    if (!canContinue) {
      return -1;
    }

    std::wstring& model_file_name{opt.model_file_name};
    std::wstring& mutate_model_dir_name{opt.mutate_model_dir_name};
    bool& repo_mode{opt.repo_mode};
    int& test_time_out{opt.test_time_out};
    unsigned int& seed{opt.seed};
    timeScale& scale{opt.scale};

    // Model file
    //
    std::wstring model_file{model_file_name};

    // Create a stream to hold the model
    //
    std::ifstream modelStream{model_file, std::ios::in | std::ios::binary};
    if (opt.user_opt.is_ort == false) {
      // Create an onnx protobuf object
      //
      onnx::ModelProto model_proto;

      // Initialize the model
      //
      if (model_proto.ParseFromIstream(&modelStream)) {
        if (repo_mode) {
          Logger::testLog << L"Running Prediction for: " << model_file_name
                          << L" with seed " << seed << Logger::endl;
          mutateModelTest(model_proto, mutate_model_dir_name, user_opt, env, seed);
          Logger::testLog << L"Finished Prediction for: " << model_file_name
                          << L" with seed " << seed << Logger::endl;
          return 0;
        } else {
          // Call the mutateModelTest
          //
          std::chrono::system_clock::time_point curr_time{std::chrono::system_clock::now()};

          std::chrono::minutes time_in_min{test_time_out};
          std::chrono::seconds time_in_sec{test_time_out};
          std::chrono::hours time_in_hrs{test_time_out};
          std::chrono::system_clock::time_point end_time{curr_time};
          end_time += scale == timeScale::Hrs   ? time_in_hrs
                      : scale == timeScale::Min ? time_in_min
                                                : time_in_sec;
          Logger::testLog << "Starting Test" << Logger::endl;
          while (curr_time < end_time) {
            try {
              onnx::ModelProto bad_model = model_proto;
              Logger::testLog << "Starting Test iteration: " << run_stats.iteration << Logger::endl;
              mutateModelTest(bad_model, mutate_model_dir_name, user_opt, env);
              run_stats.num_successful_runs++;
              Logger::testLog << "Completed Test iteration: " << run_stats.iteration++ << Logger::endl;
            } catch (...) {
              fuzz_handle_exception(run_stats);
            }
            // Update current time
            //
            curr_time = std::chrono::system_clock::now();
          }
        }
      } else {
        throw std::exception("Unable to initialize the Onnx model in memory");
      }
    } else {
      std::wstring ort_model_file = model_file;
      if (model_file.substr(model_file.find_last_of(L".") + 1) == L"onnx") {
        ort_model_file = model_file + L".ort";
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
        so.SetOptimizedModelFilePath(ort_model_file.c_str());
        so.AddConfigEntry(kOrtSessionOptionsConfigSaveModelFormat, "ORT");
        Ort::Session session(env, model_file.c_str(), so);
      } else if (model_file.substr(model_file.find_last_of(L".") + 1) != L"ort") {
        Logger::testLog << L"Input file name extension is not 'onnx' or 'ort' " << Logger::endl;
        return 1;
      }
      size_t num_bytes = std::filesystem::file_size(ort_model_file);
      std::vector<char> model_data(num_bytes);
      std::ifstream ortModelStream(ort_model_file, std::ifstream::in | std::ifstream::binary);
      ortModelStream.read(model_data.data(), num_bytes);
      ortModelStream.close();
      // Currently mutations are generated by using XOR of a byte with the preceeding byte at a time.
      // Other possible ways may be considered in future, for example swaping two bytes randomly at a time.
      Logger::testLog << "Starting Test" << Logger::endl;
      for (size_t& i = run_stats.iteration; i < num_bytes - 1; i++) {
        char tmp = model_data[i];
        model_data[i] ^= model_data[i + 1];
        try {
          Logger::testLog << "Starting Test iteration: " << i << Logger::endl;
          OnnxPrediction predict(model_data, env);
          predict.SetupInput(GenerateDataForInputTypeTensor, 0);
          predict.RunInference();
          run_stats.num_successful_runs++;
          Logger::testLog << "Completed Test iteration: " << i << Logger::endl;
        } catch (...) {
          fuzz_handle_exception(run_stats);
        }
        model_data[i] = tmp;
      }
    }
    Logger::testLog << "Ending Test" << Logger::endl;

    if (user_opt.stress) {
      Logger::testLog.enable();
    }
    size_t toal_num_exception = run_stats.num_unknown_exception + run_stats.num_std_exception + run_stats.num_ort_exception;
    Logger::testLog << L"Total number of exceptions: " << toal_num_exception << Logger::endl;
    Logger::testLog << L"Number of Unknown exceptions: " << run_stats.num_unknown_exception << Logger::endl;
    Logger::testLog << L"Number of ort exceptions: " << run_stats.num_ort_exception << Logger::endl;
    Logger::testLog << L"Number of std exceptions: " << run_stats.num_std_exception << Logger::endl;
    Logger::testLog << L"Number of unique errors: " << werr_stream_buf.get_unique_errors() << L"\n";

    if (user_opt.stress) {
      Logger::testLog.disable();
      Logger::testLog.flush();
    }
    return 0;
  } catch (const Ort::Exception& ort_exception) {
    Logger::testLog << L"onnx runtime exception: " << ort_exception.what() << Logger::endl;
  } catch (const std::exception& e) {
    Logger::testLog << L"standard exception: " << e.what() << Logger::endl;
    run_stats.status = 1;
  } catch (...) {
    Logger::testLog << L"Something Went very wrong: " << Logger::endl;
    run_stats.status = 1;
  }

  return run_stats.status;
}
