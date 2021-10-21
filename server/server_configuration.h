// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <thread>
#include <fstream>
#include <unordered_map>

#include "boost/program_options.hpp"
#include "onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace server {

namespace po = boost::program_options;

// Enumerates the different type of results which can occur
// The three different types are:
// 0. ExitSuccess which is when the program should exit with EXIT_SUCCESS
// 1. ExitFailure when program should exit with EXIT_FAILURE
// 2. No need for exiting the program, continue
enum class Result {
  ExitSuccess,
  ExitFailure,
  ContinueSuccess
};

static std::unordered_map<std::string, OrtLoggingLevel> supported_log_levels{
    {"verbose", ORT_LOGGING_LEVEL_VERBOSE},
    {"info", ORT_LOGGING_LEVEL_INFO},
    {"warning", ORT_LOGGING_LEVEL_WARNING},
    {"error", ORT_LOGGING_LEVEL_ERROR},
    {"fatal", ORT_LOGGING_LEVEL_FATAL}};

// Wrapper around Boost program_options and should provide all the functionality for options parsing
// Provides sane default values
class ServerConfiguration {
 public:
  const std::string full_desc = "ONNX Server: host an ONNX model with ONNX Runtime";
  std::string model_path;
  std::string model_name = "default";
  std::string model_version = "1";
  std::string address = "0.0.0.0";
  unsigned short http_port = 8001;
  unsigned short grpc_port = 50051;
  int num_http_threads = std::thread::hardware_concurrency();
  OrtLoggingLevel logging_level{};

  ServerConfiguration() {
    desc.add_options()("help,h", "Shows a help message and exits");
    desc.add_options()("log_level", po::value(&log_level_str)->default_value(log_level_str), "Logging level. Allowed options (case sensitive): verbose, info, warning, error, fatal");
    desc.add_options()("model_path", po::value(&model_path)->required(), "Path to ONNX model");
    desc.add_options()("model_name", po::value(&model_name)->default_value(model_name), "ONNX model name");
    desc.add_options()("model_version", po::value(&model_version)->default_value(model_version), "ONNX model version");
    desc.add_options()("address", po::value(&address)->default_value(address), "The base HTTP address");
    desc.add_options()("http_port", po::value(&http_port)->default_value(http_port), "HTTP port to listen to requests");
    desc.add_options()("num_http_threads", po::value(&num_http_threads)->default_value(num_http_threads), "Number of http threads");
    desc.add_options()("grpc_port", po::value(&grpc_port)->default_value(grpc_port), "GRPC port to listen to requests");
  }

  // Parses argc and argv and sets the values for the class
  // Returns an enum with three options: ExitSuccess, ExitFailure, ContinueSuccess
  // ExitSuccess and ExitFailure means the program should exit but is left to the caller
  Result ParseInput(int argc, char** argv) {
    try {
      po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);  // can throw

      if (ContainsHelp()) {
        PrintHelp(std::cout, full_desc);
        return Result::ExitSuccess;
      }

      po::notify(vm);  // throws on error, so do after help
    } catch (const po::error& e) {
      PrintHelp(std::cerr, e.what());
      return Result::ExitFailure;
    } catch (const std::exception& e) {
      PrintHelp(std::cerr, e.what());
      return Result::ExitFailure;
    }

    Result result = ValidateOptions();

    if (result == Result::ContinueSuccess) {
      logging_level = supported_log_levels[log_level_str];
    }

    return result;
  }

 private:
  po::options_description desc{"Allowed options"};
  po::variables_map vm{};
  std::string log_level_str = "info";

  // Print help and return if there is a bad value
  Result ValidateOptions() {
    if (vm.count("log_level") &&
        supported_log_levels.find(log_level_str) == supported_log_levels.end()) {
      PrintHelp(std::cerr, "log_level must be one of verbose, info, warning, error, or fatal");
      return Result::ExitFailure;
    } else if (num_http_threads <= 0) {
      PrintHelp(std::cerr, "num_http_threads must be greater than 0");
      return Result::ExitFailure;
    } else if (!file_exists(model_path)) {
      PrintHelp(std::cerr, "model_path must be the location of a valid file");
      return Result::ExitFailure;
    } else {
      return Result::ContinueSuccess;
    }
  }

  // Checks if program options contains help
  bool ContainsHelp() const {
    return vm.count("help") || vm.count("h");
  }

  // Prints a helpful message (param: what) to the user and then the program options
  // Example: config.PrintHelp(std::cout, "Non-negative values not allowed")
  // Which will print that message and then all publicly available options
  void PrintHelp(std::ostream& out, const std::string& what) const {
    out << what << std::endl
        << desc << std::endl;
  }

  inline bool file_exists(const std::string& fileName) {
    std::ifstream infile(fileName.c_str());
    return infile.good();
  }
};

}  // namespace server
}  // namespace onnxruntime
