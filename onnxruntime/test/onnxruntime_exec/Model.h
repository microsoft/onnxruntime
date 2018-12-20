// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#if !defined(_MSC_VER)
// HRESULT is a 4-byte long on MSVC.  We'll just make it a signed int here.
typedef int HRESULT;
// Success codes
#define S_OK ((HRESULT)0L)
#define S_FALSE ((HRESULT)1L)
#endif

#include "Runtime.h"

enum class ExecutionStatus {
  OK = 0,
  MODEL_LOADING_FAILURE = 1,
  DATA_LOADING_FAILURE = 2,
  PREDICTION_FAILURE = 3,
  ORT_NOT_IMPLEMENTED = 5
};

class Model {
 public:
  Model(const std::string& modelfile) {
    runtime_ = std::make_unique<WinMLRuntime>();
    LoadModel(modelfile);
  }

  void Execute(const std::string& datafile) {
    struct stat s;
    if (stat(datafile.c_str(), &s) == 0) {
      if (s.st_mode & S_IFDIR) {
        exec_status_ = ExecutionStatus::ORT_NOT_IMPLEMENTED;
        return;
      }
    }

    auto input_reader = LoadTestFile(datafile);
    if (!input_reader) {
      exec_status_ = ExecutionStatus::DATA_LOADING_FAILURE;
      return;
    }

    int sample = 0;
    while (!input_reader->Eof()) {
      std::map<std::string, std::vector<float>> outputs;

      // Perform the test
      int hr = runtime_->Run(*input_reader);
      if (hr != 0) {
        std::cerr << "Failed to execute example" << std::endl;
        exec_status_ = ExecutionStatus::PREDICTION_FAILURE;
        return;
      }
      sample++;
    }
  }

  ExecutionStatus GetStatus() const {
    return exec_status_;
  }

  std::string GetStatusString() const {
    return GetStatusString(exec_status_);
  }

  static std::string GetStatusString(ExecutionStatus exec_status) {
    switch (exec_status) {
      case ExecutionStatus::OK:
        return "OK";
      case ExecutionStatus::MODEL_LOADING_FAILURE:
        return "MODEL_LOADING_FAILURE";
      case ExecutionStatus::DATA_LOADING_FAILURE:
        return "DATA_LOADING_FAILURE";
      case ExecutionStatus::PREDICTION_FAILURE:
        return "PREDICTION_FAILURE";
      default:
        return "UNKNOWN";
    }
  }

 private:
  void LoadModel(const std::string& strfilepath) {
    std::wstring filepath(strfilepath.begin(), strfilepath.end());

    auto status = runtime_->LoadModel(filepath);
    if (status.IsOK()) {
      std::cerr << "'" << strfilepath.c_str() << "' loaded successfully." << std::endl;
    } else {
      std::cerr << "Loading failed for '" << strfilepath.c_str() << "'" << std::endl;
      std::cerr << "-----------------------------" << std::endl;
      std::cerr << status.ErrorMessage() << std::endl;
      exec_status_ = ExecutionStatus::MODEL_LOADING_FAILURE;
    }
  }

  std::unique_ptr<TestDataReader> LoadTestFile(const std::string& filepath) {
    std::wstring testfilepath(filepath.begin(), filepath.end());
    auto reader = TestDataReader::OpenReader(testfilepath);

    if (!reader) {
      std::cerr << "Unable to load test data file " << filepath << std::endl;
    }

    return reader;
  }

  std::unique_ptr<WinMLRuntime> runtime_;
  ExecutionStatus exec_status_ = ExecutionStatus::OK;
};
