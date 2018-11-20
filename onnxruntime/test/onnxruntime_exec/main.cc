// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "CmdParser.h"
#include "Model.h"
#include "TestDataReader.h"

void print_cmd_option() {
  std::cerr << "onnxruntime_exec.exe -m model_file [-t testdata]" << std::endl;
}

int main(int argc, const char* args[]) {
  try {
    CmdParser parser(argc, args);
    const std::string* modelfile = parser.GetCommandArg("-m");
    if (!modelfile) {
      std::cerr << "WinML model file is required." << std::endl;
      print_cmd_option();
      return -1;
    }

    Model model(*modelfile);

    if (model.GetStatus() == ExecutionStatus::OK) {
      std::cerr << "Done loading model: " << modelfile->c_str() << std::endl;
      const std::string* testfile = parser.GetCommandArg("-t");
      if (testfile) {
        model.Execute(*testfile);
      }
    }

    std::cerr << "Execution Status: " << model.GetStatusString() << std::endl;
  } catch (const DataValidationException& e) {
    std::cerr << "Execution Status: " << Model::GetStatusString(ExecutionStatus::DATA_LOADING_FAILURE) << std::endl;
    std::cout << "Exception msg: " << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Execution Status: " << Model::GetStatusString(ExecutionStatus::PREDICTION_FAILURE) << std::endl;
    std::cout << "Exception msg: " << e.what() << std::endl;
  }
}
