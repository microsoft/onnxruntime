// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>

class QNNTestEnvironment {
 public:
  // Constructor takes argc and argv directly
  explicit QNNTestEnvironment(int argc, char** argv) {
    ParseCommandLineFlags(argc, argv);
  }

  bool dump_onnx() const { return dump_onnx_; }
  bool dump_json() const { return dump_json_; }
  bool dump_dlc() const { return dump_dlc_; }
  bool verbose() const { return verbose_; }

 public:
  std::filesystem::path CreateTestcaseDirs() {
    std::string test_suite_name = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::path output_dir = std::filesystem::current_path() / (test_suite_name + "_" + test_name);
    std::filesystem::create_directories(output_dir);

    return output_dir;
  }

 private:
  void ParseCommandLineFlags(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--dump_onnx") {
        std::cout << "[QNN only] ONNX model dumping enabled." << std::endl;
        dump_onnx_ = true;
      } else if (arg == "--dump_json") {
        std::cout << "[QNN only] Json QNN Graph dumping enabled." << std::endl;
        dump_json_ = true;
      } else if (arg == "--dump_dlc") {
        std::cout << "[QNN only] DLC dumping enabled." << std::endl;
        dump_dlc_ = true;
      } else if (arg == "--verbose") {
        std::cout << "Verbose enabled" << std::endl;
        verbose_ = true;
      }
    }
  }

  bool dump_onnx_ = false;
  bool dump_json_ = false;
  bool dump_dlc_ = false;
  bool verbose_ = false;
};
