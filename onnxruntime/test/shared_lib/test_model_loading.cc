// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"
#include "core/common/narrow.h"
#include "test/util/include/asserts.h"
#include <fstream>
#include "test_fixture.h"
#include "file_util.h"

#include "gmock/gmock.h"

#ifdef _WIN32
#include <wil/Resource.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "core/platform/scoped_resource.h"
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// disable for minimal build with no exceptions as it will always attempt to throw in that scenario
#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_NO_EXCEPTIONS)
TEST(CApiTest, model_from_array) {
  const char* model_path = "testdata/matmul_1.onnx";
  std::vector<char> buffer;
  {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file)
      ORT_THROW("Error reading model");
    buffer.resize(narrow<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), buffer.size()))
      ORT_THROW("Error reading model");
  }

#if (!ORT_MINIMAL_BUILD)
  bool should_throw = false;
#else
  bool should_throw = true;
#endif

  auto create_session = [&](Ort::SessionOptions& so) {
    try {
      Ort::Session session(*ort_env.get(), buffer.data(), buffer.size(), so);
      ASSERT_FALSE(should_throw) << "Creation of session should have thrown";
    } catch (const std::exception& ex) {
      ASSERT_TRUE(should_throw) << "Creation of session should not have thrown. Exception:" << ex.what();
      ASSERT_THAT(ex.what(), testing::HasSubstr("ONNX format model is not supported in this build."));
    }
  };

  Ort::SessionOptions so;
  create_session(so);

#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2* options;
  Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&options));
  so.AppendExecutionProvider_CUDA_V2(*options);
  create_session(so);
#endif
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
TEST(CApiTest, session_options_empty_affinity_string) {
  Ort::SessionOptions options;
  options.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, "");
  constexpr auto model_path = ORT_TSTR("testdata/matmul_1.onnx");

  try {
    Ort::Session session(*ort_env.get(), model_path, options);
    ASSERT_TRUE(false) << "Creation of session should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Affinity string must not be empty"));
  }
}
#endif

#endif

#ifdef DISABLE_EXTERNAL_INITIALIZERS
TEST(CApiTest, TestDisableExternalInitiliazers) {
  constexpr auto model_path = ORT_TSTR("testdata/model_with_external_initializers.onnx");

  Ort::SessionOptions so;
  try {
    Ort::Session session(*ort_env.get(), model_path, so);
    ASSERT_TRUE(false) << "Creation of session should have thrown exception";
  } catch (const std::exception& ex) {
    ASSERT_THAT(ex.what(), testing::HasSubstr("Initializer tensors with external data is not allowed."));
  }
}

#elif !defined(ORT_MINIMAL_BUILD)
TEST(CApiTest, TestExternalInitializersInjection) {
  constexpr auto model_path = ORT_TSTR("testdata/model_with_external_initializer_come_from_user.onnx");
  std::array<int64_t, 4> Pads_not_on_disk{0, 0, 1, 1};
  constexpr std::array<int64_t, 1> init_shape{4};

  const std::vector<std::string> init_names{"Pads_not_on_disk"};
  std::vector<Ort::Value> initializer_data;

  auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto init_tensor = Ort::Value::CreateTensor(cpu_mem_info, Pads_not_on_disk.data(), Pads_not_on_disk.size(), init_shape.data(), init_shape.size());
  initializer_data.push_back(std::move(init_tensor));

  Ort::SessionOptions so;
  const ORTCHAR_T* optimized_model_path = ORT_TSTR("testdata/model_with_external_initializer_come_from_user_opt.onnx");
  so.SetOptimizedModelFilePath(optimized_model_path);
  so.AddExternalInitializers(init_names, initializer_data);
  // Dump the optimized model with external data so that it will unpack the external data from the loaded model
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersFileName, "model_with_external_initializer_come_from_user_opt.bin");
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes, "10");
  EXPECT_NO_THROW(Ort::Session(*ort_env, model_path, so));
}

static void ReadFileToBuffer(const char* file_path, std::vector<char>& buffer) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file)
    ORT_THROW("Error reading file.");
  buffer.resize(narrow<size_t>(file.tellg()));
  file.seekg(0, std::ios::beg);
  if (!file.read(buffer.data(), buffer.size()))
    ORT_THROW("Error reading file");
}

void TestLoadModelFromArrayWithExternalInitializerFromFileArray(const std::string& model_file_name,
                                                                const std::string& external_data_file_name,
                                                                const std::string& external_ini_min_size_bytes = "10",
                                                                bool compare_external_bin_file = true) {
  std::string test_folder = "testdata/";
  std::string model_path = test_folder + model_file_name;
  std::vector<char> buffer;
  ReadFileToBuffer(model_path.c_str(), buffer);

  std::vector<char> external_bin_buffer;
  std::string external_bin_path = test_folder + external_data_file_name;
  ReadFileToBuffer(external_bin_path.c_str(), external_bin_buffer);

  Ort::SessionOptions so;
  std::string optimized_model_file_name(model_file_name);
  auto length = optimized_model_file_name.length();
  optimized_model_file_name.insert(length - 5, "_opt");
  std::string optimized_file_path(test_folder + optimized_model_file_name);
  PathString optimized_file_path_t(optimized_file_path.begin(), optimized_file_path.end());

  so.SetOptimizedModelFilePath(optimized_file_path_t.c_str());
  //  Dump the optimized model with external data so that it will unpack the external data from the loaded model
  std::string opt_bin_file_name(optimized_model_file_name);
  opt_bin_file_name.replace(optimized_model_file_name.length() - 4, 4, "bin");
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersFileName, opt_bin_file_name.c_str());
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes, external_ini_min_size_bytes.c_str());

  PathString external_file_name(external_data_file_name.begin(), external_data_file_name.end());
  std::vector<PathString> file_names{external_file_name};
  std::vector<char*> file_buffers{external_bin_buffer.data()};
  std::vector<size_t> lengths{external_bin_buffer.size()};
  so.AddExternalInitializersFromFilesInMemory(file_names, file_buffers, lengths);

  Ort::Session session(*ort_env.get(), buffer.data(), buffer.size(), so);

  std::string generated_bin_path = test_folder + opt_bin_file_name;
  // If there are multiple initializers in the external bin file
  // It's hard to guarantee the generated bin for optimized model is exactly same with original one for some cases
  if (compare_external_bin_file) {
    std::vector<char> generated_bin_buffer;
    ReadFileToBuffer(generated_bin_path.c_str(), generated_bin_buffer);

    ASSERT_EQ(external_bin_buffer, generated_bin_buffer);
  }

  // Cleanup.
  ASSERT_EQ(std::remove(optimized_file_path.c_str()), 0);
  ASSERT_EQ(std::remove(generated_bin_path.c_str()), 0);
}

// Single initializer from single bin file
TEST(CApiTest, TestLoadModelFromArrayWithExternalInitializerFromFileArray) {
  std::string model_file_name = "model_with_external_initializers.onnx";
  std::string external_bin_name = "Pads.bin";
  TestLoadModelFromArrayWithExternalInitializerFromFileArray(model_file_name, external_bin_name);
}

// Several external initializers from same file
// Use offset from tensor proto to locate the buffer location
TEST(CApiTest, TestLoadModelFromArrayWithExternalInitializersFromFileArray) {
  std::string model_file_name = "conv_qdq_external_ini.onnx";
  std::string external_bin_name = "conv_qdq_external_ini.bin";
  TestLoadModelFromArrayWithExternalInitializerFromFileArray(model_file_name, external_bin_name);
}

// Several external initializers from same file
// Use offset from tensor proto to locate the buffer location
TEST(CApiTest, TestLoadModelFromArrayWithExternalInitializersFromFileArrayPathRobust) {
  std::string model_file_name = "conv_qdq_external_ini.onnx";
  std::string external_bin_name = "./conv_qdq_external_ini.bin";
  TestLoadModelFromArrayWithExternalInitializerFromFileArray(model_file_name, external_bin_name);

  external_bin_name = ".//conv_qdq_external_ini.bin";
  TestLoadModelFromArrayWithExternalInitializerFromFileArray(model_file_name, external_bin_name);

#ifdef _WIN32
  external_bin_name = ".\\\\conv_qdq_external_ini.bin";
  TestLoadModelFromArrayWithExternalInitializerFromFileArray(model_file_name, external_bin_name);

  external_bin_name = ".\\conv_qdq_external_ini.bin";
  TestLoadModelFromArrayWithExternalInitializerFromFileArray(model_file_name, external_bin_name);
#endif
}

// The model has external data, Test loading model from array
// Extra API required to set the external data path
TEST(CApiTest, TestLoadModelFromArrayWithExternalInitializersViaSetExternalDataPath) {
  std::string model_file_name = "conv_qdq_external_ini.onnx";
  std::string external_bin_name = "conv_qdq_external_ini.bin";
  std::string test_folder = "testdata/";
  std::string model_path = test_folder + model_file_name;
  std::vector<char> buffer;
  ReadFileToBuffer(model_path.c_str(), buffer);

  std::vector<char> external_bin_buffer;
  std::string external_bin_path = test_folder + external_bin_name;
  ReadFileToBuffer(external_bin_path.c_str(), external_bin_buffer);

  Ort::SessionOptions so;
  std::string optimized_model_file_name(model_file_name);
  auto length = optimized_model_file_name.length();
  optimized_model_file_name.insert(length - 5, "_opt");
  std::string optimized_file_path(test_folder + optimized_model_file_name);
  PathString optimized_file_path_t(optimized_file_path.begin(), optimized_file_path.end());

  //  Dump the optimized model with external data so that it will unpack the external data from the loaded model
  so.SetOptimizedModelFilePath(optimized_file_path_t.c_str());

  // set the model external file folder path
  so.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath, test_folder.c_str());

  std::string opt_bin_file_name(optimized_model_file_name);
  opt_bin_file_name.replace(optimized_model_file_name.length() - 4, 4, "bin");
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersFileName, opt_bin_file_name.c_str());
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes, "10");

  Ort::Session session(*ort_env.get(), buffer.data(), buffer.size(), so);

  std::string generated_bin_path = test_folder + opt_bin_file_name;
  std::vector<char> generated_bin_buffer;
  ReadFileToBuffer(generated_bin_path.c_str(), generated_bin_buffer);

  ASSERT_EQ(external_bin_buffer, generated_bin_buffer);

  // Cleanup.
  ASSERT_EQ(std::remove(optimized_file_path.c_str()), 0);
  ASSERT_EQ(std::remove(generated_bin_path.c_str()), 0);
}

#ifndef _WIN32
struct FileDescriptorTraits {
  using Handle = int;
  static Handle GetInvalidHandleValue() { return -1; }
  static void CleanUp(Handle h) {
    ASSERT_TRUE(close(h) != -1);
  }
};
using ScopedFileDescriptor = ScopedResource<FileDescriptorTraits>;
#endif

void FileMmap(const ORTCHAR_T* file_path, void*& mapped_base) {
#ifdef _WIN32
  wil::unique_hfile file_handle{CreateFile2(file_path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, NULL)};
  ASSERT_TRUE(file_handle.get() != INVALID_HANDLE_VALUE);

  wil::unique_hfile file_mapping_handle{
      CreateFileMappingW(file_handle.get(),
                         nullptr,
                         PAGE_READONLY,
                         0,
                         0,
                         nullptr)};
  ASSERT_TRUE(file_mapping_handle.get() != INVALID_HANDLE_VALUE);
  mapped_base = MapViewOfFile(file_mapping_handle.get(),
                              FILE_MAP_READ,
                              0,
                              0,
                              0);
#else
  ScopedFileDescriptor file_descriptor{open(file_path, O_RDONLY)};
  ASSERT_TRUE(file_descriptor.IsValid());
  struct stat sb;
  stat(file_path, &sb);
  mapped_base = mmap(nullptr, narrow<size_t>(sb.st_size), PROT_READ | PROT_WRITE,
                     MAP_PRIVATE, file_descriptor.Get(), 0);
#endif
  return;
}

void TestLoadModelFromArrayWithExternalInitializerFromFileMmap(const std::string& model_file_name,
                                                               const std::string& external_data_file_name,
                                                               const std::string& external_ini_min_size_bytes = "10",
                                                               bool compare_external_bin_file = true) {
  std::string test_folder = "testdata/";
  std::string model_path = test_folder + model_file_name;
  std::vector<char> buffer;
  ReadFileToBuffer(model_path.c_str(), buffer);

  std::string external_bin_path = test_folder + external_data_file_name;
  PathString external_bin_path_t(external_bin_path.begin(), external_bin_path.end());

  void* mapped_base = nullptr;
  FileMmap(external_bin_path_t.c_str(), mapped_base);
  ASSERT_TRUE(mapped_base);

  std::ifstream bin_file(external_bin_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(bin_file);
  size_t bin_file_length = narrow<size_t>(bin_file.tellg());

  Ort::SessionOptions so;
  std::string optimized_model_file_name(model_file_name);
  auto length = optimized_model_file_name.length();
  optimized_model_file_name.insert(length - 5, "_opt");
  std::string optimized_file_path(test_folder + optimized_model_file_name);
  PathString optimized_file_path_t(optimized_file_path.begin(), optimized_file_path.end());

  so.SetOptimizedModelFilePath(optimized_file_path_t.c_str());
  //  Dump the optimized model with external data so that it will unpack the external data from the loaded model
  std::string opt_bin_file_name(optimized_model_file_name);
  opt_bin_file_name.replace(optimized_model_file_name.length() - 4, 4, "bin");
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersFileName, opt_bin_file_name.c_str());
  so.AddConfigEntry(kOrtSessionOptionsOptimizedModelExternalInitializersMinSizeInBytes, external_ini_min_size_bytes.c_str());

  PathString external_file_name(external_data_file_name.begin(), external_data_file_name.end());
  std::vector<PathString> file_names{external_file_name};
  std::vector<char*> file_buffers{static_cast<char*>(mapped_base)};
  std::vector<size_t> lengths{bin_file_length};
  so.AddExternalInitializersFromFilesInMemory(file_names, file_buffers, lengths);

  Ort::Session session(*ort_env.get(), buffer.data(), buffer.size(), so);

#ifdef _WIN32
  bool ret = UnmapViewOfFile(mapped_base);
  ASSERT_TRUE(ret);
#else
  struct stat sb;
  stat(external_bin_path.c_str(), &sb);
  int ret = munmap(mapped_base, narrow<size_t>(sb.st_size));
  ASSERT_TRUE(ret == 0);
#endif

  std::string generated_bin_path = test_folder + opt_bin_file_name;
  // If there are multiple initializers in the external bin file
  // It's hard to guarantee the generated bin for optimized model is exactly same with original one for some cases
  if (compare_external_bin_file) {
    std::vector<char> external_bin_buffer;
    ReadFileToBuffer(external_bin_path.c_str(), external_bin_buffer);

    std::vector<char> generated_bin_buffer;
    ReadFileToBuffer(generated_bin_path.c_str(), generated_bin_buffer);

    ASSERT_EQ(external_bin_buffer, generated_bin_buffer);
  }

  // Cleanup.
  ASSERT_EQ(std::remove(optimized_file_path.c_str()), 0);
  ASSERT_EQ(std::remove(generated_bin_path.c_str()), 0);
}

// Load external bin file using mmap
// Several external initializers from same file
// Use offset from tensor proto to locate the buffer location
TEST(CApiTest, TestLoadModelFromArrayWithExternalInitializersFromFileMmap) {
  std::string model_file_name = "conv_qdq_external_ini.onnx";
  std::string external_bin_name = "conv_qdq_external_ini.bin";
  TestLoadModelFromArrayWithExternalInitializerFromFileMmap(model_file_name, external_bin_name);
}

#endif
}  // namespace test
}  // namespace onnxruntime
