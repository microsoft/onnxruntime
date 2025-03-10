// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <filesystem> //NOLINT
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_c_api.h"
#include "include/utils.hpp"
#include "onnx/onnx_pb.h"

std::basic_string<PATH_CHAR_TYPE> find_model_path(std::string model_dir) {
  std::basic_string<PATH_CHAR_TYPE> ret(ORT_TSTR(""));
  for (auto const& file_entry : std::filesystem::directory_iterator(model_dir)) {
    if (file_entry.is_regular_file() && file_entry.path().extension() == ORT_TSTR(".onnx")) {
      ret = file_entry.path().native();
    }
  }
  return ret;
}

std::vector<std::basic_string<PATH_CHAR_TYPE>> find_test_data_sets(std::string model_dir) {
  std::vector<std::basic_string<PATH_CHAR_TYPE>> ret = {};
  const std::basic_string<PATH_CHAR_TYPE> prefix = ORT_TSTR("test_data_set_");
  for (auto const& dir_entry : std::filesystem::directory_iterator(model_dir)) {
    if (dir_entry.is_directory() && dir_entry.path().filename().native().compare(0, prefix.size(), prefix) == 0) {
      ret.push_back(dir_entry.path().native());
    }
  }
  return ret;
}

std::string check_data_format(const std::filesystem::path test_data_set_dir) {
  if (std::filesystem::is_empty(test_data_set_dir)) {
    std::cout << "input_0.pb or input_0.raw data should be provided" << std::endl;
    exit(0);
  }
  for (auto const& dir_entry : std::filesystem::directory_iterator(test_data_set_dir)) {
    if (dir_entry.path().extension().native() == ORT_TSTR(".pb")) {
      return "pb";
    } else if (dir_entry.path().extension().native() == ORT_TSTR(".raw")) {
      return "raw";
    } else {
      std::cout << "Only .pb or .raw format of data is supported" << std::endl;
      exit(0);
    }
  }
  return std::string();
}

void load_input_tensors_from_raws(
  std::filesystem::path inp_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info,
  std::vector<std::vector<float>>* input_data
) {
  // Multiple input .raw in each inp_dir (test_data_set_X)
  input_data->resize(model_info->get_num_in_tensors());
  for (size_t in_idx = 0; in_idx < model_info->get_num_in_tensors(); in_idx++) {
#ifdef _WIN32
    std::wstring infile_name = std::wstring(L"input_") + std::to_wstring(in_idx) + std::wstring(L".raw");
#else
    std::string infile_name = std::string("input_") + std::to_string(in_idx) + std::string(".raw")
#endif
    auto infile_path = (inp_dir / infile_name);
    // input data
    size_t input_byte_nums = model_info->get_in_tensor_element_nums()[in_idx] *
                             GetONNXTypeSize(model_info->get_out_tensor_element_types()[in_idx]);
    (*input_data)[in_idx].resize(input_byte_nums);

    // CreateTensor in Ort using input_data
    // The input_data should not be released until Inference completes
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    g_ort->CreateTensorWithDataAsOrtValue(
      memory_info, reinterpret_cast<void*>((*input_data)[in_idx].data()),
      input_byte_nums,
      model_info->get_in_tensor_dims()[in_idx].data(),
      model_info->get_in_tensor_dims()[in_idx].size(),
      model_info->get_in_tensor_element_types()[in_idx],
      &model_info->get_in_tensors()[in_idx]
    );
    // Read Input .raw
    std::ifstream input_raw_file(infile_path, std::ios::binary);
    input_raw_file.read(
      reinterpret_cast<char*>(&(*input_data)[in_idx][0]),
      input_byte_nums
    );
    input_raw_file.close();
  }
  return;
}

void load_input_tensors_from_pbs(
  std::filesystem::path inp_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info,
  std::vector<std::vector<float>>* input_data
) {
  // Multiple input .pb in each inp_dir (test_data_set_X)
  input_data->resize(model_info->get_num_in_tensors());
  for (size_t in_idx = 0; in_idx < model_info->get_num_in_tensors(); in_idx++) {
#ifdef _WIN32
    std::wstring infile_name = std::wstring(L"input_") + std::to_wstring(in_idx) + std::wstring(L".pb");
#else
    std::string infile_name = std::string("input_") + std::to_string(in_idx) + std::string(".pb")
#endif
    const std::filesystem::path infile_path = (inp_dir / infile_name);
    std::string buffer;
    buffer.resize(std::filesystem::file_size(infile_path));

    // input_X.pb -> String
    std::ifstream file(infile_path, std::ios::binary);
    file.read(&buffer[0], buffer.size());
    file.close();

    // String -> TensorProto
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.ParseFromString(buffer);

    // TensorProto -> std::vector<float>
    size_t input_byte_nums = model_info->get_in_tensor_element_nums()[in_idx] *
                             GetONNXTypeSize(model_info->get_out_tensor_element_types()[in_idx]);
    assert(input_byte_nums == tensor_proto.raw_data().size());
    (*input_data)[in_idx].resize(input_byte_nums);
    std::memcpy(
      (*input_data)[in_idx].data(),
      tensor_proto.raw_data().data(), input_byte_nums
    );

    // Prepare OrtValue
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    g_ort->CreateTensorWithDataAsOrtValue(
      memory_info,
      reinterpret_cast<void*>((*input_data)[in_idx].data()),
      input_byte_nums,
      model_info->get_in_tensor_dims()[in_idx].data(),
      model_info->get_in_tensor_dims()[in_idx].size(),
      model_info->get_in_tensor_element_types()[in_idx],
      &model_info->get_in_tensors()[in_idx]
    );
  }
  return;
}

void dump_output_tensors_to_raws(
  std::filesystem::path out_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info
) {
  // Dump output of each out tensor
  for (size_t out_idx = 0; out_idx < model_info->get_num_out_tensors(); out_idx++) {
    // output data
    void* output_buffer;
    g_ort->GetTensorMutableData(model_info->get_out_tensors()[out_idx], &output_buffer);
#ifdef _WIN32
    std::wstring outfile = std::wstring(L"out_") + std::to_wstring(out_idx) + std::wstring(L".raw");
#else
    std::string outfile = std::string("out_") + std::to_string(out_idx) + std::string(".raw")
#endif
    std::ofstream fout(out_dir / outfile, std::ios::binary);
    fout.write(
      reinterpret_cast<const char*>(output_buffer),
      model_info->get_out_tensor_element_nums()[out_idx] *
          GetONNXTypeSize(model_info->get_out_tensor_element_types()[out_idx])
    );
    fout.close();
  }
  return;
}

void dump_output_tensors_to_pbs(
  std::filesystem::path out_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info
) {
  // Dump output of each out tensor
  for (size_t out_idx = 0; out_idx < model_info->get_num_out_tensors(); out_idx++) {
    // output data
    ONNX_NAMESPACE::TensorProto tensor_proto;
    for (size_t j = 0; j < model_info->get_out_tensor_dims()[out_idx].size(); ++j) {
      tensor_proto.add_dims(static_cast<int>(model_info->get_out_tensor_dims()[out_idx][j]));
    }
    void* output_buffer;
    g_ort->GetTensorMutableData(model_info->get_out_tensors()[out_idx], &output_buffer);
#ifdef _WIN32
    std::wstring outfile = std::wstring(L"out_") + std::to_wstring(out_idx) + std::wstring(L".pb");
#else
    std::string outfile = std::string("out_") + std::to_string(out_idx) + std::string(".pb")
#endif
    tensor_proto.set_data_type(
      onnx_element_type_to_tensorproto_dtype(model_info->get_out_tensor_element_types()[out_idx]));
    tensor_proto.set_raw_data(
      output_buffer,
      model_info->get_out_tensor_element_nums()[out_idx] *
          GetONNXTypeSize(model_info->get_out_tensor_element_types()[out_idx])
    );
    std::ofstream fout(out_dir / outfile, std::ios::binary);
    tensor_proto.SerializeToOstream(&fout);
    fout.close();
  }
  return;
}
