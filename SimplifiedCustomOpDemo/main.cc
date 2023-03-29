#include "onnxruntime_cxx_api.h"

#include <iostream>
#include <iterator>

#ifdef WIN32
#define FUSION_FILTER L"..\\..\\fusion_filter_2.onnx"
#define STRING_MERGE L"..\\..\\str_merge.onnx"
#define SIMPLE_CUSTOM_OP_LIB L"simple_custom_op.dll"
#else
#define FUSION_FILTER "../fusion_filter_2.onnx"
#define STRING_MERGE "../str_merge.onnx"
#define SIMPLE_CUSTOM_OP_LIB "./libsimple_custom_op.so"
#endif

void TestSelectFuseFilter() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestSelectFuseFilter");
  const auto& ortApi = Ort::GetApi();

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.RegisterCustomOpsLibrary(SIMPLE_CUSTOM_OP_LIB);
  session_options.SetLogSeverityLevel(0);

#ifdef WIN32
  const wchar_t* model_path = FUSION_FILTER;
#else
  const char* model_path = FUSION_FILTER;
#endif

  Ort::Session session(env, model_path, session_options);

  const char* input_names[] = {"vector_1", "vector_2", "alpha", "indices"};
  const char* output_names[] = {"vector_filtered"};

  float vector_1_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
  int64_t vector_1_dim[] = {10};

  float vector_2_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  int64_t vector_2_dim[] = {6};

  int32_t alpha_value[] = {2};
  int64_t alpha_dim[] = {1};

  int32_t indices_value[] = {0, 1, 2, 3, 4, 5};
  int64_t indices_dim[] = {6};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, 10, vector_1_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, 6, vector_2_dim, 1),
      Ort::Value::CreateTensor<int32_t>(memory_info, alpha_value, 1, alpha_dim, 1),
      Ort::Value::CreateTensor<int32_t>(memory_info, indices_value, 6, indices_dim, 1)};

  Ort::RunOptions run_optoins;
  auto output_tensors = session.Run(run_optoins, input_names, input_tensors, 4, output_names, 1);
  const auto& vector_filterred = output_tensors.at(0);
  auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
  size_t num_output = type_shape_info.GetElementCount();
  const float* floats_output = static_cast<const float*>(vector_filterred.GetTensorRawData());

  std::cout << std::endl
            << "/////////////////////////////// OUTPUT ///////////////////////////////" << std::endl;
  std::copy(floats_output, floats_output + num_output, std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;
}

void TestStrMerge() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestStrMerge");
  const auto& ortApi = Ort::GetApi();

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.RegisterCustomOpsLibrary(SIMPLE_CUSTOM_OP_LIB);
  session_options.SetLogSeverityLevel(0);

#ifdef WIN32
  const wchar_t* model_path = STRING_MERGE;
#else
  const char* model_path = STRING_MERGE;
#endif

  Ort::Session session(env, model_path, session_options);

  const char* input_names[] = {"str_in_1", "str_in_2"};
  const char* output_names[] = {"str_out"};

  OrtAllocator* allocator = nullptr;
  ortApi.GetAllocatorWithDefaultOptions(&allocator);
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  int64_t str_1_dims[] = {2};
  int64_t str_2_dims[] = {1};

  Ort::Value input_tensors[] = {Ort::Value::CreateTensor(allocator, str_1_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
                                Ort::Value::CreateTensor(allocator, str_2_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)};

  const char* str_1_raw[] = {"abc", "de"};
  const char* str_2_raw[] = {"fg"};

  input_tensors[0].FillStringTensor(str_1_raw, 2);
  input_tensors[1].FillStringTensor(str_2_raw, 1);

  Ort::RunOptions run_optoins;
  auto output_tensors = session.Run(run_optoins, input_names, input_tensors, 2, output_names, 1);
  const auto& str_out_tensor = output_tensors.at(0);
  auto num_chars = str_out_tensor.GetStringTensorDataLength();
  // todo - too much copies here ...
  std::vector<char> chars(num_chars + 1, '\0');
  std::vector<size_t> offsets(3);
  std::vector<std::string> str_output(3);
  str_out_tensor.GetStringTensorContent(static_cast<void*>(chars.data()), num_chars, offsets.data(), offsets.size());
  for (int64_t i = 2; i >= 0; --i) {
    if (i < 2) {
      chars[offsets[i + 1]] = '\0';
    }
    str_output[i] = chars.data() + offsets[i];
  }

  std::cout << std::endl
            << "/////////////////////////////// OUTPUT ///////////////////////////////" << std::endl;
  std::copy(str_output.begin(), str_output.end(), std::ostream_iterator<std::string>(std::cout, ""));
  std::cout << std::endl;
}

int main() {
  // TestSelectFuseFilter();
  TestStrMerge();
  std::cout << "done" << std::endl;
}
