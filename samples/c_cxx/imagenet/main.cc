// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string.h>
#include <sstream>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>
#include <setjmp.h>
#include <algorithm>
#include <vector>
#include <memory>
#include <atomic>

#include "providers.h"
#include "local_filesystem.h"
#include "sync_api.h"

#include <onnxruntime/core/session/onnxruntime_c_api.h>

#include "image_loader.h"
#include "async_ring_buffer.h"
#include <fstream>
#include <condition_variable>
#ifdef _WIN32
#include <atlbase.h>
#endif
using namespace std::chrono;

class Validator : public OutputCollector<TCharString> {
 private:
  static std::vector<std::string> ReadFileToVec(const TCharString& file_path, size_t expected_line_count) {
    std::ifstream ifs(file_path);
    if (!ifs) {
      throw std::runtime_error("open file failed");
    }
    std::string line;
    std::vector<std::string> labels;
    while (std::getline(ifs, line)) {
      if (!line.empty()) labels.push_back(line);
    }
    if (labels.size() != expected_line_count) {
      std::ostringstream oss;
      oss << "line count mismatch, expect " << expected_line_count << " from " << file_path.c_str() << ", got "
          << labels.size();
      throw std::runtime_error(oss.str());
    }
    return labels;
  }

  // input file name has pattern like:
  //"C:\tools\imagnet_validation_data\ILSVRC2012_val_00000001.JPEG"
  //"C:\tools\imagnet_validation_data\ILSVRC2012_val_00000002.JPEG"
  static int ExtractImageNumberFromFileName(const TCharString& image_file) {
    size_t s = image_file.rfind('.');
    if (s == std::string::npos) throw std::runtime_error("illegal filename");
    size_t s2 = image_file.rfind('_');
    if (s2 == std::string::npos) throw std::runtime_error("illegal filename");

    const ORTCHAR_T* start_ptr = image_file.c_str() + s2 + 1;
    const ORTCHAR_T* endptr = nullptr;
    long value = my_strtol(start_ptr, (ORTCHAR_T**)&endptr, 10);
    if (start_ptr == endptr || value > INT32_MAX || value <= 0) throw std::runtime_error("illegal filename");
    return static_cast<int>(value);
  }

  static void VerifyInputOutputCount(OrtSession* session) {
    size_t count;
    ORT_THROW_ON_ERROR(OrtSessionGetInputCount(session, &count));
    assert(count == 1);
    ORT_THROW_ON_ERROR(OrtSessionGetOutputCount(session, &count));
    assert(count == 1);
  }

  OrtSession* session_ = nullptr;
  const int output_class_count_ = 1001;
  std::vector<std::string> labels_;
  std::vector<std::string> validation_data_;
  std::atomic<int> top_1_correct_count_;
  std::atomic<int> finished_count_;
  int image_size_;

  std::mutex m_;
  char* input_name_ = nullptr;
  char* output_name_ = nullptr;
  OrtEnv* const env_;
  const TCharString model_path_;
  system_clock::time_point start_time_;

 public:
  int GetImageSize() const { return image_size_; }

  ~Validator() {
    free(input_name_);
    free(output_name_);
    OrtReleaseSession(session_);
  }

  void PrintResult() {
    if (finished_count_ == 0) return;
    printf("Top-1 Accuracy %f\n", ((float)top_1_correct_count_.load() / finished_count_));
  }

  void ResetCache() override {
    OrtReleaseSession(session_);
    CreateSession();
  }

  void CreateSession() {
    OrtSessionOptions* session_option;
    ORT_THROW_ON_ERROR(OrtCreateSessionOptions(&session_option));
#ifdef USE_CUDA
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0));
#endif
    ORT_THROW_ON_ERROR(OrtCreateSession(env_, model_path_.c_str(), session_option, &session_));
    OrtReleaseSessionOptions(session_option);
  }

  Validator(OrtEnv* env, const TCharString& model_path, const TCharString& label_file_path,
            const TCharString& validation_file_path, size_t input_image_count)
      : labels_(ReadFileToVec(label_file_path, 1000)),
        validation_data_(ReadFileToVec(validation_file_path, input_image_count)),
        top_1_correct_count_(0),
        finished_count_(0),
        env_(env),
        model_path_(model_path) {
    CreateSession();
    VerifyInputOutputCount(session_);
    OrtAllocator* ort_alloc;
    ORT_THROW_ON_ERROR(OrtGetAllocatorWithDefaultOptions(&ort_alloc));
    {
      char* t;
      ORT_THROW_ON_ERROR(OrtSessionGetInputName(session_, 0, ort_alloc, &t));
      input_name_ = my_strdup(t);
      OrtAllocatorFree(ort_alloc, t);
      ORT_THROW_ON_ERROR(OrtSessionGetOutputName(session_, 0, ort_alloc, &t));
      output_name_ = my_strdup(t);
      OrtAllocatorFree(ort_alloc, t);
    }

    OrtTypeInfo* info;
    ORT_THROW_ON_ERROR(OrtSessionGetInputTypeInfo(session_, 0, &info));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    ORT_THROW_ON_ERROR(OrtCastTypeInfoToTensorInfo(info, &tensor_info));
    size_t dim_count;
    ORT_THROW_ON_ERROR(OrtGetDimensionsCount(tensor_info, &dim_count));
    assert(dim_count == 4);
    std::vector<int64_t> dims(dim_count);
    ORT_THROW_ON_ERROR(OrtGetDimensions(tensor_info, dims.data(), dims.size()));
    if (dims[1] != dims[2] || dims[3] != 3) {
      throw std::runtime_error("This model is not supported by this program. input tensor need be in NHWC format");
    }

    image_size_ = static_cast<int>(dims[1]);
    start_time_ = system_clock::now();
  }

  void operator()(const std::vector<TCharString>& task_id_list, const OrtValue* input_tensor) override {
    {
      std::lock_guard<std::mutex> l(m_);
      const size_t remain = task_id_list.size();
      OrtValue* output_tensor = nullptr;
      ORT_THROW_ON_ERROR(OrtRun(session_, nullptr, &input_name_, &input_tensor, 1, &output_name_, 1, &output_tensor));
      float* probs;
      ORT_THROW_ON_ERROR(OrtGetTensorMutableData(output_tensor, (void**)&probs));
      for (const auto& s : task_id_list) {
        float* end = probs + output_class_count_;
        float* max_p = std::max_element(probs + 1, end);
        auto max_prob_index = std::distance(probs, max_p);
        assert(max_prob_index >= 1);
        int test_data_id = ExtractImageNumberFromFileName(s);
        assert(test_data_id >= 1);
        if (labels_[max_prob_index - 1] == validation_data_[test_data_id - 1]) {
          ++top_1_correct_count_;
        }
        probs = end;
      }
      size_t finished = finished_count_ += static_cast<int>(remain);
      float progress = static_cast<float>(finished) / validation_data_.size();
      auto elapsed = system_clock::now() - start_time_;
      auto eta = progress > 0 ? duration_cast<minutes>(elapsed * (1 - progress) / progress).count() : 9999999;
      float accuracy = finished > 0 ? top_1_correct_count_ / static_cast<float>(finished) : 0;
      printf("accuracy = %.2f, progress %.2f%%, expect to be finished in %d minutes\n", accuracy, progress * 100, eta);
      OrtReleaseValue(output_tensor);
    }
  }
};

int real_main(int argc, ORTCHAR_T* argv[]) {
  if (argc < 6) return -1;
  std::vector<TCharString> image_file_paths;
  TCharString data_dir = argv[1];
  TCharString model_path = argv[2];
  // imagenet_lsvrc_2015_synsets.txt
  TCharString label_file_path = argv[3];
  TCharString validation_file_path = argv[4];
  const int batch_size = std::stoi(argv[5]);

  // TODO: remove the slash at the end of data_dir string
  LoopDir(data_dir, [&data_dir, &image_file_paths](const ORTCHAR_T* filename, OrtFileType filetype) -> bool {
    if (filetype != OrtFileType::TYPE_REG) return true;
    if (filename[0] == '.') return true;
    const ORTCHAR_T* p = my_strrchr(filename, '.');
    if (p == nullptr) return true;
    // as we tested filename[0] is not '.', p should larger than filename
    assert(p > filename);
    if (my_strcasecmp(p, ORT_TSTR(".JPEG")) != 0 && my_strcasecmp(p, ORT_TSTR(".JPG")) != 0) return true;
    TCharString v(data_dir);
#ifdef _WIN32
    v.append(1, '\\');
#else
    v.append(1, '/');
#endif
    v.append(filename);
    image_file_paths.emplace_back(v);
    return true;
  });

  std::vector<uint8_t> data;
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");

  Validator v(env, model_path, label_file_path, validation_file_path, image_file_paths.size());

  //Which image size does the model expect? 224, 299, or ...?
  int image_size = v.GetImageSize();
  const int channels = 3;
  std::atomic<int> finished(0);

  InceptionPreprocessing prepro(image_size, image_size, channels);
  Controller c;
  AsyncRingBuffer<std::vector<TCharString>::iterator> buffer(batch_size, 160, c, image_file_paths.begin(),
                                                             image_file_paths.end(), &prepro, &v);
  buffer.StartDownloadTasks();
  std::string err = c.Wait();
  if (err.empty()) {
    buffer.ProcessRemain();
    v.PrintResult();
    return 0;
  }
  fprintf(stderr, "%s\n", err.c_str());
  return -1;
}
#ifdef _WIN32
int wmain(int argc, ORTCHAR_T* argv[]) {
  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  if (!SUCCEEDED(hr)) return -1;
#else
int main(int argc, ORTCHAR_T* argv[]) {
#endif
  int ret = -1;
  try {
    ret = real_main(argc, argv);
  } catch (const std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
  }
#ifdef _WIN32
  CoUninitialize();
#endif
  return ret;
}