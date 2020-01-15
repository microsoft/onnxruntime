// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <string>
#include "cached_interpolation.h"
#include "sync_api.h"
#include "data_processing.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>

template <typename T>
void ResizeImageInMemory(const T* input_data, float* output_data, int in_height, int in_width, int out_height,
                         int out_width, int channels);

template <typename InputType>
class OutputCollector {
 public:
  virtual void operator()(const std::vector<InputType>& task_id_list, const Ort::Value& tensor) = 0;
  // Release the internal cache. It need be called whenever batchsize is changed
  virtual void ResetCache() = 0;
  virtual ~OutputCollector() = default;
};

bool CreateImageLoader(void** out);
OrtStatus* LoadImageFromFileAndCrop(void* loader, const ORTCHAR_T* filename, double central_crop_fraction, float** out,
                                    int* out_width, int* out_height);

void ReleaseImageLoader(void* p);

class InceptionPreprocessing : public DataProcessing {
 private:
  const int out_height_;
  const int out_width_;
  const int channels_;
  const double central_fraction_ = 0.875;
  void* image_loader_;

 public:
  InceptionPreprocessing(int out_height, int out_width, int channels);

  void operator()(_In_ const void* input_data, _Out_writes_bytes_all_(output_len) void* output_data, size_t output_len) const override;

  // output data from this class is in NWHC format
  std::vector<int64_t> GetOutputShape(size_t batch_size) const override {
    return {(int64_t)batch_size, out_height_, out_width_, channels_};
  }
};
