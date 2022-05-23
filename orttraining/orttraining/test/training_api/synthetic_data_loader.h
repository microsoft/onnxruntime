// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

struct SyntheticDataLoader {
  SyntheticDataLoader(size_t sample_count, size_t batch_size);

  void GetNextBatch(std::vector<Ort::Value>& batches);

  size_t NumOfBatches() {
    return num_of_batches_;
  }

 private:
  // Be noted: all raw data MUST remain during the training, because all Ort::Value created as session inputs
  // did not explicitly copy the data in.
  // And also, the created Ort::Value also won't clean the raw data pointer. The raw data should be removed when
  // the life time of this struct ends.
  std::vector<float> input1_;
  std::vector<int32_t> label_;
  size_t sample_count_;
  size_t batch_size_;
  size_t batch_index_;
  size_t num_of_batches_;
  size_t hidden_size_ = 784;
};

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime