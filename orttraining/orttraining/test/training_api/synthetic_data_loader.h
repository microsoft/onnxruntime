// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

namespace onnxruntime {
namespace training {
namespace test {
namespace training_api {

template <typename T>
struct TypedSynctheticInput;

struct SyntheticInput {
  SyntheticInput(const std::vector<int64_t>& shape) : shape_(shape) {
    for (auto d : shape) {
      num_of_bytes_per_sample_ *= d;
    }
  }

  virtual ~SyntheticInput(){};

  template <typename T>
  std::vector<T>& GetData() {
    auto ptr = dynamic_cast<TypedSynctheticInput<T>*>(this);
    return ptr->Data();
  }

  size_t NumOfBytesPerSample() {
    return num_of_bytes_per_sample_;
  }

  std::vector<int64_t> ShapeVector() const {
    return shape_;
  }

 protected:
  std::vector<int64_t> shape_;
  size_t num_of_bytes_per_sample_{1};
};

template <typename T>
struct TypedSynctheticInput : public SyntheticInput {
  TypedSynctheticInput(const std::vector<int64_t>& shape)
      : SyntheticInput(shape) {
    data_.resize(num_of_bytes_per_sample_);
  }

  std::vector<T>& Data() {
    return data_;
  }

 private:
  std::vector<T> data_;
};

struct SyntheticSampleBatch {
  SyntheticSampleBatch() {}

  void AddInt32Input(const std::vector<int64_t>& shape, int32_t low, int32_t high);
  void AddInt64Input(const std::vector<int64_t>& shape, int64_t low, int64_t high);
  void AddFloatInput(const std::vector<int64_t>& shape);

  size_t NumOfInput() {
    return data_vector_.size();
  }

  SyntheticInput* GetInputAtIndex(size_t index) {
    return data_vector_[index].get();
  }

 private:
  std::vector<std::unique_ptr<SyntheticInput>> data_vector_;
};

struct SyntheticDataLoader {
  SyntheticDataLoader() {}

  void AddSyntheticSampleBatch(std::unique_ptr<SyntheticSampleBatch> samples) {
    sample_batch_collections_.emplace_back(std::move(samples));
    num_of_sample_batches += 1;
  }

  bool GetNextSampleBatch(std::vector<Ort::Value>& batches);

  size_t NumOfSampleBatches() {
    return num_of_sample_batches;
  }

  void ResetIterateIndex() {
    sample_batch_iter_index_ = 0;
  }

 private:
  // Be noted: all raw data MUST remain during the training, because all Ort::Value created as session inputs
  // did not explicitly copy the data in.
  // And also, the created Ort::Value also won't clean the raw data pointer. The raw data should be removed when
  // the life time of this struct ends.
  std::vector<std::unique_ptr<SyntheticSampleBatch>> sample_batch_collections_;
  int64_t sample_batch_count_;
  size_t sample_batch_iter_index_{0};
  size_t num_of_sample_batches{0};
};

}  // namespace training_api
}  // namespace test
}  // namespace training
}  // namespace onnxruntime