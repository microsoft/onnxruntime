// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <array>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <string>
#include <type_traits>
#include <vector>

// Make a bootleg std::span for C++ versions older than 20
template <typename T>
class Span {
 public:
  Span() = default;
  Span(T* data, size_t size) : data_(data), size_(size) {}
  Span(std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {}
  Span(const std::vector<std::remove_const_t<T>>& vec) : data_(vec.data()), size_(vec.size()) {}

  template <size_t N>
  Span(std::array<T, N> arr) : data_(arr.data()), size_(N) {}

  Span(const Span& other) = default;
  Span(Span&& other) = default;

  Span& operator=(const Span& other) = default;
  Span& operator=(Span&& other) = default;

  T& operator[](size_t index) const {
    return data_[index];
  }

  T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  T* data_{nullptr};
  size_t size_{0};
};

template <typename T>
static Span<T> ReinterpretBytesAsSpan(Span<std::conditional_t<std::is_const_v<T>, const char, char>> bytes_span) {
  return Span<T>(reinterpret_cast<T*>(bytes_span.data()), bytes_span.size() / sizeof(T));
}

template <typename T>
constexpr int64_t GetShapeSize(Span<T> shape) {
  int64_t size = 1;

  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }

  return size;
}

int32_t GetFileIndexSuffix(const std::string& filename_wo_ext, const char* prefix);
bool FillBytesFromBinaryFile(Span<char> array, const std::string& binary_filepath);

constexpr double EPSILON_DBL = 2e-16;

struct AccMetrics {
  double rmse = 0.0;
  double snr = 0.0;
  double min_val = 0.0;
  double max_val = 0.0;
  double min_expected_val = 0.0;
  double max_expected_val = 0.0;
  friend bool operator==(const AccMetrics& l, const AccMetrics& r) {
    if (l.rmse != r.rmse) return false;
    if (l.min_val != r.min_val) return false;
    if (l.max_val != r.max_val) return false;
    if (l.min_expected_val != r.min_expected_val) return false;
    if (l.max_expected_val != r.max_expected_val) return false;
    if (l.snr != r.snr) return false;

    return true;
  }
  friend bool operator!=(const AccMetrics& l, const AccMetrics& r) {
    return !(l == r);
  }
};

template <typename T>
void GetAccuracy(Span<const T> expected_output, Span<const T> actual_output, AccMetrics& metrics) {
  // Compute RMSE. This is not a great way to measure accuracy, but ....
  assert(expected_output.size() == actual_output.size());
  const size_t num_outputs = expected_output.size();

  metrics.rmse = 0.0;
  metrics.min_val = static_cast<double>(actual_output[0]);
  metrics.max_val = static_cast<double>(actual_output[0]);
  metrics.min_expected_val = static_cast<double>(expected_output[0]);
  metrics.max_expected_val = static_cast<double>(expected_output[0]);
  double tensor_norm = 0.0;
  double diff_norm = 0.0;
  for (size_t i = 0; i < num_outputs; i++) {
    double diff = static_cast<double>(actual_output[i]) - static_cast<double>(expected_output[i]);
    diff_norm += diff * diff;
    tensor_norm += static_cast<double>(expected_output[i]) * static_cast<double>(expected_output[i]);

    metrics.rmse += diff * diff;
    metrics.min_val = std::min(metrics.min_val, static_cast<double>(actual_output[i]));
    metrics.max_val = std::max(metrics.max_val, static_cast<double>(actual_output[i]));
    metrics.min_expected_val = std::min(metrics.min_expected_val, static_cast<double>(expected_output[i]));
    metrics.max_expected_val = std::max(metrics.max_expected_val, static_cast<double>(expected_output[i]));
  }

  metrics.rmse = std::sqrt(metrics.rmse / static_cast<double>(num_outputs));

  tensor_norm = std::max(std::sqrt(tensor_norm), EPSILON_DBL);
  diff_norm = std::max(std::sqrt(diff_norm), EPSILON_DBL);
  metrics.snr = 20.0 * std::log10(tensor_norm / diff_norm);
}
