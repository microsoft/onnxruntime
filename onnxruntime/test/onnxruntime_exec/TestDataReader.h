/*
* Copyright (c) Microsoft Corporation. All rights reserved.
* Licensed under the MIT License.
*/

#pragma once

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <locale>
#include <codecvt>

std::string wstr2str(const std::wstring& wstr) {
  std::string str = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(wstr);
  return str;
}

class DataValidationException : public std::exception {
 public:
  DataValidationException(const std::string& str) : str_(str) {
  }
  const char* what() const noexcept override {
    return str_.c_str();
  }

 private:
  std::string str_;
};

class TestDataReader {
 public:
  static std::unique_ptr<TestDataReader> OpenReader(std::wstring data_file);

  void BufferNextSample();
  bool Eof();

  template <typename T>
  std::vector<T> GetSample(int sample_count, bool variable_batch_size = false);

  std::vector<std::wstring> GetSampleStrings(int sample_count, bool variable_batch_size = false);

 private:
  std::wstring line_;
  std::wifstream reader_stream_;
  std::unique_ptr<std::wstringstream> row_stream_;
};

bool TestDataReader::Eof() {
  return reader_stream_.eof();
}

void TestDataReader::BufferNextSample() {
  if (Eof())
    return;

  std::getline(reader_stream_, line_);

  if (Eof())
    return;

  row_stream_ = std::make_unique<std::wstringstream>(line_);
  std::wstring feature;
  std::getline(*row_stream_, feature, L',');  //Skip the Label which is actually.
}

template <typename T>
std::vector<T> TestDataReader::GetSample(int sample_count, bool variable_batch_size) {
  assert(sample_count == -1 || sample_count > 0);

  std::wstring feature;
  std::vector<T> result;

  int s = 0;
  while ((s++ < sample_count || sample_count == -1 || variable_batch_size) &&
         std::getline(*row_stream_, feature, L','))  // -1 means read all data in the sample
  {
    T feature_value;
    std::wstringstream feature_convert(feature);
    feature_convert >> feature_value;
    if (feature_convert.fail()) {
      feature_value = (T)NAN;
    }

    result.push_back(feature_value);
  }

  if (line_.length() > 0 && line_.back() == L',')
    result.push_back((T)NAN);

  if (sample_count != -1 && !variable_batch_size) {
    //Remove the last NAN inserted if it is not part of this feature.
    if (result.size() == sample_count + 1)
      result.pop_back();

    if (result.size() != sample_count)
      throw DataValidationException("Not enough features in sample.");
  }

  if (variable_batch_size && (result.size() % sample_count != 0) && (sample_count != -1))
    throw DataValidationException("Input count is not a multiple of dimension.");

  return result;
}

std::vector<std::wstring> TestDataReader::GetSampleStrings(int sample_count, bool variable_batch_size) {
  std::wstring feature;
  std::vector<std::wstring> result;

  int s = 0;
  while (s < sample_count || sample_count == -1 || variable_batch_size)  // -1 means read all data in the sample
  {
    if (std::getline(*row_stream_, feature, L','))
      result.push_back(feature);
    else {
      if (sample_count == -1 || variable_batch_size)
        break;

      throw DataValidationException("Not enough features in sample.");
    }
    s++;
  }

  if (line_.length() > 0 && line_.back() == L',')
    result.push_back(L"");

  if (variable_batch_size && (result.size() % sample_count != 0) && (sample_count != -1))
    throw DataValidationException("Input count is not a multiple of dimension.");

  return result;
}

std::unique_ptr<TestDataReader> TestDataReader::OpenReader(std::wstring dataFile) {
  auto reader = std::make_unique<TestDataReader>();

  reader->reader_stream_.open(wstr2str(dataFile));
  if (!reader->reader_stream_) {
    reader = nullptr;
  }

  return reader;
}
