// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <winstring.h>

// String Helpers
namespace _winml::Strings {
struct HStringBuilder {
  HStringBuilder(HStringBuilder const&) = delete;
  HStringBuilder& operator=(HStringBuilder const&) = delete;

  explicit HStringBuilder(UINT32 size) { winrt::check_hresult(WindowsPreallocateStringBuffer(size, &data_, &buffer_)); }

  ~HStringBuilder() noexcept {
    if (buffer_ != nullptr) {
      WindowsDeleteStringBuffer(buffer_);
    }
  }

  wchar_t* data() noexcept { return data_; }

  winrt::hstring to_hstring() {
    winrt::hstring result;
    winrt::check_hresult(WindowsPromoteStringBuffer(buffer_, reinterpret_cast<HSTRING*>(put_abi(result))));
    buffer_ = nullptr;
    return result;
  }

 private:
  wchar_t* data_{nullptr};
  HSTRING_BUFFER buffer_{nullptr};
};

inline winrt::hstring HStringFromUTF8(const char* input, size_t input_length) {
  if (input_length == 0) {
    return {};
  } else if (input_length <= (std::numeric_limits<size_t>::max)()) {
    int output_length = MultiByteToWideChar(CP_UTF8, 0, input, static_cast<int>(input_length), nullptr, 0);
    if (output_length > 0) {
      HStringBuilder buffer(static_cast<UINT32>(output_length));
      MultiByteToWideChar(CP_UTF8, 0, input, static_cast<int>(input_length), buffer.data(), output_length);
      return buffer.to_hstring();
    } else {
      winrt::throw_hresult(E_INVALIDARG);
    }
  } else {
    winrt::throw_hresult(E_INVALIDARG);
  }
}

inline winrt::hstring HStringFromUTF8(const char* input) {
  return input != nullptr ? HStringFromUTF8(input, strlen(input)) : L"";
}

inline winrt::hstring HStringFromUTF8(const std::string& input) {
  return HStringFromUTF8(input.c_str(), input.size());
}

inline std::string UTF8FromUnicode(const wchar_t* input, size_t input_length) {
  if (input_length == 0) {
    return {};
  } else if (input_length <= (std::numeric_limits<size_t>::max)()) {
    int output_length =
      WideCharToMultiByte(CP_UTF8, 0, input, static_cast<int>(input_length), nullptr, 0, nullptr, nullptr);
    if (output_length > 0) {
      std::string output(output_length, 0);
      WideCharToMultiByte(
        CP_UTF8, 0, input, static_cast<int>(input_length), &output[0], output_length, nullptr, nullptr
      );
      return output;
    } else {
      winrt::throw_hresult(E_INVALIDARG);
    }
  } else {
    winrt::throw_hresult(E_INVALIDARG);
  }
}

inline std::string UTF8FromHString(const winrt::hstring& input) {
  return UTF8FromUnicode(input.data(), input.size());
}

inline std::wstring WStringFromString(const std::string& string) {
  std::wostringstream woss;
  woss << string.data();
  return woss.str();
}

}  // namespace _winml::Strings
