// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>
namespace vaip_core {
template <typename T>
class DllSafe {
 public:
  DllSafe() : value_{nullptr}, deleter_{nullptr} {}
  DllSafe(const DllSafe& other) = delete;
  DllSafe(DllSafe&& other) : value_{other.value_}, deleter_{other.deleter_} {
    other.value_ = nullptr;
    other.deleter_ = nullptr;
  }
  explicit DllSafe(T* value)
      : value_{value}, deleter_{[](T* value) noexcept {
          std::default_delete<T>()(value);
        }} {}

  explicit DllSafe(T&& value) : DllSafe(new T(std::move(value))) {}
  explicit DllSafe(const T& value) : DllSafe(new T(value)) {}

  ~DllSafe() noexcept {
    if (value_ != nullptr) {
      deleter_(value_);
    }
  }

  T& operator*() { return *value_; }
  T* operator->() { return value_; }
  DllSafe& operator=(DllSafe&& other) {
    std::swap(this->value_, other.value_);
    std::swap(this->deleter_, other.deleter_);
    return *this;
  }
  T* get() const { return value_; }

  std::unique_ptr<T, void (*)(T*)> to_ptr() {
    auto value = value_;
    value_ = nullptr;
    return std::unique_ptr<T, void (*)(T*)>(value, deleter_);
  }

 private:
  T* value_;
  void (*deleter_)(T*) noexcept;
};
}  // namespace vaip_core
