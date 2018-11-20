// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>

namespace onnxruntime {
/**
   Container has T* entries. e.g. std::vector<T*>, and this class provides const access to those
   via iterators and direct access, as the standard behavior only makes the pointer constant,
   and not what is pointed too. i.e. you get a const pointer to T not a pointer to const T without this wrapper.
   See https://stackoverflow.com/questions/8017036/understanding-const-iterator-with-pointers
*/
template <typename Container>
class ConstPointerContainer {
 public:
  using T = typename std::remove_pointer<typename Container::value_type>::type;

  class ConstIterator {
   public:
    using const_iterator = typename Container::const_iterator;

    /** Construct iterator for container that will return const T* entries.*/
    explicit ConstIterator(const_iterator position) noexcept : current_(position) {}

    bool operator==(const ConstIterator& other) const noexcept { return current_ == other.current_; }
    bool operator!=(const ConstIterator& other) const noexcept { return current_ != other.current_; }
    void operator++() { ++current_; }
    const T* operator*() { return *current_; }

   private:
    const_iterator current_;
  };

  /**
     Construct wrapper class that will provide const access to the pointers in a container of non-const pointers.
     @param data Container with non-const pointers. e.g. std::vector<T*>
  */
  explicit ConstPointerContainer(const Container& data) noexcept : data_(data) {}

  size_t size() const noexcept { return data_.size(); }

  ConstIterator begin() const noexcept { return ConstIterator(data_.cbegin()); }
  ConstIterator end() const noexcept { return ConstIterator(data_.cend()); }

  const T* operator[](size_t index) const { return data_[index]; }

  const T* at(size_t index) const {
    ONNXRUNTIME_ENFORCE(index < data_.size());
    return data_[index];
  }

 private:
  const Container& data_;
};
}  // namespace onnxruntime
