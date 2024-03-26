#pragma once

#include <cstddef>

namespace std_ {

template <typename T>
class span {
public:
    span(T* data, std::size_t size) : data_(data), size_(size) {}

    inline T* data() const { return data_; }
    inline std::size_t size() const { return size_; }
    inline T& operator[](std::size_t index) const { return data_[index]; }
    inline T* begin() const { return data_; }
    inline T* end() const { return data_ + size_; }

private:
    T* data_;
    std::size_t size_;
};

} // namespace std_
