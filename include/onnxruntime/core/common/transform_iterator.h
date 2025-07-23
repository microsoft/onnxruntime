// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <iterator>
#include <type_traits>

namespace onnxruntime {

namespace detail {

template <typename Iterator, typename UnaryFunction,
          bool IsReference = std::is_reference_v<
              std::invoke_result_t<UnaryFunction, typename std::iterator_traits<Iterator>::reference>>>
class transform_iterator_base;

// Primary template: for when UnaryFunction returns a value (prvalue)
template <typename Iterator, typename UnaryFunction>
class transform_iterator_base<Iterator, UnaryFunction, false> {
 protected:
  using traits = std::iterator_traits<Iterator>;
  using value_type = std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<UnaryFunction, typename traits::reference>>>;
  using func_return_type = std::invoke_result_t<UnaryFunction, typename traits::reference>;

 public:
  using pointer = value_type*;

 protected:
  pointer operator_arrow(const Iterator& current) const {
    result_cache_ = f_(*current);
    return &result_cache_;
  }

  explicit transform_iterator_base(std::function<func_return_type(typename traits::reference)> f)
      : f_(std::move(f)) {}

  std::function<func_return_type(typename traits::reference)> f_{};

 private:
  mutable value_type result_cache_;
};

// Specialization: for when UnaryFunction returns a reference
template <typename Iterator, typename UnaryFunction>
class transform_iterator_base<Iterator, UnaryFunction, true> {
 protected:
  using traits = std::iterator_traits<Iterator>;
  using reference = std::invoke_result_t<UnaryFunction, typename traits::reference>;
  using func_return_type = std::invoke_result_t<UnaryFunction, typename traits::reference>;

  explicit transform_iterator_base(std::function<func_return_type(typename traits::reference)> f)
      : f_(std::move(f)) {}

 public:
  using pointer = std::remove_reference_t<reference>*;

 protected:
  pointer operator_arrow(const Iterator& current) const {
    return &f_(*current);
  }

  std::function<func_return_type(typename traits::reference)> f_{};
};

}  // namespace detail

/**
 * @brief An iterator adapter that applies a unary function to the result of dereferencing another iterator.
 *
 * This iterator wraps an underlying iterator and transforms its dereferenced value using a provided function.
 * It supports different iterator categories (input, forward, bidirectional, random access) based on the
 * capabilities of the underlying iterator.
 *
 * This is conceptually similar to `boost::transform_iterator`.
 *
 * @tparam Iterator The type of the underlying iterator.
 * @tparam UnaryFunction The type of the function to apply to the elements.
 */
template <typename Iterator, typename UnaryFunction>
class transform_iterator : public detail::transform_iterator_base<Iterator, UnaryFunction> {
  using base = detail::transform_iterator_base<Iterator, UnaryFunction>;

 public:
  using underlying_iterator_type = Iterator;
  using function_type = UnaryFunction;

  using traits = std::iterator_traits<Iterator>;
  using iterator_category = typename traits::iterator_category;
  using difference_type = typename traits::difference_type;
  using value_type = std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<UnaryFunction,
                                                                                   typename traits::reference>>>;
  using reference = std::invoke_result_t<UnaryFunction, typename traits::reference>;
  using pointer = typename base::pointer;

  transform_iterator() = default;
  transform_iterator(Iterator it, UnaryFunction f) : base(std::move(f)), current_(std::move(it)) {}

  reference operator*() const {
    return this->f_(*current_);
  }

  pointer operator->() const {
    return this->operator_arrow(current_);
  }

  transform_iterator& operator++() {
    ++current_;
    return *this;
  }

  transform_iterator operator++(int) {
    transform_iterator tmp = *this;
    ++current_;
    return tmp;
  }

  // Bidirectional iterator support
  template <typename It = Iterator,
            typename = std::enable_if_t<std::is_base_of_v<std::bidirectional_iterator_tag,
                                                          typename std::iterator_traits<It>::iterator_category>>>
  transform_iterator& operator--() {
    --current_;
    return *this;
  }
  template <typename It = Iterator,
            typename = std::enable_if_t<std::is_base_of_v<std::bidirectional_iterator_tag,
                                                          typename std::iterator_traits<It>::iterator_category>>>
  transform_iterator operator--(int) {
    transform_iterator tmp = *this;
    --current_;
    return tmp;
  }

  // Random access iterator support
  template <typename It = Iterator,
            typename = std::enable_if_t<std::is_base_of_v<std::random_access_iterator_tag,
                                                          typename std::iterator_traits<It>::iterator_category>>>
  transform_iterator& operator+=(difference_type n) {
    current_ += n;
    return *this;
  }

  template <typename It = Iterator,
            typename = std::enable_if_t<std::is_base_of_v<std::random_access_iterator_tag,
                                                          typename std::iterator_traits<It>::iterator_category>>>
  transform_iterator& operator-=(difference_type n) {
    current_ -= n;
    return *this;
  }

  template <typename It = Iterator,
            typename = std::enable_if_t<std::is_base_of_v<std::random_access_iterator_tag,
                                                          typename std::iterator_traits<It>::iterator_category>>>
  reference operator[](difference_type n) const {
    return f_(current_[n]);
  }

  const underlying_iterator_type& get_underlying() const { return current_; }

  // Swap support
  void swap(transform_iterator& other) noexcept(
      std::is_nothrow_swappable_v<Iterator>) {
    f_.swap(other.f_);
    std::swap(current_, other.current_);
  }

 private:
  Iterator current_{};
};

// Non-member swap function
template <typename Iterator, typename UnaryFunction>
void swap(transform_iterator<Iterator, UnaryFunction>& a,
          transform_iterator<Iterator, UnaryFunction>& b) noexcept(noexcept(a.swap(b))) {
  a.swap(b);
}

// Comparison operators
template <typename Iter1, typename Func1, typename Iter2, typename Func2>
bool operator==(const transform_iterator<Iter1, Func1>& a, const transform_iterator<Iter2, Func2>& b) {
  return a.get_underlying() == b.get_underlying();
}

template <typename Iter1, typename Func1, typename Iter2, typename Func2>
bool operator!=(const transform_iterator<Iter1, Func1>& a, const transform_iterator<Iter2, Func2>& b) {
  return !(a == b);
}

// Random access iterator support for comparison
template <typename Iter1, typename Func1, typename Iter2, typename Func2>
auto operator<(const transform_iterator<Iter1, Func1>& a, const transform_iterator<Iter2, Func2>& b)
    -> decltype(a.get_underlying() < b.get_underlying()) {
  return a.get_underlying() < b.get_underlying();
}

template <typename Iter1, typename Func1, typename Iter2, typename Func2>
auto operator>(const transform_iterator<Iter1, Func1>& a, const transform_iterator<Iter2, Func2>& b)
    -> decltype(a.get_underlying() > b.get_underlying()) {
  return a.get_underlying() > b.get_underlying();
}

template <typename Iter1, typename Func1, typename Iter2, typename Func2>
auto operator<=(const transform_iterator<Iter1, Func1>& a, const transform_iterator<Iter2, Func2>& b)
    -> decltype(a.get_underlying() <= b.get_underlying()) {
  return a.get_underlying() <= b.get_underlying();
}

template <typename Iter1, typename Func1, typename Iter2, typename Func2>
auto operator>=(const transform_iterator<Iter1, Func1>& a, const transform_iterator<Iter2, Func2>& b)
    -> decltype(a.get_underlying() >= b.get_underlying()) {
  return a.get_underlying() >= b.get_underlying();
}
// Random access iterator support for arithmetic
template <typename It, typename Func>
auto operator+(const transform_iterator<It, Func>& it, typename transform_iterator<It, Func>::difference_type n)
    -> std::enable_if_t<std::is_base_of_v<std::random_access_iterator_tag,
                                          typename std::iterator_traits<It>::iterator_category>,
                        transform_iterator<It, Func>> {
  return transform_iterator<It, Func>(it.get_underlying() + n, {});
}

template <typename It, typename Func>
auto operator+(typename transform_iterator<It, Func>::difference_type n, const transform_iterator<It, Func>& it)
    -> std::enable_if_t<std::is_base_of_v<std::random_access_iterator_tag,
                                          typename std::iterator_traits<It>::iterator_category>,
                        transform_iterator<It, Func>> {
  return transform_iterator<It, Func>(it.get_underlying() + n, {});
}

template <typename It, typename Func>
auto operator-(const transform_iterator<It, Func>& it, typename transform_iterator<It, Func>::difference_type n)
    -> std::enable_if_t<std::is_base_of_v<std::random_access_iterator_tag,
                                          typename std::iterator_traits<It>::iterator_category>,
                        transform_iterator<It, Func>> {
  return transform_iterator<It, Func>(it.get_underlying() - n, {});
}

template <typename Iter1, typename Func1, typename Iter2, typename Func2>
auto operator-(const transform_iterator<Iter1, Func1>& a, const transform_iterator<Iter2, Func2>& b)
    -> decltype(a.get_underlying() - b.get_underlying()) {
  return a.get_underlying() - b.get_underlying();
}

// Helper function to create a transform_iterator
template <typename Iterator, typename UnaryFunction>
transform_iterator<Iterator, UnaryFunction> make_transform_iterator(Iterator it, UnaryFunction f) {
  return transform_iterator<Iterator, UnaryFunction>(it, f);
}
}  // namespace onnxruntime
