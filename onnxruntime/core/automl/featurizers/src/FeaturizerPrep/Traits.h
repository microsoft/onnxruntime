// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#pragma once
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <vector>

namespace Microsoft {
namespace Featurizer {
namespace Traits {

// XXX: Define the type
template<class T>
struct Nullable {};

/////////////////////////////////////////////////////////////////////////
///  \namespace     Traits
///  \brief         We have a range of of types we are dealing with. Many types
///                 have different ways to represent what a `NULL` value is
///                 (float has NAN for example) as well as different ways to
///                 convert the value to a string representation. By using
///                 templates combined with partial template specialization
///                 we can handle scenarios like these that vary based on the data type.
///
///                 Example: This allows us to do things like `Traits<std::int8_t>::IsNull()`
///                 and `Traits<float>::IsNull()` and let the trait itself deal with the
///                 actual implementation and allows us as developers to not worry about that.
///
///                 This benefit is magnified because we are also using templates for our
///                 transformers. When we declare that a transformer has type T = std::int8_t,
///                 we can then also use `Traits<T>::IsNull()` and the compiler will know that
///                 `T` is a `std::int8_t` and call the appropate template specialization.
///
template <typename T>
struct Traits {};

/////////////////////////////////////////////////////////////////////////
///  \namespace     Traits
///  \brief         When using partial template specilization, if the compiler
///                 cannot find a more specfic implementation of the template
///                 it will fall back to the base template and use whatever is
///                 defined there. If you have methods defined in that base template,
///                 it makes it very difficult to debug what is going on. By
///                 putting no implementation in the `Traits<>` template and
///                 having the real base struct be `TraitsImpl<>`, if you try and
///                 specify a trait that doesn't have a specilization, the compiler
///                 can detect that and throw an error during compilation.
///
///                 Example: There is no template `Traits<char>`. If you try and use it
///                 the compiler will fall back to the `Traits<>` struct which has no methods
///                 defined. Trying to then use `Traits<char>` will cause a compile time error
///                 letting you know something isn't correct.
///
template <typename T>
struct TraitsImpl {
  using nullable_type = Nullable<T>;
  static bool IsNull(nullable_type const& value) {
    return !value.is_initialized();
  }
};

template <>
struct Traits<float> : public TraitsImpl<float> {
  using nullable_type = float;
  static bool IsNull(nullable_type const& value) {
    return std::isnan(value);
  }

  // static std::string ToString(nullable_type const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<double> : public TraitsImpl<double> {
  using nullable_type = double;
  static bool IsNull(nullable_type const& value) {
    return std::isnan(value);
  }

  // static std::string ToString(nullable_type const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::int8_t> : public TraitsImpl<std::int8_t> {
  // static std::string ToString(std::int8_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::int16_t> : public TraitsImpl<std::int16_t> {
  // static std::string ToString(std::int16_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::int32_t> : public TraitsImpl<std::int32_t> {
  // static std::string ToString(std::int32_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::int64_t> : public TraitsImpl<std::int64_t> {
  // static std::string ToString(std::int64_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::uint8_t> : public TraitsImpl<std::uint8_t> {
  // static std::string ToString(std::uint8_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::uint16_t> : public TraitsImpl<std::uint16_t> {
  using nullable_type = Nullable<std::uint16_t>;
  // static std::string ToString(std::uint16_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::uint32_t> : public TraitsImpl<std::uint32_t> {
  // static std::string ToString(std::uint32_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::uint64_t> : public TraitsImpl<std::uint64_t> {
  // static std::string ToString(std::uint64_t const& value) {
  //     return std::to_string(value);
  // }
};

template <>
struct Traits<std::string> : public TraitsImpl<std::string> {
  // static std::string ToString(std::string const& value) {
  //     value;
  // }
};

template <typename T, size_t size>
struct Traits<std::array<T, size>> : public TraitsImpl<std::array<T, size>> {
  // static std::string ToString(std::array<T, size> const& value) {
  //     // Decide what to return here
  //     throw std::logic_error("Function not yet implemented");
  // }
};

template <>
struct Traits<bool> : public TraitsImpl<bool> {
  // static std::string ToString(bool const& value) {
  //     // Decide what to return here
  //     throw std::logic_error("Function not yet implemented");
  // }
};

template <typename KeyT, typename T, typename CompareT, typename AllocatorT>
struct Traits<std::map<KeyT, T, CompareT, AllocatorT>> : public TraitsImpl<std::map<KeyT, T, CompareT, AllocatorT>> {
  // static std::string ToString(std::map<KeyT, T, CompareT, AllocatorT> const& value) {
  //     // Decide what to return here
  //     throw std::logic_error("Function not yet implemented");
  // }
};

template <typename T, typename AllocatorT>
struct Traits<std::vector<T, AllocatorT>> : public TraitsImpl<std::vector<T, AllocatorT>> {
  // static std::string ToString(std::vector<T, AllocatorT> const& value) {
  //     // Decide what to return here
  //     throw std::logic_error("Function not yet implemented");
  // }
};

template <typename... Types>
struct Traits<std::function<Types...>> : public TraitsImpl<std::function<Types...>> {
  // static std::string ToString(std::function<Types ...> const& value) {
  //     // Decide what to return here
  //     throw std::logic_error("Function not yet implemented");
  // }
};

template <typename T>
struct Traits<Nullable<T>> : public TraitsImpl<Nullable<T>> {
  using nullable_type = Nullable<T>;

  // static std::string ToString(nullable_type const& value) {
  //     if (value) {
  //         return Traits<T>::ToString(value.get());
  //     }

  //     return "NULL";
  // }
};

template <typename... Types>
struct Traits<std::tuple<Types...>> : public TraitsImpl<std::tuple<Types...>> {
  // static std::string ToString(std::tuple<Types ...> const& value) {
  //     // Decide what to return here
  //     throw std::logic_error("Function not yet implemented");
  // }
};

}  // namespace Traits
}  // namespace Featurizer
}  // namespace Microsoft
