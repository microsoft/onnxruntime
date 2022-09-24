// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

// hack in a version of gsl::narrow for builds with no exceptions
// adapted from "gsl/narrow"
#if defined(ORT_NO_EXCEPTIONS)

namespace gsl {

// narrow() : a checked version of narrow_cast() that throws if the cast changed the value
template <class T, class U, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
constexpr T narrow(U u) noexcept(false) {
  constexpr const bool is_different_signedness =
      (std::is_signed<T>::value != std::is_signed<U>::value);

  const T t = gsl::narrow_cast<T>(u);  // While this is technically undefined behavior in some cases (i.e., if the source value is of floating-point type
                                       // and cannot fit into the destination integral type), the resultant behavior is benign on the platforms
                                       // that we target (i.e., no hardware trap representations are hit).

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
  if (static_cast<U>(t) != u || (is_different_signedness && ((t < T{}) != (u < U{})))) {
    std::terminate();
  }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  return t;
}

template <class T, class U, typename std::enable_if<!std::is_arithmetic<T>::value>::type* = nullptr>
constexpr T narrow(U u) noexcept(false) {
  const T t = gsl::narrow_cast<T>(u);

  if (static_cast<U>(t) != u) {
    std::terminate();
  }

  return t;
}

}  // namespace gsl

#endif  // !defined(ORT_NO_EXCEPTIONS)
