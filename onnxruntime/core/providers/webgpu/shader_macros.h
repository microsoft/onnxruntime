// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// macro "D": append to the ostream only in debug build
//
// Usage example:
//
// ss << "error code: " << err_code D(" (") << D(err_msg) D(")");
//
// This resolves to: (debug build)
// ss << "error code: " << err_code << " (" << err_msg << ")";
//
// This resolves to: (release build)
// ss << "error code: " << err_code;

#ifdef D
#undef D
#endif

#ifndef NDEBUG  // if debug build
#define D(str) << str
#else
#define D(str)
#endif

// macro "DSS" append to the ostream only in debug build
// (assume variable "ss" is in scope)
//
// Usage example:
//
// DSS << "detail error message: " << err_msg;
//
// This resolves to: (debug build)
// ss << "detail error message: " << err_msg;
//
// This resolves to: (release build)
// if constexpr (false) ss << "detail error message: " << err_msg;  // no-op

#ifdef DSS
#undef DSS
#endif

#ifndef NDEBUG  // if debug build
#define DSS ss
#else
#define DSS \
  if constexpr (false) ss
#endif

// macro "SS" - use function call style to append to the ostream
// (assume variable "ss" is in scope)
//
// Usage example:
//
// SS("error code: ", err_code, " (", err_msg, ")");
//
// This resolves to:
// ss << "error code: " << err_code << " (" << err_msg << ")";

#ifdef SS
#undef SS
#endif

#define SS(...) ::onnxruntime::detail::MakeStringImpl(ss, __VA_ARGS__)
