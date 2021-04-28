// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

// this type represents a compile-time list of types
template <typename... T>
struct TypeList {};

}  // namespace onnxruntime

// type list type containing the given types
// Note: this is useful for passing TypeLists to macros which don't accept the
//       comma-separated template arguments
#define ORT_TYPE_LIST(...) ::onnxruntime::TypeList<__VA_ARGS__>
