// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/model/objc_str_utils.h"

#include "core/common/common.h"

namespace onnxruntime::coreml::util {

NSString* _Nonnull Utf8StringToNSString(const char* _Nonnull utf8_str) {
  NSString* result = [NSString stringWithUTF8String:utf8_str];
  ORT_ENFORCE(result != nil, "NSString conversion failed.");
  return result;
}

}  // namespace onnxruntime::coreml::util
