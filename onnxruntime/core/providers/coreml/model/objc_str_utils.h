// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#import <Foundation/Foundation.h>

namespace onnxruntime::coreml::util {

// Converts a UTF8 const char* to an NSString. Throws on failure.
// Prefer this to directly calling [NSString stringWithUTF8String:] as that may return nil.
NSString* _Nonnull Utf8StringToNSString(const char* _Nonnull utf8_str);

}  // namespace onnxruntime::coreml::util
