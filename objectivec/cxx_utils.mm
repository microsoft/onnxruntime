// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "cxx_utils.h"

#import <vector>
#import <optional>
#import <string>

#import "error_utils.h"

#import "ort_value_internal.h"

NS_ASSUME_NONNULL_BEGIN

namespace utils {

NSString* toNSString(const std::string& str) {
  NSString* nsStr = [NSString stringWithUTF8String:str.c_str()];
  if (!nsStr) {
    ORT_CXX_API_THROW("Failed to convert std::string to NSString", ORT_INVALID_ARGUMENT);
  }

  return nsStr;
}

NSString* _Nullable toNullableNSString(const std::optional<std::string>& str) {
  if (str.has_value()) {
    return [NSString stringWithUTF8String:str->c_str()];
  }
  return nil;
}

std::string toStdString(NSString* str) {
  return std::string([str UTF8String]);
}

std::optional<std::string> toStdOptionalString(NSString* _Nullable str) {
  if (str) {
    return std::optional<std::string>([str UTF8String]);
  }
  return std::nullopt;
}

std::vector<std::string> toStdStringVector(NSArray<NSString*>* strs) {
  std::vector<std::string> result;
  result.reserve(strs.count);
  for (NSString* str in strs) {
    result.push_back([str UTF8String]);
  }
  return result;
}

NSArray<NSString*>* toNSStringNSArray(const std::vector<std::string>& strs) {
  NSMutableArray<NSString*>* result = [NSMutableArray arrayWithCapacity:strs.size()];
  for (const std::string& str : strs) {
    NSString* nsStr = [NSString stringWithUTF8String:str.c_str()];
    if (nsStr) {
      [result addObject:nsStr];
    } else {
      ORT_CXX_API_THROW("Failed to convert std::string to NSString", ORT_INVALID_ARGUMENT);
    }
  }
  return result;
}

NSArray<ORTValue*>* _Nullable wrapUnownedCAPIOrtValues(const std::vector<OrtValue*>& values, NSError** error) {
  NSMutableArray<ORTValue*>* result = [NSMutableArray arrayWithCapacity:values.size()];
  for (size_t i = 0; i < values.size(); ++i) {
    ORTValue* val = [[ORTValue alloc] initWithCAPIOrtValue:values[i] externalTensorData:nil error:error];
    if (!val) {
      // clean up all the C API Ortvalues which haven't been wrapped by ORTValue
      for (size_t j = i; j < values.size(); ++j) {
        Ort::GetApi().ReleaseValue(values[j]);
      }
      return nil;
    }
    [result addObject:val];
  }
  return result;
}

std::vector<const OrtValue*> getWrappedCAPIOrtValues(NSArray<ORTValue*>* values) {
  std::vector<const OrtValue*> result;
  for (ORTValue* val in values) {
    result.push_back(static_cast<const OrtValue*>([val CXXAPIOrtValue]));
  }
  return result;
}

}  // namespace utils

NS_ASSUME_NONNULL_END
