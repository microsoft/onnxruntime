// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <vector>
#import <optional>
#import <string>

#import "cxx_api.h"
#import "cxx_utils.h"
#import "error_utils.h"

#import "ort_value_internal.h"

NS_ASSUME_NONNULL_BEGIN

NSString* Utils::toNSString(const std::string& str) {
  return [NSString stringWithUTF8String:str.c_str()];
}

NSString* Utils::toNullableNSString(const std::optional<std::string>& str) {
  if (str.has_value()) {
    return [NSString stringWithUTF8String:str.value().c_str()];
  }
  return nil;
}

std::string Utils::toStdString(NSString* str) {
  return std::string([str UTF8String]);
}

std::optional<std::string> Utils::toStdOptionalString(NSString* _Nullable str) {
  if (str) {
    return std::optional<std::string>([str UTF8String]);
  }
  return std::nullopt;
}

std::vector<std::string> Utils::toStdStringVector(NSArray<NSString*>* strs) {
  std::vector<std::string> result;
  for (NSString* str in strs) {
    result.push_back([str UTF8String]);
  }
  return result;
}

NSArray<NSString*>* Utils::toNSStringNSArray(const std::vector<std::string>& strs) {
  NSMutableArray<NSString*>* result = [NSMutableArray arrayWithCapacity:strs.size()];
  for (const std::string& str : strs) {
    [result addObject:[NSString stringWithUTF8String:str.c_str()]];
  }
  return result;
}

NSArray<ORTValue*>* Utils::toORTValueNSArray(const std::vector<OrtValue*>& values, NSError** error) {
  NSMutableArray<ORTValue*>* result = [NSMutableArray arrayWithCapacity:values.size()];
  for (NSUInteger i = 0; i < values.size(); ++i) {
    ORTValue* val = [[ORTValue alloc] initWithCAPIOrtValue:values[i] externalTensorData:nil error:error];
    if (!val) {
      // clean up all the output values which haven't been wrapped by ORTValue
      for (NSUInteger j = i; j < values.size(); ++j) {
        Ort::GetApi().ReleaseValue(values[j]);
      }
      return nil;
    }
    [result addObject:val];
  }
  return result;
}

std::vector<const OrtValue*> Utils::toOrtValueVector(NSArray<ORTValue*>* values) {
  std::vector<const OrtValue*> result;
  for (ORTValue* val : values) {
    result.push_back(static_cast<const OrtValue*>([val CXXAPIOrtValue]));
  }
  return result;
}

NS_ASSUME_NONNULL_END
