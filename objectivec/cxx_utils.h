
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include <optional>
#include <string>
#include <variant>

NS_ASSUME_NONNULL_BEGIN
@class ORTValue;

@interface ORTUtils : NSObject

+ (NSString*)toNSString:(const std::string&)str;
+ (nullable NSString*)toNullableNSString:(const std::optional<std::basic_string<char>>&)str;

+ (std::string)toStdString:(NSString*)str;
+ (std::optional<std::basic_string<char>>)toStdOptionalString:(nullable NSString*)str;

+ (std::vector<std::string>)toStdStringVector:(NSArray<NSString*>*)strs;
+ (NSArray<NSString*>*)toNSStringNSArray:(const std::vector<std::string>&)strs;

+ (nullable NSArray<ORTValue*>*)toORTValueNSArray:(const std::vector<OrtValue*>&)values
                                            error:(NSError**)error;

+ (std::vector<const OrtValue*>)toOrtValueVector:(NSArray<ORTValue*>*)values;

@end

NS_ASSUME_NONNULL_END
