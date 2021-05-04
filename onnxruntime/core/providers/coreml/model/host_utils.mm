// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include <string>
#include "host_utils.h"

namespace onnxruntime {
namespace coreml {
namespace util {

bool HasRequiredBaseOS() {
  // This may look strange, but it is required "@available(macOS ....)" to safe-guard some code
  // otherwise the compiler will spit -Wunsupported-availability-guard
  if (HAS_VALID_BASE_OS_VERSION)
    return true;
  else
    return false;
}

std::string GetTemporaryFilePath() {
  // Get temporary directory.
  NSURL* temporary_directory_url = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
  // Generate a Unique file name to use.
  NSString* temporary_filename = [[NSProcessInfo processInfo] globallyUniqueString];
  // Create URL to that file.
  NSURL* temporary_file_url = [temporary_directory_url URLByAppendingPathComponent:temporary_filename];

  return std::string([[temporary_file_url path] UTF8String]);
}

}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
