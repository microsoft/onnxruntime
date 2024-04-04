// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"
#include "core/providers/coreml/model/host_utils.h"

#import <Foundation/Foundation.h>

namespace onnxruntime {
namespace coreml {
namespace util {

bool HasRequiredBaseOS() {
  return CoreMLVersion() >= 3;
}

int32_t CoreMLVersion() {
  if (HAS_COREML7_OR_LATER)
    return 7;
  if (HAS_COREML6_OR_LATER)
    return 6;
  if (HAS_COREML5_OR_LATER)
    return 5;
  if (HAS_COREML4_OR_LATER)
    return 4;
  if (HAS_COREML3_OR_LATER)
    return 3;

  return -1;
}

std::string GetTemporaryFilePath() {
  // Get temporary directory for user.
  NSURL* temporary_directory_url = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];

#if !defined(NDEBUG)
  std::string path_override = Env::Default().GetEnvironmentVar(kOverrideModelOutputDirectoryEnvVar);
  if (!path_override.empty()) {
    NSString* ns_path_override = [NSString stringWithUTF8String:path_override.c_str()];
    temporary_directory_url = [NSURL fileURLWithPath:ns_path_override isDirectory:YES];
  }
#endif

  // Generate a Unique file name to use.
  NSString* temporary_filename = [[NSProcessInfo processInfo] globallyUniqueString];

  // make it easy to see who generated it
  temporary_filename = [@"onnxruntime-" stringByAppendingString:temporary_filename];

  // Create URL to that file.
  NSURL* temporary_file_url = [temporary_directory_url URLByAppendingPathComponent:temporary_filename];

  return std::string([[temporary_file_url path] UTF8String]);
}

}  // namespace util
}  // namespace coreml
}  // namespace onnxruntime
