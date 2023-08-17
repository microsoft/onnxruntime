// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_checkpoint_internal.h"

#include <optional>
#include <string>
#include <variant>
#import "cxx_api.h"

#import "error_utils.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTCheckpoint {
  std::optional<Ort::CheckpointState> _checkpoint;
}

- (nullable instancetype)initWithPath:(NSString*)path
                                error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _checkpoint = Ort::CheckpointState::LoadCheckpoint(path.UTF8String);
    return self;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)saveCheckpointToPath:(NSString*)path
          withOptimizerState:(BOOL)includeOptimizerState
                       error:(NSError**)error {
  try {
    Ort::CheckpointState::SaveCheckpoint([self CXXAPIOrtCheckpoint], path.UTF8String, includeOptimizerState);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)addIntPropertyWithName:(NSString*)name
                         value:(int64_t)value
                         error:(NSError**)error {
  try {
    [self CXXAPIOrtCheckpoint].AddProperty(name.UTF8String, value);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)addFloatPropertyWithName:(NSString*)name
                           value:(float)value
                           error:(NSError**)error {
  try {
    [self CXXAPIOrtCheckpoint].AddProperty(name.UTF8String, value);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)addStringPropertyWithName:(NSString*)name
                            value:(NSString*)value
                            error:(NSError**)error {
  try {
    [self CXXAPIOrtCheckpoint].AddProperty(name.UTF8String, value.UTF8String);
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (nullable NSString*)getStringPropertyWithName:(NSString*)name error:(NSError**)error {
  try {
    Ort::Property value = [self CXXAPIOrtCheckpoint].GetProperty(name.UTF8String);
    if (std::string* str = std::get_if<std::string>(&value)) {
      return [NSString stringWithUTF8String:str->c_str()];
    }
    ORT_CXX_API_THROW("Property is not a string.", ORT_INVALID_ARGUMENT);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (int64_t)getIntPropertyWithName:(NSString*)name error:(NSError**)error {
  try {
    Ort::Property value = [self CXXAPIOrtCheckpoint].GetProperty(name.UTF8String);
    if (int64_t* i = std::get_if<int64_t>(&value)) {
      return *i;
    }
    ORT_CXX_API_THROW("Property is not an integer.", ORT_INVALID_ARGUMENT);
  }
  ORT_OBJC_API_IMPL_CATCH(error, 0)
}

- (float)getFloatPropertyWithName:(NSString*)name error:(NSError**)error {
  try {
    Ort::Property value = [self CXXAPIOrtCheckpoint].GetProperty(name.UTF8String);
    if (float* f = std::get_if<float>(&value)) {
      return *f;
    }
    ORT_CXX_API_THROW("Property is not a float.", ORT_INVALID_ARGUMENT);
  }
  ORT_OBJC_API_IMPL_CATCH(error, 0.0f)
}

- (Ort::CheckpointState&)CXXAPIOrtCheckpoint {
  return *_checkpoint;
}

@end

NS_ASSUME_NONNULL_END
