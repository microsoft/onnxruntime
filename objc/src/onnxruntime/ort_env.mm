// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_env.h"

#include "core/common/optional.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "onnxruntime/error_utils.h"

@implementation ORTEnv {
    onnxruntime::optional<Ort::Env> _env;
}

- (instancetype) init:(NSError **)error {
    self = [super init];
    if (self) {
        try {
            _env.emplace();
        } catch (const Ort::Exception& e) {
            [ORTErrorUtils saveErrorCode:e.GetOrtErrorCode()
                             description:e.what()
                                 toError:error];
            self = nil;
        }
    }
    return self;
}

- (void*) handle {
    return static_cast<Ort::Env::contained_type*>(*_env);
}

@end
