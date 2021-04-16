#include "core/session/onnxruntime_cxx_api.h"

#import "onnxruntime/ort_env.h"

@interface ORTEnv ()
- (Ort::Env*)handle;
@end