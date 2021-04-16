#include "core/session/onnxruntime_cxx_api.h"

#import "onnxruntime/ort_value.h"

@interface ORTValue ()
- (Ort::Value*) handle;
@end
