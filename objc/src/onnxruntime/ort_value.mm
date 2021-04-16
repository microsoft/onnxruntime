// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "onnxruntime/ort_value.h"
#import "onnxruntime/ort_value_internal.h"

#include "onnxruntime/error_utils.h"

#include "core/common/optional.h"
#include "core/session/onnxruntime_cxx_api.h"

static ONNXTensorElementDataType get_onnx_element_data_type(ORTElementDataType value) {
    switch(value) {
        case ORTElementDataTypeFloat:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case ORTElementDataTypeInt32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case ORTElementDataTypeUndefined:
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

@implementation ORTValue {
    onnxruntime::optional<Ort::Value> _value;
}

-(instancetype) initTensorWithData:(NSMutableData*) data
                       elementType:(ORTElementDataType) type
                             shape:(const int64_t*)shape
                          shapeLen:(size_t)shape_len
                             error:(NSError**) error {
    self = [super init];
    if (self) {
        try {
            const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            const auto element_type = get_onnx_element_data_type(type);
            _value = Ort::Value::CreateTensor(memory_info, data.mutableBytes, data.length, shape, shape_len, element_type);
        } catch (const Ort::Exception& e) {
            [ORTErrorUtils saveErrorCode:e.GetOrtErrorCode()
                             description:e.what()
                                 toError:error];
            self = nil;
        }
    }
    return self;
}

- (Ort::Value*) handle {
    return &(*_value);
}

@end
