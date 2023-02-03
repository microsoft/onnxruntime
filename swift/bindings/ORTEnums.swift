// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import onnxruntimec

public enum ORTLoggingLevel: Int32 {
    case verbose
    case info
    case warning
    case error
    case fatal
}

public enum ORTValueType: Int32 {
    case unknown
    case tensor
}

public enum ORTTensorElementDataType: Int32 {
    case undefined
    case float
    case int8
    case uInt8
    case int32
    case uInt32
}

public enum ORTGraphOptimizationLevel : Int32 {
    case none
    case basic
    case extended
    case all
}
