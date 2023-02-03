// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import onnxruntimec

public class ORTValueTypeInfo {
    public var type: ORTValueType?
    public var tensorTypeAndShapeInfo: ORTTensorTypeAndShapeInfo?
    
    public init() {}
}
