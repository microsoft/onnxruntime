// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import onnxruntimec

public class ORTTensorTypeAndShapeInfo {
    public init() {}
    
    public var elementType: ORTTensorElementDataType?
    public var shape: [NSNumber]?
}
