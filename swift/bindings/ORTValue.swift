// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import onnxruntimec

public class ORTValue {
    public init(tensorData: inout Data?, elementType: ORTTensorElementDataType, shape: [NSNumber]?) throws {
    }
    
    public func typeInfo() throws -> ORTValueTypeInfo? {
        return nil
    }
    
    public func tensorTypeAndShapeInfo() throws -> ORTTensorTypeAndShapeInfo? {
        return nil
    }
    
    public func tensorData() throws -> Data? {
        return nil
    }
}
