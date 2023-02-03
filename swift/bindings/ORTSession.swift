// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import onnxruntimec

public class ORTSession {
    public var session: ORTSession?
    public init(env: ORTEnv?, modelPath path: String?, sessionOptions: ORTSessionOptions?) throws {
    }
    
    public func run(withInputs inputs: [String : ORTValue]?, outputs: [String : ORTValue]?, runOptions: ORTRunOptions?) throws {
    }
    
    public func run(withInputs inputs: [String : ORTValue]?, outputNames: Set<String>?, runOptions: ORTRunOptions?) throws -> [String: ORTValue]? {
    }
    
    public func inputNames() throws -> [String]? {
    }
    
    public func overridableInitializerNames() throws -> [String]? {
    }
    
    public func outputNames() throws -> [String]? {
    }
}
