// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import onnxruntimec

public class ORTSessionOptions {
    public init() throws {
    }
    
    public func appendExecutionProvider(_ providerName: String?, providerOptions: [String : String]?) throws {
    }
    
    public func setIntraOpNumThreads(_ intraOpNumThreads: Int) throws {
    }
    
    public func setGraphOptimizationLevel(_ graphOptimizationLevel: ORTGraphOptimizationLevel) throws {
    }
    
    public func setOptimizedModelFilePath(_ optimizedModelFilePath: String?) throws {
    }
    
    public func setLogID(_ logID: String?) throws {
    }
    
    public func setLogSeverityLevel(_ loggingLevel: ORTLoggingLevel) throws {
    }

    public func addConfigEntry(withKey key: String?, value: String?) throws {
    }

    public func registerCustomOps(usingFunction registrationFuncName: String?) throws {
    }
}
