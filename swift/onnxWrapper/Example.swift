// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import objcOnnxWrapper

public class OnnxWrapper {
    public private(set) var session: ORTSession
    public private(set) var env: ORTEnv
    
    public init(modelPath: String, threadCount: Int32 = 1) {
        do {
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.verbose)
            try options.setIntraOpNumThreads(threadCount)
            // Create the ORTSession
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch {
            fatalError(error.localizedDescription)
        }
    }
}
