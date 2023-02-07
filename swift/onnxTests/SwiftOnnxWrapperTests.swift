// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import XCTest
import Foundation
@testable import objcOnnxWrapper

final class SwiftOnnxWrapperTests: XCTestCase {
    func testExample() throws {
        let modelPath: String = ""
        let threadCount: Int32 = 1

        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.verbose)
            try options.setIntraOpNumThreads(threadCount)
            // Create the ORTSession
            let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
            XCTAssertNotNil(session)
        } catch {
            XCTFail(error.localizedDescription)
        }
    }
}
