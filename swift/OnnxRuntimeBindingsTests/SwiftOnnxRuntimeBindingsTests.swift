// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import XCTest
import Foundation
@testable import OnnxRuntimeBindings

final class SwiftOnnxRuntimeBindingsTests: XCTestCase {
    let modelPath: String = Bundle.module.url(forResource: "single_add.basic", withExtension: "ort")!.path

    func testGetVersionString() throws {
        do {
            let version = ORTVersion()
            XCTAssertNotNil(version)
        } catch let error {
            XCTFail(error.localizedDescription)
        }
    }

    func testCreateSession() throws {
        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.verbose)
            try options.setIntraOpNumThreads(1)
            // Create the ORTSession
            _ = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch let error {
            XCTFail(error.localizedDescription)
        }
    }

    func testAppendCoreMLEP() throws {
        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
            let sessionOptions: ORTSessionOptions = try ORTSessionOptions()
            let coreMLOptions: ORTCoreMLExecutionProviderOptions = ORTCoreMLExecutionProviderOptions()
            coreMLOptions.enableOnSubgraphs = true
            try sessionOptions.appendCoreMLExecutionProvider(with: coreMLOptions)

            XCTAssertTrue(ORTIsCoreMLExecutionProviderAvailable())
            _ = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)
        } catch let error {
            XCTFail(error.localizedDescription)
        }
    }

    func testAppendXnnpackEP() throws {
        do {
            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
            let sessionOptions: ORTSessionOptions = try ORTSessionOptions()
            let XnnpackOptions: ORTXnnpackExecutionProviderOptions = ORTXnnpackExecutionProviderOptions()
            XnnpackOptions.intra_op_num_threads = 2
            try sessionOptions.appendXnnpackExecutionProvider(with: XnnpackOptions)

            XCTAssertTrue(ORTIsCoreMLExecutionProviderAvailable())
            _ = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)
        } catch let error {
            XCTFail(error.localizedDescription)
        }
    }
}
