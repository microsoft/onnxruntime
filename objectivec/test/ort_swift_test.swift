// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import XCTest
import onnxruntime_objc

@objc(ORTSwiftInferenceTest)
final class ORTSwiftInferenceTest: XCTestCase {
  @objc func testAddInference() {
    XCTAssertNoThrow(
      try assertAddInference(),
      "Swift Add inference must succeed with the bundled model and source-built Objective-C module.")
  }

  private func assertAddInference() throws {
    #if SWIFT_PACKAGE
    let resourceBundle = Bundle.module
    #else
    let resourceBundle = Bundle(for: ORTSwiftInferenceTest.self)
    #endif
    let modelPath = try XCTUnwrap(
      resourceBundle.path(
        forResource: "single_add.basic", ofType: "ort"),
      "Missing single_add.basic.ort; check the Bazel XCTest target's resources.")
    let environment = try ORTEnv(loggingLevel: .warning)
    let session = try ORTSession(
      env: environment, modelPath: modelPath, sessionOptions: ORTSessionOptions())

    var a: Float = 1
    var b: Float = 2
    let aData = NSMutableData(bytes: &a, length: MemoryLayout<Float>.size)
    let bData = NSMutableData(bytes: &b, length: MemoryLayout<Float>.size)
    let aTensor = try ORTValue(
      tensorData: aData, elementType: .float, shape: [NSNumber(value: 1)])
    let bTensor = try ORTValue(
      tensorData: bData, elementType: .float, shape: [NSNumber(value: 1)])
    let outputs = try session.run(
      withInputs: ["A": aTensor, "B": bTensor], outputNames: ["C"], runOptions: nil)
    let output = try XCTUnwrap(outputs["C"], "Add inference did not produce output C.")
    let data = try output.tensorData()
    XCTAssertEqual(data.length, MemoryLayout<Float>.size, "Expected one Float32 output.")
    guard data.length == MemoryLayout<Float>.size else { return }
    var actual: Float = .nan
    data.getBytes(&actual, length: MemoryLayout<Float>.size)
    XCTAssertEqual(actual, 3)
  }
}
