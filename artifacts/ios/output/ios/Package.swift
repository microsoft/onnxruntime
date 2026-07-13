// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "onnxruntime-ios-custom-1.27.0",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "onnxruntime",
            targets: [
                "onnxruntime"
            ]
        )
    ],
    targets: [
        .binaryTarget(
            name: "onnxruntime",
            path: "framework_out/onnxruntime.xcframework"
        )
    ]
)
