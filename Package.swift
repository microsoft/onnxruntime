// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "onnxruntime",
    platforms: [.iOS(.v11)],
    products: [
        .library(name: "onnxruntime",
                 type: .static,
                 targets: ["OnnxWrapper"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "OnnxWrapper",
                dependencies: ["onnxruntime"],
                path: "objectivec",
                exclude: ["test", "docs"],
                cxxSettings: [
                    .unsafeFlags(["-std=c++17",
                                  "-fobjc-arc-exceptions"
                                 ]),
                ], linkerSettings: [
                    .unsafeFlags(["-ObjC"]),
                ]),
        
        // to generate checksum use `shasum -a 256 path/tp/my/zip` or `swift package compute-checksum path/tp/my/zip`
        .binaryTarget(name: "onnxruntime",
                      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-objc-1.14.0.zip",
                      checksum: "2edc19ba12de3d78716b4bdf657376302ae32e9afb793cddfe38bc7fde7e5feb"),
        
        .testTarget(name: "onnxTests",
                    dependencies: ["OnnxWrapper"],
                    path: "swift/onnxTests"),
    ]
)
