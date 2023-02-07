// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "onnxruntime",
    platforms: [.iOS(.v16)],
    products: [
        .library(name: "onnxruntime",
                 type: .static,
                 targets: ["objcOnnxWrapper"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "objcOnnxWrapper",
                dependencies: ["onnxruntime"],
                path: "swift/objcOnnxWrapper",
                cxxSettings: [
                    .unsafeFlags(["-std=c++17",
                                  "-fobjc-arc-exceptions"
                                 ]),
                ], linkerSettings: [
                    .unsafeFlags(["-ObjC"]),
                ]),
        
        // to generate checksum use `shasum -a 256 path/tp/my/zip` or `swift package compute-checksum path/tp/my/zip`
        .binaryTarget(name: "onnxruntime",
                      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-1.13.1.zip",
                      checksum: "f83c847421a6198d060af5e8535314cfe540a208552db7870b838ebc90cdaa3f"),
        
        .testTarget(name: "onnxTests",
                    dependencies: ["objcOnnxWrapper"],
                    path: "swift/onnxTests"),
    ]
)
