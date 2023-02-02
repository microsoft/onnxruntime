// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "onnxruntime",
    platforms: [.iOS(.v13)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "onnxruntime",
            targets: ["objectivec","onnxruntime"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "objectivec",
            dependencies: ["onnxruntime"],
            path: "objectivec",
            exclude: ["test/"],
            sources: ["include/", "src/"]),
        
        // to generate checksum use `shasum -a 256 path/tp/my/zip` or `swift package compute-checksum path/tp/my/zip`
        .binaryTarget(name: "onnxruntime",
                      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-mobile-c-1.13.1.zip",
                      checksum: "e7c9a70f422d25df506cd77cf0ca299003945a850ba7fea46375691166ef01cd"),
        
//        .binaryTarget(name: "objectivec",
//                      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-mobile-objc-1.13.1.zip",
//                      checksum: "03c7bbe6fc635b8eebb4fd9bb7c580a6a04098f5070b5b4296bc155aab5286f8"),
    ]
)
/**
  Source files for target onnxruntime-mobile-objc should be located under 'Sources/onnxruntime-mobile-objc', or a custom sources path can be set with the 'path' property in Package.swift
 product 'onnxruntime' is declared in the same package 'onnxruntime' and can't be used as a dependency for target 'onnxruntime-mobile-objc'.

 */
