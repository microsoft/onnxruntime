// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "onnxruntime",
    platforms: [.iOS(.v13)],
    products: [
        .library(
            name: "onnxruntime",
            targets: ["onnxruntime"]),
    ],
    dependencies: [
    ],
    targets: [
//        .target(
//            name: "objectivec",
//            dependencies: ["onnxruntime"],
//            path: "objectivec",
//            exclude: ["test/*", "docs/*"]),
                    
        
        // to generate checksum use `shasum -a 256 path/tp/my/zip` or `swift package compute-checksum path/tp/my/zip`
        .binaryTarget(name: "onnxruntime",
                      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-mobile-c-1.13.1.zip",
                      checksum: "e7c9a70f422d25df506cd77cf0ca299003945a850ba7fea46375691166ef01cd"),
    ]
)
