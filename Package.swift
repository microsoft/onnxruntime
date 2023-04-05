// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// swift-tools-version: 5.7
//   The swift-tools-version declares the minimum version of Swift required to build this package.
//
// A user of the Swift Package Manager (SPM) package will consume this file directly from the ORT github repository.
// For context, the end user's config will look something like:
//
//     dependencies: [
//       .package(url: "https://github.com/microsoft/onnxruntime", branch: "rel-1.14.0-spm"),
//       ...
//     ],
//
// NOTE: The direct consumption creates a somewhat complicated setup to 'release' a new version of the ORT SPM package.
//  Proposed steps:
//   - release new ORT version, including the iOS pod archive for the native ORT library
//   - update the `url:` field in this file to point to the onnxruntime-c pod archive for the release.
//     - use tools/ci_build/github/apple/update_swift_package_manager_config.py
//   - once Package.swift is updated and checked in, add a tag to the commit with the release tag and '-spm' suffix
//     - e.g. rel-1.14.0-spm for the 1.14.0 release

import PackageDescription

let package = Package(
    name: "onnxruntime",
    platforms: [.iOS(.v11)],
    products: [
        .library(name: "onnxruntime",
                 type: .static,
                 targets: ["OnnxRuntimeBindings"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "OnnxRuntimeBindings",
                dependencies: ["OnnxRuntimeNativePod"],
                path: "objectivec",
                exclude: ["test", "docs"],
                cxxSettings: [
                    .unsafeFlags(["-std=c++17",
                                  "-fobjc-arc-exceptions"
                                 ]),
                ], linkerSettings: [
                    .unsafeFlags(["-ObjC"]),
                ]),

        // Please use tools/tools/ci_build/github/apple/update_swift_package_manager_config.py to update the pod zip
        // so that the formatting and checksum is correct.
        //
        // e.g. python3 ./tools/ci_build/github/apple/update_swift_package_manager_config.py
        //        --spm_config ./Package.swift
        //        --ort_package "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-1.14.0.zip"
        .binaryTarget(name: "OnnxRuntimeNativePod",
                      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-1.14.0.zip",
                      checksum: "c89cd106ff02eb3892243acd7c4f2bd8e68c2c94f2751b5e35f98722e10c042b"),

        .testTarget(name: "OnnxRuntimeBindingsTests",
                    dependencies: ["OnnxRuntimeBindings"],
                    path: "swift/OnnxRuntimeBindingsTests",
                    resources: [
                        .copy("Resources/single_add.basic.ort")
                    ]),
    ]
)
