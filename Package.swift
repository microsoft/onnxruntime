// swift-tools-version: 5.6
//   The swift-tools-version declares the minimum version of Swift required to build this package and MUST be the first
//   line of this file. 5.6 is required to support zip files for the pod archive binaryTarget.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// A user of the Swift Package Manager (SPM) package will consume this file directly from the ORT github repository.
// For context, the end user's config will look something like:
//
//     dependencies: [
//       .package(url: "https://github.com/microsoft/onnxruntime", branch: "rel-1.15.0"),
//       ...
//     ],
//
// NOTE: The direct consumption creates a somewhat complicated setup to 'release' a new version of the ORT SPM package.
//       TBD: how to manage the release process

import PackageDescription
import class Foundation.ProcessInfo

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
                dependencies: ["onnxruntime"],
                path: "objectivec",
                exclude: ["test", "docs", "ReadMe.md", "format_objc.sh"],
                cxxSettings: [
                    .define("SPM_BUILD"),
                    .unsafeFlags(["-std=c++17",
                                  "-fobjc-arc-exceptions"
                                 ]),
                ], linkerSettings: [
                    .unsafeFlags(["-ObjC"]),
                ]),
        .testTarget(name: "OnnxRuntimeBindingsTests",
                    dependencies: ["OnnxRuntimeBindings"],
                    path: "swift/OnnxRuntimeBindingsTests",
                    resources: [
                        .copy("Resources/single_add.basic.ort")
                    ]),
    ]
)

// Add the ORT iOS Pod archive as a binary target.
//
// There are 2 scenarios:
//
// Release branch of ORT github repo:
//    Target will be set to the released pod archive and its checksum.
//
// Any other branch/tag of ORT github repo:
//    Invalid by default. We do not have a pod archive that is guaranteed to work
//    as the objective-c bindings may have changed since the pod archive was released.

// CI or local testing where you have built/obtained the iOS Pod archive matching the current source code.
// Requires the ORT_IOS_POD_LOCAL_PATH environment variable to be set to specify the location of the pod.
if let pod_archive_path = ProcessInfo.processInfo.environment["ORT_IOS_POD_LOCAL_PATH"] {
    // ORT_IOS_POD_LOCAL_PATH MUST be a path that is relative to Package.swift.
    //
    // To build locally, tools/ci_build/github/apple/build_and_assemble_ios_pods.py can be used
    // See https://onnxruntime.ai/docs/build/custom.html#ios
    //  Example command:
    //    python3 tools/ci_build/github/apple/build_and_assemble_ios_pods.py \
    //      --variant Full \
    //      --build-settings-file tools/ci_build/github/apple/default_full_ios_framework_build_settings.json
    //
    // This should produce the pod archive in build/ios_pod_staging, and ORT_IOS_POD_LOCAL_PATH can be set to
    // "build/ios_pod_staging/pod-archive-onnxruntime-c-???.zip" where '???' is replaced by the version info in the
    // actual filename.
    package.targets.append(Target.binaryTarget(name: "onnxruntime", path: pod_archive_path))

} else {
    // ORT 1.15.0 release
    package.targets.append(
       Target.binaryTarget(name: "onnxruntime",
                           url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-1.15.0.zip",
                           checksum: "9b41412329a73d7d298b1d94ab40ae9adb65cb84f132054073bc82515b4f5f82")
    )
}
