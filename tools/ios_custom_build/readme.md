# Custom ONNX Runtime iOS Build

This tooling builds a static custom ONNX Runtime XCFramework for iOS.

The iOS package is built using the same:

- ONNX Runtime branch or tag used by Android
- reduced operator configuration used by Android
- build configuration used by Android
- custom package version used by Android

## Requirements

The iOS build must run on macOS.

Required tools:

- Xcode
- Xcode command-line tools
- Python 3.9 or later
- Git
- CMake
- Ninja, where required by the selected ONNX Runtime release

Verify Xcode:

```bash
xcodebuild -version
