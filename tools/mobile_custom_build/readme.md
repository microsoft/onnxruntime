# Unified Android and iOS Custom ONNX Runtime Build

This wrapper coordinates the Android and iOS custom ONNX Runtime builds.

It ensures both platforms receive the same:

- ONNX Runtime branch or tag
- ONNX Runtime repository
- reduced operator configuration
- build configuration
- custom package version

## Important platform requirements

Android uses the existing Docker-based custom build.

iOS must run natively on macOS because the build requires:

- Xcode
- the iPhoneOS SDK
- the iPhoneSimulator SDK
- `xcodebuild`
- Apple framework packaging tools

Building both platforms with one command therefore requires macOS with Docker installed.

## Build Android and iOS together

```bash
python3 tools/mobile_custom_build/build_custom_mobile_packages.py \
  ./artifacts/mobile \
  --platform all \
  --onnxruntime_branch_or_tag v1.27.0 \
  --include_ops_by_config ./configs/required_operators.config \
  --android_build_settings tools/ci_build/github/android/default_full_aar_build_settings.json \
  --ios_build_settings tools/ios_custom_build/configs/default_minimal_ios_framework_build_settings.json \
  --config Release \
  --package_version 1.27.0-custom.1 \
  --clean_ios
