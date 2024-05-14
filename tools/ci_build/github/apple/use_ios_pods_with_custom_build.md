# How to create and use iOS CocoaPods pods with a custom ONNX Runtime build

If you require a custom build of ONNX Runtime, you can create CocoaPods pods with your custom build locally and use them from a Podfile.

**Prerequisite** - The custom build must be able to be done with [build_apple_framework.py](./build_apple_framework.py).

To do a custom build and create the pods, run [build_and_assemble_apple_pods.py](./build_and_assemble_apple_pods.py).
Use the `--help` argument to see more information.

## Example usage

In the following example, we will have a staging directory to contain the local pod files: `/path/to/staging/dir`.

Our custom build will use a custom reduced operator kernel config file: `/path/to/custom.config`

Run the script:
```bash
python3 tools/ci_build/github/apple/build_and_assemble_apple_pods.py \
  --staging-dir /path/to/staging/dir \
  --include-ops-by-config /path/to/custom.config \
  --build-settings-file tools/ci_build/github/apple/default_mobile_ios_framework_build_settings.json
```

This will do a custom build and create the pod package files for it in `/path/to/staging/dir`.

Next, update the Podfile to use the local pods:
```diff
-  pod 'onnxruntime-mobile-objc'
+  pod 'onnxruntime-mobile-objc', :path => "/path/to/staging/dir/onnxruntime-mobile-objc"
+  pod 'onnxruntime-mobile-c', :path => "/path/to/staging/dir/onnxruntime-mobile-c"
```

Note:
The `onnxruntime-mobile-objc` pod depends on the `onnxruntime-mobile-c` pod.
If the released `onnxruntime-mobile-objc` pod is used, this dependency is automatically handled.
However, if a local `onnxruntime-mobile-objc` pod is used, the local `onnxruntime-mobile-c` pod that it depends on also needs to be specified in the Podfile.
