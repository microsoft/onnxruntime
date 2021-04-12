# Testing Android Changes using the Emulator

See [Android build instructions](https://www.onnxruntime.ai/docs/how-to/build.html#android) and information on the locations of the various files referred to here.

## Install the emulator

If using Android Studio this is included in the base install.

If using sdkmanager install the emulator by running 
  - `sdkmanager[.bat] --install "emulator"`

The emulator will emulate the Android device not its processor, so you need to build onnxruntime 
with an ABI that's valid for the host machine, and install a system image that matches. 
For example you can emulate a Pixel 3 device on an Intel 64-bit host, but it will require a binary built against x86_64
rather than the arm64-v8a ABI of the real device.

e.g. on Intel 64-bit you would build with `--android_abi x86_64` to create onnxruntime libraries/executables that can be run on the Android emulator

## Create the device to emulate

### Android Studio

Tools->AVD Manager->Create Virtual Device...

Once created the emulator can be started using the 'play' button in AVD Manager.

### sdkmanager

First install a system image. Use `sdkmanager --list` to see the available system images. 

e.g. `sdkmanager --install "system-images;android-27;default;x86_64`

Create the virtual device using avdmanager[.bat] (which should be in the same directory as sdkmanager[.bat]).

e.g. `avdmanager create avd -n android27_emulator -k "system-images;android-27;default;x86_64"`

Run the emulator
e.g. `.../Android/emulator/emulator -avd android27_emulator -partition-size 2048 -no-snapshot -no-audio`

## Testing running a model on the emulator directly

Use ADB to copy files and execute commands

https://developer.android.com/studio/command-line/adb

ADB is located in the 'platform-tools' folder of the SDK directory. 

Copy onnx_test_runner and the directory of the model to test (in ONNX test directory format) to /data/local/tmp.

```
adb push <onnxruntime repo>/build/<platform>/<config>/onnx_test_runner /data/local/tmp/
adb push <onnxruntime repo>/build/<platform>/<config>/testdata/transform/gemm_activation_fusion /data/local/tmp/
```

e.g. on Windows that might be 
```
<Android SDK path>\platform-tools\adb.exe push <onnxruntime repo>\build\Windows\Debug\onnx_test_runner /data/local/tmp/testdata
<Android SDK path>\platform-tools\adb.exe push <onnxruntime repo>\build\Windows\Debug\testdata\transform\gemm_activation_fusion /data/local/tmp/
```

You may need to change permissions to make onnx_test_runner executable: 
`<Android SDK path>\platform-tools\adb.exe shell chmod +x /data/local/tmp/onnx_test_runner`

Run onnx_test_runner with the model directory: 
`<Android SDK path>\platform-tools\adb.exe shell 'cd /data/local/tmp && ./onnx_test_runner gemm_activation_fusion'`

The output should look something like this:

```
D:\Android\platform-tools> .\adb.exe shell 'cd /data/local/tmp && ./onnx_test_runner gemm_activation_fusion'
result:
        Models: 1
        Total test cases: 1
                Succeeded: 1
                Not implemented: 0
                Failed: 0
        Stats by Operator type:
                Not implemented(0):
                Failed:
Failed Test Cases:
```