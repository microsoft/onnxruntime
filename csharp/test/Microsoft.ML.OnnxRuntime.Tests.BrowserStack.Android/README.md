# BrowserStack Android test
This project will run the Android MAUI tests on BrowserStack, which allows you to run automated tests on a variety of mobile devices.

## Context
Microsoft.ML.OnnxRuntime.Tests.MAUI uses DeviceRunners.VisualRunners to allow running the unit tests (found in Microsoft.ML.OnnxRuntime.Tests.Common) across multiple devices. DeviceRunners.VisualRunners provides a simple UI with a button that will run the unit tests and a panel with the unit test results. 

In order to automate the process of running the unit tests across mobile devices, Appium is used for UI testing orchestration (it provides a way to interact with the UI), and BrowserStack automatically runs these Appium tests across different mobile devices.

This project does not include the capability to start an Appium server locally or attach to a local emulator or device. 

## Build & run instructions
### Requirements
* A BrowserStack account with access to App Automate
    * You can set BrowserStack credentials as environment variables as shown [here](https://www.browserstack.com/docs/app-automate/appium/getting-started/c-sharp/nunit/integrate-your-tests#CLI)
* ONNXRuntime NuGet package
    1. You can either download the [stable NuGet package](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) then follow the instructions from [NativeLibraryInclude.props file](../Microsoft.ML.OnnxRuntime.Tests.Common/NativeLibraryInclude.props) to use the downloaded .nupkg file
    2. Or follow the [build instructions](https://onnxruntime.ai/docs/build/android.html) to build the Android package locally
* The dotnet workloads for maui and maui-android, which will not always automatically install correctly
    1. `dotnet workload install maui`
    2. `dotnet workload install maui-android`
* [Appium](https://appium.io/docs/en/latest/quickstart/) and the [UiAutomator2 driver](https://appium.io/docs/en/latest/quickstart/uiauto2-driver/)

### Run instructions
1. Build the Microsoft.ML.OnnxRuntime.Tests.MAUI project into a signed APK.
    1. Run the following: `dotnet publish -c Release -f net8.0-android` in the Microsoft.ML.OnnxRuntime.Tests.MAUI directory.
    2. Search for the APK files generated. They should be located in `bin\Release\net8.0-android\publish`. 
    3. If they're in a different location, edit the `browserstack.yml` file to target the path to the signed APK.
2. Ensure you've set the BrowserStack credentials as environment variables.
3. Run the following in the Microsoft.ML.OnnxRuntime.Tests.Android.BrowserStack directory: `dotnet test`
4. Navigate to the [BrowserStack App Automate dashboard](https://app-automate.browserstack.com/dashboard/v2/builds) to see your test running!

## Troubleshooting & Resources
### BrowserStack Resources
- [Configuration docs](https://www.browserstack.com/docs/app-automate/appium/sdk-params#test-context) for browserstack.yml
- [Configuration generator](https://www.browserstack.com/docs/app-automate/capabilities) for browserstack.yml
- [Integration guide](https://www.browserstack.com/docs/app-automate/appium/getting-started/c-sharp/nunit/integrate-your-tests#CLI)

### Troubleshooting
- Issues building the MAUI app: 
    - Make sure that the maui and maui-android workloads are installed correctly by running `dotnet workload list`
    - If you believe the issues are workload related, you can also try running `dotnet workload repair` (this has personally never worked for me)
    - Try running `dotnet clean`. However, this does not fully remove all the previous intermediaries. If you're still running into the errors, manually deleting the bin and obj folders can sometimes resolve them. 
- After building the MAUI app, try installing on an emulator and clicking the "Run All" button to ensure that everything is working. (If you are missing the ONNXRuntime package, it will not show up as an error until you click "Run All".)
    - Running the MAUI app from Visual Studio will not replicate running it through BrowserStack. Instead, use `adb install [path to signed apk]` to install the app then use the emulator to launch the app.
- Issues with the Android.BrowserStack test app: there is an Appium Doctor package on npm -- run `npm install @appium/doctor --location=global` then `appium-doctor --android` and follow the directed instructions. Some errors with Appium on Android will not appear until runtime.
- Connection refused by Appium server: this can happen if you already have an Appium server running locally. If you do, stop the Appium server then try `dotnet test` again.
- App is crashing on BrowserStack or it emits an error that it cannot run this APK file: make sure that you are passing in the correct signed APK from the publish folder. 
- It appears that a test runs on CLI but a build is not launched on BrowserStack: this happens when the BrowserStack Test Adapter cannot find the browserstack.yml file (which has to be named "browserstack.yml" -- do not be tricked by BrowserStack's article on custom-named configuration files)
