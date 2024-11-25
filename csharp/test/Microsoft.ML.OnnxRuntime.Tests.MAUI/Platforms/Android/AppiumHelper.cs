using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Android;
using OpenQA.Selenium.Appium.Enums;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// Extension of shared class to provide platform specific CreateDriver method
/// </summary>
internal static partial class AppiumHelper
{
    internal static AppiumDriver CreateDriver() {
        var androidOptions = new AppiumOptions
        {
            // Specify UIAutomator2 as the driver, typically don't need to change this
            AutomationName = "UIAutomator2",
            // Always Android for Android
            PlatformName = "Android",

            // RELEASE BUILD SETUP
            // The full path to the .apk file
            // This only works with release builds because debug builds have fast deployment enabled
            // and Appium isn't compatible with fast deployment
            // TBC: Not sure if this path is correct. 
            // App = Path.Join(TestContext.CurrentContext.TestDirectory, "../../../../Microsoft.ML.OnnxRuntime.Tests.MAUI/bin/Release/net8.0-android/ORT.CSharp.Tests.MAUI-Signed.apk"),
            // END RELEASE BUILD SETUP
        };

        // DEBUG BUILD SETUP
        // If you're running your tests against debug builds you'll need to set NoReset to true
        // otherwise appium will delete all the libraries used for Fast Deployment on Android
        // Release builds have Fast Deployment disabled
        // https://learn.microsoft.com/xamarin/android/deploy-test/building-apps/build-process#fast-deployment
        androidOptions.AddAdditionalAppiumOption(MobileCapabilityType.NoReset, "true");
        androidOptions.AddAdditionalAppiumOption(AndroidMobileCapabilityType.AppPackage, "ORT.CSharp.Tests.MAUI");

        //Make sure to set [Register("ORT.CSharp.Tests.MAUI.MainActivity")] on the MainActivity of your android application
        androidOptions.AddAdditionalAppiumOption(AndroidMobileCapabilityType.AppActivity, $"ORT.CSharp.Tests.MAUI.MainActivity");
        // END DEBUG BUILD SETUP


        // Specifying the avd option will boot the emulator for you
        // make sure there is an emulator with the name below
        // If not specified, make sure you have an emulator booted
        //androidOptions.AddAdditionalAppiumOption("avd", "pixel_5_-_api_33");

        // Note there are many more options that you can use to influence the app under test according to your needs

        return new AndroidDriver(androidOptions);
    }
}