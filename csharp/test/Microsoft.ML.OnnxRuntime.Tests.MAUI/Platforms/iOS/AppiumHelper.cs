using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.iOS;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// Extension of shared class to provide platform specific CreateDriver method
/// </summary>
internal partial class AppiumHelper
{
    internal static AppiumDriver CreateDriver()
    {
        var iOSOptions = new AppiumOptions
        {
            // Specify XCUITest as the driver, typically don't need to change this
            AutomationName = "XCUITest",
            // Always iOS for iOS
            PlatformName = "iOS",
            // iOS Version
            PlatformVersion = "17.0",
            // Don't specify if you don't want a specific device
            // DeviceName = "iPhone 15 Pro",
            // The full path to the .app file to test or the bundle id if the app is already installed on the device.
            App = "ORT.CSharp.Tests.MAUI",
        };

        // Note there are many more options that you can use to influence the app under test according to your needs

        return new IOSDriver(iOSOptions);
    }
}