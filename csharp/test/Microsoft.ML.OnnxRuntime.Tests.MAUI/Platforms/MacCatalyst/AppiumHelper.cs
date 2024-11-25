using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Enums;
using OpenQA.Selenium.Appium.Mac;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// Extension of shared class to provide platform specific CreateDriver method
/// </summary>
internal static partial class AppiumHelper
{
    internal static AppiumDriver CreateDriver()
    {
        var macOptions = new AppiumOptions
        {
            // Specify mac2 as the driver, typically don't need to change this
            AutomationName = "mac2",
            // Always Mac for Mac
            PlatformName = "Mac",
            // The full path to the .app file to test. TBD what the correct path is or the best way to determine it.
            // Might need to define it in the csproj. 
            App = "/path/to/MauiApp/bin/Debug/net8.0-maccatalyst/maccatalyst-x64/Microsoft.ML.OnnxRuntime.Tests.MAUI.app",
        };

        // Setting the Bundle ID is required, else the automation will run on Finder
        macOptions.AddAdditionalAppiumOption(IOSMobileCapabilityType.BundleId, "ORT.CSharp.Tests.MAUI");

        // Note there are many more options that you can use to influence the app under test according to your needs

        return new MacDriver(macOptions);
    }
}