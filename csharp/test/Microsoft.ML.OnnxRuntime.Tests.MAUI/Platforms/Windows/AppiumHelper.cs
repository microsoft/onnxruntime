using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// Extension of shared class to provide platform specific CreateDriver method
/// </summary>
internal static partial class AppiumHelper
{
    internal static AppiumDriver CreateDriver()
    {
        var windowsOptions = new AppiumOptions
        {
            // Specify windows as the driver, typically don't need to change this
            AutomationName = "windows",
            // Always Windows for Windows
            PlatformName = "Windows",
            // The identifier of the deployed application to test
            App = "ORT.CSharp.Tests.MAUI", 
        };

        // Note there are many more options that you can use to influence the app under test according to your needs

        return new WindowsDriver(windowsOptions);
    }
}