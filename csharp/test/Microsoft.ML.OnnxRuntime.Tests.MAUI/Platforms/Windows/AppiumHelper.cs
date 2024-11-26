using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// Extension of shared class to provide platform specific CreateDriver method
/// </summary>
internal partial class AppiumHelper
{
    internal static AppiumDriver CreateDriver()
    {
        var windowsOptions = new AppiumOptions
        {
            // Specify windows as the driver, typically don't need to change this
            AutomationName = "Windows",
            // Always Windows for Windows
            PlatformName = "Windows",
            // The identifier of the deployed application to test. Needs to be the path of the exe.
            App = Environment.ProcessPath,
        };

        // Note there are many more options that you can use to influence the app under test according to your needs
        return new WindowsDriver(new Uri("http://127.0.0.1:4723/wd/hub"), windowsOptions);
    }
}