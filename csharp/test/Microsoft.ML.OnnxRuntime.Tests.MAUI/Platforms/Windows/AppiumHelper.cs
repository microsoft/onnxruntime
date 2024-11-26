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
            DeviceName = "WindowsPC",
            // The identifier of the deployed application to test
            // Note sure where this is defined, but log from the 'Deployment' tab in the Visual Studio 'Output' window
            // shows the full package name.
            // App = "ORT.CSharp.Tests.MAUI_1.0.0.1_x64__9zz4h110yvjzm", 
            App = @"D:\src\github\ort\csharp\test\Microsoft.ML.OnnxRuntime.Tests.MAUI\bin\Debug\net8.0-windows10.0.19041.0\win10-x64\Microsoft.ML.OnnxRuntime.Tests.MAUI.exe",
        };

        // Note there are many more options that you can use to influence the app under test according to your needs
        return new WindowsDriver(new Uri("http://127.0.0.1:4723/wd/hub"), windowsOptions);
    }
}