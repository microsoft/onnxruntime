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
            // Problem is the Appium exit leaves a zombie process and this breaks future use as it has multiple
            // processes for that app even though only one is valid.
            // Setting appTopLevelWindow below instead:
            // App = Environment.ProcessPath,            
        };

        // TODO: in theory this should check for no windows or a null PlatformView and throw
        // https://github.com/microsoft/WinAppDriver/issues/1091#issuecomment-597437731
        IntPtr hwnd = ((MauiWinUIWindow)Application.Current.Windows[0].Handler.PlatformView).WindowHandle;
        windowsOptions.AddAdditionalAppiumOption("appTopLevelWindow", hwnd.ToString("x"));

        // Note there are many more options that you can use to influence the app under test according to your needs
        return new WindowsDriver(new Uri("http://127.0.0.1:4723/wd/hub"), windowsOptions);
    }
}