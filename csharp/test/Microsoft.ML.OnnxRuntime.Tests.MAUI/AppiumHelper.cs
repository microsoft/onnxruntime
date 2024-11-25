using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Service;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// Helper to start/stop Appium. The platform specific code provides the createFunc to create the AppiumDriver.
/// </summary>
internal static partial class AppiumHelper
{
    private static AppiumDriver? driver;
    private static AppiumLocalService? appiumLocalService;

    public const string DefaultHostAddress = "127.0.0.1";
    public const int DefaultHostPort = 4723;

    public static AppiumDriver App => driver ?? throw new NullReferenceException("AppiumDriver is null");

    public static void Start()
    {
        if (driver is not null)
        {
            throw new InvalidOperationException("AppiumDriver is already started.");
        }

        StartAppiumLocalServer();
        driver = CreateDriver(); // this function is implemented in the platform specific code
    }

    public static void Stop()
    {
        driver?.Quit();

        // If an Appium server was started locally above, make sure we clean it up here
        DisposeAppiumLocalServer();
        driver = null;
    }

    private static void StartAppiumLocalServer(string host = DefaultHostAddress, int port = DefaultHostPort)
    {
        if (appiumLocalService is not null)
        {
            return;
        }

        var builder = new AppiumServiceBuilder()
            .WithIPAddress(host)
            .UsingPort(port);

        // TODO: Only set this if unset and valid. 
        Environment.SetEnvironmentVariable("NODE_BINARY_PATH", @"C:\Users\scmckay\AppData\Local\fnm_multishells\147872_1732254096669\node.exe");

        // Start the server with the builder
        appiumLocalService = builder.Build();
        appiumLocalService.Start();
    }

    public static void DisposeAppiumLocalServer()
    {
        appiumLocalService?.Dispose();
        appiumLocalService = null;    
    }
}
