using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Service;
using OpenQA.Selenium.Appium.Service.Options;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI;

/// <summary>
/// Helper to start/stop Appium. The platform specific code provides the createFunc to create the AppiumDriver.
/// </summary>
internal partial class AppiumHelper
{
    private AppiumLocalServer? server;
    private AppiumDriver? driver;

    public const string DefaultHostAddress = "127.0.0.1";
    public const int DefaultHostPort = 4723;

    public AppiumDriver App => driver ?? throw new NullReferenceException("AppiumDriver is null");

    public void Start()
    {
        if (driver is not null)
        {
            throw new InvalidOperationException("AppiumDriver is already started.");
        }

        server = new AppiumLocalServer();
        driver = CreateDriver(); // this function is implemented in the platform specific code
    }

    public void Stop()
    {
        driver?.Quit();
        driver = null;

        // If an Appium server was started locally above, make sure we clean it up here
        server?.Dispose();
        server = null;
    }

    class AppiumLocalServer : IDisposable
    {
        private AppiumLocalService? service;
        private bool disposedValue;

        internal AppiumLocalServer(string host = DefaultHostAddress, int port = DefaultHostPort)
        {
            if (service is not null)
            {
                return;
            }

            // this is needed on windows
            var options = new OptionCollector();
            options.AddArguments(new KeyValuePair<string, string>("--base-path", "/wd/hub"));
            options.AddArguments(new KeyValuePair<string, string>("--log-level", "debug"));

            var builder = new AppiumServiceBuilder()
                .WithIPAddress(host)
                .WithArguments(options)
                .WithLogFile(new FileInfo(@"D:\temp\appium_helper.log"))
                .UsingPort(port);

            // TODO: Only set this if unset and valid. 
            Environment.SetEnvironmentVariable("NODE_BINARY_PATH", @"C:\Users\scmckay\AppData\Local\fnm_multishells\147872_1732254096669\node.exe");

            // Start the server with the builder
            service = builder.Build();
            service.Start();
            Console.WriteLine(service.ServiceUrl);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (service != null)
            {
                service?.Dispose();
                service = null;
            }
        }

        ~AppiumLocalServer()
        {
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
