using System.Reflection;
using Foundation;
using UIKit;
using Xunit.Runner;
using Xunit.Sdk;

namespace Microsoft.ML.OnnxRuntime.Tests.iOS
{
    [Register("AppDelegate")]
    public partial class AppDelegate : RunnerAppDelegate
    {
        public override bool FinishedLaunching(UIApplication app, NSDictionary options)
        {
            AddExecutionAssembly(typeof(ExtensibilityPointFactory).Assembly);

#if __NATIVE_DEPENDENCIES_EXIST__
            AddTestAssembly(Assembly.GetExecutingAssembly());
#endif

            return base.FinishedLaunching(app, options);
        }
    }
}