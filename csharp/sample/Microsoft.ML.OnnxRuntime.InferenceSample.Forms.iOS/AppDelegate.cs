using Foundation;
using UIKit;

namespace Microsoft.ML.OnnxRuntime.InferenceSample.Forms.iOS
{
    // The UIApplicationDelegate for the application. This class is responsible for launching the 
    // User Interface of the application, as well as listening (and optionally responding) to 
    // application events from iOS.
    [Register("AppDelegate")]
    public partial class AppDelegate : global::Xamarin.Forms.Platform.iOS.FormsApplicationDelegate
    {
        //
        // This method is invoked when the application has loaded and is ready to run. In this 
        // method you should instantiate the window, load the UI into it and then make the window
        // visible.
        //
        // You have 17 seconds to return from this method, or iOS will terminate your application.
        //
        public override bool FinishedLaunching(UIApplication app, NSDictionary options)
        {
            global::Xamarin.Forms.Forms.Init();

#if !__NATIVE_DEPENDENCIES_EXIST__
            throw new System.Exception("The requisite onnxruntime.framework file(s) were not found. You must build the native iOS components before running this sample");
#else
            App.PlatformSessionOptions.AppendExecutionProvider_CoreML(COREMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
            LoadApplication(new App());
#endif

#pragma warning disable CS0162 // Unreachable code detected
            return base.FinishedLaunching(app, options);
#pragma warning restore CS0162 // Unreachable code detected
        }
    }
}