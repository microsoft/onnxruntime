using Microsoft.Extensions.Logging;

namespace MauiModelTester;

public static class MauiProgram
{
    public static MauiApp CreateMauiApp()
    {
        var builder = MauiApp.CreateBuilder();
        builder.UseMauiApp<App>().ConfigureFonts(fonts =>
                                                 {
                                                     fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                                                     fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                                                 });

#if DEBUG
        // NOTE: Enabling this does allow Debug.WriteLine to work for debugging C# code.
        //       However it seems to kill native logging on Android using __android_log_print that ORT and
        //       onnxruntime-extensions use, at least in the emulator. Due to that, enabled if you want to debug C#
        //       code and disable to debug native code.
        //
        // Add the extension debug logger so Debug.WriteLine output shows up in the Output window when running in VS
        // builder.Logging.AddDebug();
        // System.Diagnostics.Debug.WriteLine("Debug output enabled.");
#endif

        return builder.Build();
    }
}
