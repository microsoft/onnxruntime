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
        // Add the extension debug logger so Debug.WriteLine output shows up in the Output window when running in VS
        builder.Logging.AddDebug();
        System.Diagnostics.Debug.WriteLine("Debug output enabled.");
#endif

        return builder.Build();
    }
}
