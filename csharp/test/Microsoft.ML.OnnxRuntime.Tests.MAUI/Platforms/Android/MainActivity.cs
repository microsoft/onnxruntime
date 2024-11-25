using Android.App;
using Android.Content.PM;
using Android.OS;
using Android.Runtime;

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI
{
    [Activity(Theme = "@style/Maui.SplashTheme", MainLauncher = true, LaunchMode = LaunchMode.SingleTop, ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation | ConfigChanges.UiMode | ConfigChanges.ScreenLayout | ConfigChanges.SmallestScreenSize | ConfigChanges.Density)]
    [Register("ORT.CSharp.Tests.MAUI.MainActivity")]
    public class MainActivity : MauiAppCompatActivity
    {
    }
}
