using Android.App;
using Android.Content.PM;
using Android.Runtime;
using Android.OS;

namespace Microsoft.ML.OnnxRuntime.InferenceSample.Forms.Droid
{
    [Activity(Label = "Microsoft.ML.OnnxRuntime.InferenceSample.Forms", Icon = "@mipmap/icon",
              Theme = "@style/MainTheme", MainLauncher = true,
              ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation | ConfigChanges.UiMode |
                                     ConfigChanges.ScreenLayout | ConfigChanges.SmallestScreenSize)]
    public class MainActivity : global::Xamarin.Forms.Platform.Android.FormsAppCompatActivity
    {
        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            Xamarin.Essentials.Platform.Init(this, savedInstanceState);
            global::Xamarin.Forms.Forms.Init(this, savedInstanceState);

            // Register default session options configuration. This is optional.
            SessionOptionsContainer.Register((options) => { options.LogId = "Ort"; });

            // Register a named session options configuration that enables NNAPI
            SessionOptionsContainer.Register("ort_with_npu", (options) => {
                options.AppendExecutionProvider_Nnapi();
                options.LogId = "Ort+Nnapi";
            });

            LoadApplication(new App());
        }

        public override void OnRequestPermissionsResult(int requestCode, string[] permissions,
                                                        [GeneratedEnum] Android.Content.PM.Permission[] grantResults)
        {
            Xamarin.Essentials.Platform.OnRequestPermissionsResult(requestCode, permissions, grantResults);
            base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}
