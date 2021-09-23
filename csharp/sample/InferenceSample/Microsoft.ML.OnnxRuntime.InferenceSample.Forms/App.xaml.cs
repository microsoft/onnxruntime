using Xamarin.Forms;

namespace Microsoft.ML.OnnxRuntime.InferenceSample.Forms
{
    public partial class App : Application
    {
        public static SessionOptions PlatformSessionOptions { get; set; } = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        };

        public App()
        {
            InitializeComponent();

            MainPage = new MainPage();
        }

        protected override void OnStart() {}

        protected override void OnSleep() {}

        protected override void OnResume() {}
    }
}