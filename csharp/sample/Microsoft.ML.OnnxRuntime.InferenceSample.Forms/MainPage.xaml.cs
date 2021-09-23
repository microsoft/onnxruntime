using System;
using Xamarin.Forms;

namespace Microsoft.ML.OnnxRuntime.InferenceSample.Forms
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();

            Console.WriteLine("Using API");
            InferenceSampleApi.Execute();
            Console.WriteLine("Done");

            Console.WriteLine("Using API (using platform-specific session options)");
            InferenceSampleApi.Execute(App.PlatformSessionOptions);
            Console.WriteLine("Done");
        }
    }
}