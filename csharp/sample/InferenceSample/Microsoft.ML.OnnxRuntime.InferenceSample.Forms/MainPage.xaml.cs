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

            Console.WriteLine("Using API (using default platform-specific session options)");
            InferenceSampleApi.Execute(SessionOptionsContainer.Create());
            Console.WriteLine("Done");

            Console.WriteLine("Using API (using named platform-specific session options)");
            InferenceSampleApi.Execute(SessionOptionsContainer.Create("ort_with_npu"));
            Console.WriteLine("Done");

            Console.WriteLine(
                "Using API (using default platform-specific session options via ApplyConfiguration extension)");
            InferenceSampleApi.Execute(new SessionOptions().ApplyConfiguration());
            Console.WriteLine("Done");

            Console.WriteLine(
                "Using API (using named platform-specific session options via ApplyConfiguration extension)");
            InferenceSampleApi.Execute(new SessionOptions().ApplyConfiguration("ort_with_npu"));
            Console.WriteLine("Done");
        }
    }
}
