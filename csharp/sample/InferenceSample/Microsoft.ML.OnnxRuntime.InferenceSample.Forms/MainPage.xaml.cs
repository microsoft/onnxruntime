using System;
using Xamarin.Forms;

namespace Microsoft.ML.OnnxRuntime.InferenceSample.Forms
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();

            // in general create the inference session (which loads and optimizes the model) once and not per inference
            // as it can be expensive and time consuming. 
            inferenceSampleApi = new InferenceSampleApi();
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();

            Console.WriteLine("Using API");
            inferenceSampleApi.Execute();
            Console.WriteLine("Done");

            // demonstrate a range of usages by recreating the inference session with different session options. 
            Console.WriteLine("Using API (using default platform-specific session options)");
            inferenceSampleApi.CreateInferenceSession(SessionOptionsContainer.Create());
            inferenceSampleApi.Execute();
            Console.WriteLine("Done");

            Console.WriteLine("Using API (using named platform-specific session options)");
            inferenceSampleApi.CreateInferenceSession(SessionOptionsContainer.Create("ort_with_npu"));
            inferenceSampleApi.Execute();
            Console.WriteLine("Done");

            Console.WriteLine(
                "Using API (using default platform-specific session options via ApplyConfiguration extension)");
            inferenceSampleApi.CreateInferenceSession(new SessionOptions().ApplyConfiguration());
            inferenceSampleApi.Execute();
            Console.WriteLine("Done");

            Console.WriteLine(
                "Using API (using named platform-specific session options via ApplyConfiguration extension)");
            inferenceSampleApi.CreateInferenceSession(new SessionOptions().ApplyConfiguration("ort_with_npu"));
            inferenceSampleApi.Execute();
            Console.WriteLine("Done");
        }

        private readonly InferenceSampleApi inferenceSampleApi;
    }
}
