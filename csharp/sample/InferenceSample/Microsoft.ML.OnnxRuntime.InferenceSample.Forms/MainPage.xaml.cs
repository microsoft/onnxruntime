using System;
using System.Threading;
using System.Threading.Tasks;
using Xamarin.Essentials;
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

            OutputLabel.Text = "Press 'Run Tests'.\n";
        }

        private readonly InferenceSampleApi inferenceSampleApi;

        private async Task ExecuteTests()
        {
            Action<Label, string> addOutput = (label, text) =>
            {
                Device.BeginInvokeOnMainThread(() => { label.Text += text; });
                Console.Write(text);
            };

            OutputLabel.Text = "Testing execution\nComplete output is written to Console in this trivial example.\n\n";

            // run the testing in a background thread so updates to the UI aren't blocked
            await Task.Run(() =>
            {
                addOutput(OutputLabel, "Testing using default platform-specific session options... ");
                inferenceSampleApi.Execute();
                addOutput(OutputLabel, "done.\n");
                Thread.Sleep(1000); // artificial delay so the UI updates gradually

                // demonstrate a range of usages by recreating the inference session with different session options.
                addOutput(OutputLabel, "Testing using default platform-specific session options... ");
                inferenceSampleApi.CreateInferenceSession(SessionOptionsContainer.Create());
                inferenceSampleApi.Execute();
                addOutput(OutputLabel, "done.\n");
                Thread.Sleep(1000);

                addOutput(OutputLabel, "Testing using named platform-specific session options... ");
                inferenceSampleApi.CreateInferenceSession(SessionOptionsContainer.Create("ort_with_npu"));
                inferenceSampleApi.Execute();
                addOutput(OutputLabel, "done.\n");
                Thread.Sleep(1000);

                addOutput(OutputLabel, "Testing using default platform-specific session options via ApplyConfiguration extension... ");
                inferenceSampleApi.CreateInferenceSession(new SessionOptions().ApplyConfiguration());
                inferenceSampleApi.Execute();
                addOutput(OutputLabel, "done.\n");
                Thread.Sleep(1000);

                addOutput(OutputLabel, "Testing using named platform-specific session options via ApplyConfiguration extension... ");
                inferenceSampleApi.CreateInferenceSession(new SessionOptions().ApplyConfiguration("ort_with_npu"));
                inferenceSampleApi.Execute();
                addOutput(OutputLabel, "done.\n\n");
                Thread.Sleep(1000);
            });

            addOutput(OutputLabel, "Testing successfully completed! See the Console log for more info.");
        }

        private async void Start_Clicked(object sender, EventArgs e)
        {
            await ExecuteTests()
                .ContinueWith(
                (task) =>
                {
                    if (task.IsFaulted)
                        MainThread.BeginInvokeOnMainThread(() => DisplayAlert("Error", task.Exception.Message, "OK"));
                })
                .ConfigureAwait(false);
        }
    }
}
