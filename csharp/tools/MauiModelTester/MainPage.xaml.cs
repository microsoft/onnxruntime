using System.Diagnostics;

namespace MauiModelTester;

public partial class MainPage : ContentPage
{
    public MainPage()
    {
        InitializeComponent();

        // See:
        // ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/
        // Core ML: https://developer.apple.com/documentation/coreml
        // NNAPI: https://developer.android.com/ndk/guides/neuralnetworks
        ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.CPU));

        if (DeviceInfo.Platform == DevicePlatform.Android)
        {
            ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.NNAPI));
        }

        if (DeviceInfo.Platform == DevicePlatform.iOS)
        {
            ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.CoreML));
        }

        // XNNPACK provides optimized CPU execution on ARM64 and ARM platforms for models using float
        var arch = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture;
        if (arch == System.Runtime.InteropServices.Architecture.Arm64 ||
            arch == System.Runtime.InteropServices.Architecture.Arm)
        {
            ExecutionProviderOptions.Items.Add(nameof(ExecutionProviders.XNNPACK));
        }

        ExecutionProviderOptions.SelectedIndex = 0; // default to CPU
        ExecutionProviderOptions.SelectedIndexChanged += ExecutionProviderOptions_SelectedIndexChanged;

        _currentExecutionProvider = ExecutionProviders.CPU;

        // start creating session in background.
#pragma warning disable CS4014 // intentionally not awaiting this task
        CreateInferenceSession();
#pragma warning restore CS4014
    }

    private async Task CreateInferenceSession()
    {
        // wait if we're already creating an inference session.
        if (_inferenceSessionCreationTask != null)
        {
            await _inferenceSessionCreationTask.ConfigureAwait(false);
            _inferenceSessionCreationTask = null;
        }

        _inferenceSessionCreationTask = CreateInferenceSessionImpl();
    }

    private async Task CreateInferenceSessionImpl()
    {
        var executionProvider = ExecutionProviderOptions.SelectedItem switch {
            nameof(ExecutionProviders.NNAPI) => ExecutionProviders.NNAPI,
            nameof(ExecutionProviders.CoreML) => ExecutionProviders.CoreML,
            nameof(ExecutionProviders.XNNPACK) => ExecutionProviders.XNNPACK,
            _ => ExecutionProviders.CPU
        };

        if (_inferenceSession == null || executionProvider != _currentExecutionProvider)
        {
            _currentExecutionProvider = executionProvider;

            // re/create an inference session with the execution provider.
            // this is an expensive operation as we have to reload the model, and should be avoided in production apps.
            _inferenceSession = new OrtInferenceSession(_currentExecutionProvider);
            await _inferenceSession.Create();

            // Display the results which at this point will have the model load time and the warmup Run() time.
            ShowResults();
        }
    }

    private void ExecutionProviderOptions_SelectedIndexChanged(object sender, EventArgs e)
    {
        // update in background
#pragma warning disable CS4014 // intentionally not awaiting this task
        UpdateExecutionProvider();
#pragma warning restore CS4014
    }

    private void OnRunClicked(object sender, EventArgs e)
    {
        // run in background
#pragma warning disable CS4014 // intentionally not awaiting this task
        RunAsync();
#pragma warning restore CS4014
    }

    private async Task UpdateExecutionProvider()
    {
        try
        {
            await SetBusy(true);
            await CreateInferenceSession();
            await SetBusy(false);
        }
        catch (Exception ex)
        {
            await SetBusy(false);
            MainThread.BeginInvokeOnMainThread(() => DisplayAlert("Error", ex.Message, "OK"));
        }
    }

    private async Task RunAsync()
    {
        try
        {
            await SetBusy(true);

            await ClearResult();

            var iterationsStr = Iterations.Text;
            int iterations = iterationsStr == string.Empty ? 10 : int.Parse(iterationsStr);

            // create inference session if it doesn't exist or EP has changed
            await CreateInferenceSession();

            await Task.Run(() => _inferenceSession.Run(iterations));

            await SetBusy(false);

            ShowResults();
        }
        catch (Exception ex)
        {
            await SetBusy(false);
            MainThread.BeginInvokeOnMainThread(() => DisplayAlert("Error", ex.Message, "OK"));
        }
    }

    private async Task SetBusy(bool busy)
    {
        await MainThread.InvokeOnMainThreadAsync(() =>
                                                 {
                                                     // disable controls that would create a new session or another
                                                     // Run call until we're done with the current Run.
                                                     ExecutionProviderOptions.IsEnabled = !busy;
                                                     RunButton.IsEnabled = !busy;

                                                     BusyIndicator.IsRunning = busy;
                                                     BusyIndicator.IsVisible = busy;
                                                 });
    }

    private async Task ClearResult()
    {
        await MainThread.InvokeOnMainThreadAsync(() =>
                                                 { TestResults.Clear(); });
    }

    private void ShowResults()
    {
        var createResults = () =>
        {
            var stats = _inferenceSession.PerfStats;
            var label = new Label { TextColor = Colors.GhostWhite };

            label.Text = $"Model load time: {stats.LoadTime.TotalMilliseconds:F4} ms\n";
            label.Text += $"Warmup run time: {stats.WarmupTime.TotalMilliseconds:F4} ms\n\n";
            label.Text += string.Join('\n', stats.GetRunStatsReport(true));
            TestResults.Add(label);

            Debug.WriteLine(label.Text);
        };

        MainThread.BeginInvokeOnMainThread(createResults);
    }

    private ExecutionProviders _currentExecutionProvider;
    private OrtInferenceSession _inferenceSession;
    private Task _inferenceSessionCreationTask;
}
