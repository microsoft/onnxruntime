using Microsoft.ML.OnnxRuntime;
using System;
using Xunit;
using Xunit.Abstractions;


namespace Microsoft.ML.OnnxRuntime.Tests
{
    /// <summary>
    /// Collection of OrtEnv tests that must be ran sequentially
    /// </summary>
    [Collection("Ort Inference Tests")]
    public class OrtEnvCollectionTest
    {
        [Fact(DisplayName = "EnablingAndDisablingTelemetryEventCollection")]
        public void EnablingAndDisablingTelemetryEventCollection()
        {
            var ortEnvInstance = OrtEnv.Instance();
            ortEnvInstance.DisableTelemetryEvents();

            // no-op on non-Windows builds
            // may be no-op on certain Windows builds based on build configuration

            ortEnvInstance.EnableTelemetryEvents();
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvGetVersion
    {
        [Fact(DisplayName = "GetVersionString")]
        public void GetVersionString()
        {
            var ortEnvInstance = OrtEnv.Instance();
            string versionString = ortEnvInstance.GetVersionString();
            Assert.False(versionString.Length == 0);
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvGetAvailableProviders
    {

        [Fact(DisplayName = "GetAvailableProviders")]
        public void GetAvailableProviders()
        {
            var ortEnvInstance = OrtEnv.Instance();
            string[] providers = ortEnvInstance.GetAvailableProviders();

            Assert.True(providers.Length > 0);
            Assert.Equal("CPUExecutionProvider", providers[providers.Length - 1]);

#if USE_CUDA
            Assert.True(Array.Exists(providers, provider => provider == "CUDAExecutionProvider"));
#endif
#if USE_ROCM
            Assert.True(Array.Exists(providers, provider => provider == "ROCMExecutionProvider"));
#endif
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithCustomLogLevel
    {

        [Fact(DisplayName = "TestUpdatingEnvWithCustomLogLevel")]
        public void TestUpdatingEnvWithCustomLogLevel()
        {
            var ortEnvInstance = OrtEnv.Instance();
            Assert.True(OrtEnv.IsCreated);
            ortEnvInstance.Dispose();
            Assert.False(OrtEnv.IsCreated);

            // Must be default level of warning
            ortEnvInstance = OrtEnv.Instance();
            ortEnvInstance.Dispose();
            Assert.False(OrtEnv.IsCreated);

            var envOptions = new EnvironmentCreationOptions
            {
                // Everything else is unpopulated
                logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL
            };

            ortEnvInstance = OrtEnv.CreateInstanceWithOptions(envOptions);
            Assert.True(OrtEnv.IsCreated);
            Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, ortEnvInstance.EnvLogLevel);

            ortEnvInstance.Dispose();
            Assert.False(OrtEnv.IsCreated);
            envOptions = new EnvironmentCreationOptions
            {
                // Everything else is unpopulated
                logId = "CSharpOnnxRuntimeTestLogid"
            };

            ortEnvInstance = OrtEnv.CreateInstanceWithOptions(envOptions);
            Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, ortEnvInstance.EnvLogLevel);

            // Change and see if this takes effect
            ortEnvInstance.EnvLogLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, ortEnvInstance.EnvLogLevel);
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithThreadingOptions
    {
        [Fact(DisplayName = "TestUpdatingEnvWithThreadingOptions")]
        public void TestUpdatingEnvWithThreadingOptions()
        {
            OrtEnv.Instance().Dispose();
            Assert.False(OrtEnv.IsCreated);

            using (var opt = new OrtThreadingOptions())
            {
                var envOptions = new EnvironmentCreationOptions
                {
                    threadOptions = opt
                };

                // Make sure we start anew
                var env = OrtEnv.CreateInstanceWithOptions(envOptions);
                Assert.True(OrtEnv.IsCreated);
            }
        }
    }

    public class CustomLoggerBase
    {
        // Custom logging constants
        protected static readonly string TestLogId = "CSharpTestLogId";
        protected static readonly IntPtr TestLogParam = (IntPtr)5;
        protected static int LoggingInvokes = 0;

        protected static void CustomLoggingFunction(IntPtr param,
                                                  OrtLoggingLevel severity,
                                                  string category,
                                                  string logId,
                                                  string codeLocation,
                                                  string message)
        {
            Assert.Equal(TestLogParam, param); // Passing test param
            Assert.False(string.IsNullOrEmpty(codeLocation));
            Assert.False(string.IsNullOrEmpty(message));
            LoggingInvokes++;
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithCustomLogger : CustomLoggerBase
    {

        [Fact(DisplayName = "TesEnvWithCustomLogger")]
        public void TesEnvWithCustomLogger()
        {
            // Make sure we start anew
            OrtEnv.Instance().Dispose();
            Assert.False(OrtEnv.IsCreated);
            var envOptions = new EnvironmentCreationOptions
            {
                logId = TestLogId,
                logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                loggingFunction = CustomLoggingFunction,
                loggingParam = TestLogParam
            };

            LoggingInvokes = 0;

            var env = OrtEnv.CreateInstanceWithOptions(envOptions);
            Assert.True(OrtEnv.IsCreated);

            var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
            // Trigger some logging
            using (var session = new InferenceSession(model)) ;
            Assert.True(LoggingInvokes > 0);
        }
    }

    [Collection("Ort Inference Tests")]
    public class OrtEnvWithCustomLoggerAndThreadindOptions : CustomLoggerBase
    {
        [Fact(DisplayName = "TestEnvWithCustomLoggerAndThredingOptions")]
        public void TestEnvWithCustomLoggerAndThredingOptions()
        {
            OrtEnv.Instance().Dispose();
            Assert.False(OrtEnv.IsCreated);

            using (var opt = new OrtThreadingOptions())
            {
                var envOptions = new EnvironmentCreationOptions
                {
                    logId = TestLogId,
                    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                    threadOptions = opt,
                    loggingFunction = CustomLoggingFunction,
                    loggingParam = TestLogParam
                };

                LoggingInvokes = 0;

                var env = OrtEnv.CreateInstanceWithOptions(envOptions);
                Assert.True(OrtEnv.IsCreated);

                var model = TestDataLoader.LoadModelFromEmbeddedResource("squeezenet.onnx");
                // Trigger some logging
                using (var session = new InferenceSession(model)) ;
                Assert.True(LoggingInvokes > 0);
            }
        }
    }
}

