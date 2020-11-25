## Validating ETW Sink unit test output

## Setup
Install Windows Performance Toolkit from <https://docs.microsoft.com/en-us/windows-hardware/get-started/adk-install>
You get to select components when installing, so can select just the performance toolkit.

Overview of the steps is at <https://msdn.microsoft.com/en-us/library/windows/desktop/dn904629(v=vs.85).aspx> if you want more detail.

## Capturing ETW trace output

From an elevated prompt, assuming wpr.exe and wpa.exe are in the path (if not, most likely they can be found under C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\) do the following:

Start the ETW tracing
  `<path to repo>\onnxruntime\test\platform\windows\logging> wpr -start .\etw_provider.wprp`

  Note: If you add '-start GeneralProfile' a huge amount of data will be captured. Not necessary for checking log messages are produced. It will also result in a huge number of ngen.exe calls when you attempt to stop logging the first type.

Run the ETW sink unit tests

 * Run the tests in etw_sink_test.cc and then stop WPR.
  * This can be done by executing the specific tests in Visual Studio,
  * or by running the exe with all the tests from the platform library
    * Assuming debug build on Windows run `<path to repo>\build\Windows\Debug\Debug\onnxruntime_test_framework.exe`

Stop the ETW tracing
    `<path to repo>\onnxruntime\test\platform\windows\logging> wpr -stop TraceCaptureFile.etl EtwSinkTest`

## View the output

Open TraceCaptureFile.etl file Windows Performance Analyzer.

Expand the "System Activity" dropdown in the left pane, and double-click "Generic Events".
That should open events in an Analysis window in the right pane. You should see an event
with provider of ONNXRuntimeTraceLoggingProvider, and the log message output.
