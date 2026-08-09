## About the ETW Sink

The ETW Sink (ONNXRuntimeTraceLoggingProvider) allows ONNX semi-structured printf style logs to be output via ETW.

ETW makes it easy and useful to only enable and listen for events with great performance, and when you need them instead of only at compile time.
Therefore ONNX will preserve any existing loggers and log severity [provided at compile time](/docs/FAQ.md?plain=1#L7).

However, when the provider is enabled a new ETW logger sink will also be added and the severity separately controlled via ETW dynamically.

- Provider GUID: 929DD115-1ECB-4CB5-B060-EBD4983C421D
- Keyword: Logs (0x2) keyword per [logging.h](/include/onnxruntime/core/common/logging/logging.h)
- Level: 1-5 ([CRITICAL through VERBOSE](https://learn.microsoft.com/en-us/windows/win32/api/evntprov/ns-evntprov-event_descriptor)) [mapping](/onnxruntime/core/platform/windows/logging/etw_sink.cc) to [ONNX severity](/include/onnxruntime/core/common/logging/severity.h) in an intuitive manner

Notes:
- The ETW provider must be enabled prior to session creation, as that as when internal logging setup is complete
- Other structured ETW logs are output via the other Microsoft.ML.ONNXRuntime ETW provider. Both used together are recommended

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

## View the trace output

### Setup
- Install Windows Performance Analyzer (Preview) from the Windows Store - <https://www.microsoft.com/en-us/p/windows-performance-analyzer-preview/9n58qrw40dfw>
- Or from the ADK <https://docs.microsoft.com/en-us/windows-hardware/get-started/adk-install>
  - You get to select components when installing, so can select just the performance toolkit.
  - Overview of the steps is at <https://msdn.microsoft.com/en-us/library/windows/desktop/dn904629(v=vs.85).aspx> if you want more detail.

### Viewing

Open TraceCaptureFile.etl file in Windows Performance Analyzer.

Expand the "System Activity" dropdown in the left pane, and double-click "Generic Events".
That should open events in an Analysis window in the right pane. You should see an event
with provider of ONNXRuntimeTraceLoggingProvider, and the log message output.
