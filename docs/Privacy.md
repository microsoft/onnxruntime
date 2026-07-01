# Privacy

## Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

### Private Builds
No data collection is performed when using your private builds built from source code.

### Official Builds
ONNX Runtime does not maintain any independent telemetry collection mechanisms outside of what is provided by the platforms it supports. However, where applicable, ONNX Runtime will take advantage of platform-supported telemetry systems to collect trace events with the goal of improving product quality.

Telemetry is turned **ON** by default in the official Windows builds distributed in their respective package management repositories ([see here](../README.md#binaries)), where it is implemented with the platform ETW provider. Builds for other platforms can additionally be compiled with the cross-platform 1DS telemetry provider by configuring with `--use_telemetry`; this is **not** enabled in the default builds. Data collection is implemented via 'Platform Telemetry' per vendor platform providers (see [telemetry.h](../onnxruntime/core/platform/telemetry.h)).

#### Technical Details
The Windows provider uses the [TraceLogging](https://docs.microsoft.com/en-us/windows/win32/tracelogging/trace-logging-about) API for its implementation. This enables ONNX Runtime trace events to be collected by the operating system, and based on user consent, this data may be periodically sent to Microsoft servers following GDPR and privacy regulations for anonymity and data access controls. 

Windows ML and onnxruntime C APIs allow Trace Logging to be turned on/off (see [API pages](../README.md#api-documentation) for details).
For the ways to disable telemetry, see the [Disabling Telemetry](#disabling-telemetry) section below. 
There are equivalent APIs in the C#, Python, and Java language bindings as well.

### Disabling Telemetry

Telemetry can be disabled in any of these ways:

- **Don't build it in.** Telemetry is only compiled when configuring with `--use_telemetry` (`onnxruntime_USE_TELEMETRY=OFF` is the default), so a build without that flag collects no data.
- **At runtime, via environment variable.** Set `ORT_TELEMETRY_DISABLED=1` (also accepts `true`/`yes`/`on`/`y`, case-insensitive) before ONNX Runtime initializes. On the non-Windows 1DS provider this prevents the telemetry uploader from being created. The same variable is also honored by ONNX Runtime GenAI.
- **At runtime, via the API.** The C API (and the C#, Python, and Java bindings) expose calls to turn telemetry on/off. On Windows, ETW is passive — events are only emitted when an external trace session is collecting.
