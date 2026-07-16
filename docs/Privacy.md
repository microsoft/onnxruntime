# Privacy

## Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

### Private Builds
On Windows, private builds compiled from source perform no data collection. On the non-Windows platforms, telemetry is enabled by default — including in builds compiled from source — so it is present unless you turn it off (see [Disabling Telemetry](#disabling-telemetry)).

### Official Builds
ONNX Runtime collects trace events with the goal of improving product quality. On Windows it uses the platform's built-in ETW telemetry system; on the non-Windows platforms it uses the cross-platform 1DS telemetry SDK that is built into ONNX Runtime. In all cases, collection is subject to user consent and handled following Microsoft's privacy practices.

Telemetry is turned **ON** by default in the official builds ([see here](../README.md#binaries)): on Windows it is implemented with the platform ETW provider, and on the non-Windows platforms — Linux, macOS, Android, and iOS — with the cross-platform 1DS telemetry provider (the standard build scripts enable the `--use_telemetry` build option for these). WebAssembly builds do not include telemetry. Both providers are accessed through ONNX Runtime's common telemetry interface (see [telemetry.h](../onnxruntime/core/platform/telemetry.h)).

#### Technical Details

**Windows.** The Windows provider uses the [TraceLogging](https://docs.microsoft.com/en-us/windows/win32/tracelogging/trace-logging-about) API for its implementation. This enables ONNX Runtime trace events to be collected by the operating system, and based on user consent, this data may be periodically sent to Microsoft servers following GDPR and privacy regulations for anonymity and data access controls. Windows ML and onnxruntime C APIs allow Trace Logging to be turned on/off (see [API pages](../README.md#api-documentation) for details); there are equivalent APIs in the C#, Python, and Java language bindings as well.

**Non-Windows (Linux, macOS, Android, iOS).** These platforms use the cross-platform 1DS SDK (cpp_client_telemetry) to send the same trace events to Microsoft's telemetry backend over HTTPS. Based on user consent, this data is handled following GDPR and privacy regulations for anonymity and data access controls.

For the ways to disable telemetry, see the [Disabling Telemetry](#disabling-telemetry) section below.
For the telemetry platform matrix and build behavior, see [ONNX Runtime Telemetry](Telemetry.md).

### Disabling Telemetry

Telemetry can be disabled in any of these ways:

- **Don't build it in.** The telemetry provider is only compiled when configuring with `--use_telemetry`, so a build configured without it collects no data.
- **At runtime, via environment variable (non-Windows).** Set `ORT_TELEMETRY_DISABLED=1` (also accepts `true`/`yes`/`on`/`y`, case-insensitive) before ONNX Runtime initializes to disable non-essential telemetry.
- **At runtime, via the API.** The C API (and the C#, Python, and Java bindings) expose calls to turn telemetry on/off. On **Windows**, ETW events are still emitted if an external trace session is collecting.
