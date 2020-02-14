# Privacy

## Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

### Private Builds
No data collection is performed when using your private builds.

### Official Builds
Currently telemetry is only implemented for Windows builds, but may be expanded in the future to cover other platforms. Telemetry is turned OFF by default while this feature is in BETA. When the feature moves from BETA to RELEASE, developers should expect telemetry to be ON by default when using the Official Builds. This is implemented via 'Platform Telemetry' per vendor platform providers (see telemetry.h).

#### Technical Details
The Windows provider uses the [TraceLogging](https://docs.microsoft.com/en-us/windows/win32/tracelogging/trace-logging-about) API for its implementation.

Windows ML and onnxruntime C APIs allow telemetry collection to be turned on/off (see API pages for details).
In all Official builds:
1. Telemetry collection is *disabled* for calls to the ORT API. See additional information about how to enable and disable telemetry in the [C API: Telemetry](./C_API.md#telemetry) section on the C-API documentation page.
2. Telemetry collection is *enabled* for calls to the Windows ML API.
