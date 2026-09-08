# Privacy

## Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft's privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

***

### Official Builds
ONNX Runtime collects trace events with the goal of improving product quality. On Windows, it uses the platform's built-in ETW telemetry system; on supported Linux architectures, macOS, Android, and iOS, it uses the cross-platform 1DS telemetry SDK that is built into ONNX Runtime. Targets without a supported telemetry provider, including WebAssembly, tvOS, visionOS, Mac Catalyst, AIX, and RISC-V, do not include telemetry. In all cases, collection is subject to user consent and handled following Microsoft's privacy practices.

Telemetry is turned **ON** by default in the official builds ([see here](../README.md#binaries)). Both providers are accessed through ONNX Runtime's common telemetry interface (see [telemetry.h](../onnxruntime/core/platform/telemetry.h)).

### Private Builds
The build driver enables telemetry by default for supported native platforms. The standard Windows `build.bat` wrapper explicitly passes `--no_telemetry`, so private builds made with that wrapper perform no data collection. Targets without a supported provider and builds that disable C++ exceptions automatically exclude telemetry. For information on how to disable telemetry in other builds, see [Disabling Telemetry](#disabling-telemetry) below.

#### Technical Details

**Windows.** The Windows provider uses the [TraceLogging](https://docs.microsoft.com/en-us/windows/win32/tracelogging/trace-logging-about) API for its implementation. This enables ONNX Runtime trace events to be collected by the operating system, and based on user consent, this data may be periodically sent to Microsoft servers following GDPR and privacy regulations for anonymity and data access controls. Windows ML and ONNX Runtime C APIs allow Trace Logging to be turned on/off (see [API pages](../README.md#api-documentation) for details); there are equivalent APIs in the C#, Python, and Java language bindings as well.

**Non-Windows (Linux, macOS, Android, iOS).** These platforms use the cross-platform 1DS SDK (cpp_client_telemetry) to send the same trace events to Microsoft's telemetry backend over HTTPS. Based on user consent, this data is handled following GDPR and privacy regulations for anonymity and data access controls. ONNX Runtime C APIs allow 1DS to be turned on/off (see [API pages](../README.md#api-documentation) for details); there are equivalent APIs in the C#, Python, and Java language bindings as well.

For ways to disable telemetry, see the [Disabling Telemetry](#disabling-telemetry) section below.

### Disabling Telemetry

Telemetry can be disabled in any of these ways:

- **Disable it at build time.** Pass `--no_telemetry` to `build.py` or `build.sh`. This omits the 1DS provider from non-Windows builds and disables the Microsoft telemetry configuration on Windows. The standard Windows `build.bat` wrapper does this automatically. Unsupported targets and exception-free builds never include telemetry.
- **Disable all telemetry at runtime (non-Windows).** Set `ORT_DISABLE_TELEMETRY=1` before ONNX Runtime initializes. This prevents the uploader, events, and persistent device identifier from being created for the process lifetime.
- **Disable non-essential events via the API.** The C API (and the C#, Python, and Java bindings) can suppress non-essential telemetry. ONNX Runtime may already have emitted a minimal initialization event before the API can be called. On **Windows**, ETW events are recorded only when an external trace session is collecting.

Telemetry-enabled static Linux builds use static curl and mbedTLS. FetchContent packages include those archives;
vcpkg builds resolve them from the same vcpkg installation. A static consumer that links another curl or mbedTLS
copy into the same final binary must build ORT with `--no_telemetry` to avoid ordinary static-symbol collisions.
