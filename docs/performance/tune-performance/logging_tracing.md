---
title: Logging & Tracing
grand_parent: Performance
parent: Tune performance
nav_order: 2
---

# Logging & Tracing

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Developer Logging

ONNX Runtime has built-in cross-platform internal [printf style logging LOGS()](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/macros.h). This logging is available to configure in *production builds* for a dev **using the API**.

There will likely be a performance penalty for using the default sink output (stdout) with higher log severity levels.

### log_severity_level
[Python](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions.log_severity_level) (below) - [C/C++ CreateEnv](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a22085f699a2d1adb52f809383f475ed1) / [OrtLoggingLevel](https://onnxruntime.ai/docs/api/c/group___global.html#ga1c0fbcf614dbd0e2c272ae1cc04c629c) - [.NET/C#](https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.SessionOptions.html#Microsoft_ML_OnnxRuntime_SessionOptions_LogSeverityLevel)
```python
sess_opt = SessionOptions()
sess_opt.log_severity_level = 0 // Verbose
sess = ort.InferenceSession('model.onnx', sess_opt)
```

### Note
Note that [log_verbosity_level](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions.log_verbosity_level) is a separate setting and only available in DEBUG custom builds.

## Tracing About

Tracing is a super-set of logging in that tracing 
- Includes the previously mentioned logging
- Adds tracing events that are more structured than printf style logging
- Can be integrated with a larger tracing eco-system of the OS, such that
  - Tracing from multiple systems with ONNX, OS system level, and user-mode software that uses ONNX can be combined
  - Timestamps are high resolution and consistent with other traced components
  - Can log at high performance with a high number of events / second.
  - Events are not logged via stdout, but instead usually via a high performance in memory sink
  - Can be enabled dynamically at run-time to investigate issues including in production systems

Currently, only Tracelogging combined with Windows ETW is supported, although [TraceLogging](https://github.com/microsoft/tracelogging) is cross-platform and support for other OSes instrumentation systems could be added.

## Tracing - Windows

There are 2 main ONNX Runtime TraceLogging providers that can be enabled at run-time that can be captured with Windows [ETW](https://learn.microsoft.com/en-us/windows-hardware/test/weg/instrumenting-your-code-with-etw)

### Quickstart Tracing with WPR

On Windows, you can use Windows Performance Recorder ([WPR](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options)) to capture a trace. The 2 providers covered below are already configured in these WPR profiles.

- Download [ort.wprp](https://github.com/microsoft/onnxruntime/blob/main/ort.wprp) and [etw_provider.wprp](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/platform/windows/logging/etw_provider.wprp) (these could also be combined later)

```dos
wpr -start ort.wprp -start etw_provider.wprp
echo Repro the issue allowing ONNX to run
wpr -stop onnx.etl -compress
```

### ONNXRuntimeTraceLoggingProvider
Beginning in ONNX Runtime 1.17 the [ONNXRuntimeTraceLoggingProvider](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/platform/windows/logging/HowToValidateEtwSinkOutput.md) can also be enabled.

This will dynamically trace with high-performance the previously mentioned LOGS() macro printf logs that were previously only controlled by log_severity_level. A user or developer tracing with this provider will have the log severity level set dynamically with what ETW level they provide at run-time.

Provider Name: ONNXRuntimeTraceLoggingProvider  
Provider GUID: 929DD115-1ECB-4CB5-B060-EBD4983C421D  
Keyword: Logs (0x2) keyword per [logging.h](https://github.com/ivberg/onnxruntime/blob/9cb97ee507b9b45d4a896f663590083e7e7568ac/include/onnxruntime/core/common/logging/logging.h#L83)  
Level: 1 (CRITICAL ) through 5 (VERBOSE) per [TraceLoggingLevel](https://learn.microsoft.com/en-us/windows/win32/api/traceloggingprovider/nf-traceloggingprovider-tracelogginglevel#remarks)  

### Microsoft.ML.ONNXRuntime

The [Microsoft.ML.ONNXRuntime](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/platform/windows/telemetry.cc#L47) provider provides structured logging.  

Provider Name: Microsoft.ML.ONNXRuntime  
Provider GUID: 3a26b1ff-7484-7484-7484-15261f42614d  
Keywords: Multiple per [logging.h](https://github.com/ivberg/onnxruntime/blob/9cb97ee507b9b45d4a896f663590083e7e7568ac/include/onnxruntime/core/common/logging/logging.h#L81)  
Level: 1 (CRITICAL ) through 5 (VERBOSE) per [TraceLoggingLevel](https://learn.microsoft.com/en-us/windows/win32/api/traceloggingprovider/nf-traceloggingprovider-tracelogginglevel#remarks)  
Note: This provider supports ETW [CaptureState](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/capturestateonsave) (Rundown) for logging state for example when a trace is saved

ORT 1.17 includes new events logging session options and EP provider options

#### Profiling

Microsoft.ML.ONNXRuntime can also output profiling events. That is covered in [profiling](profiling-tools.md)

### WinML

WindowsML has it's own tracing providers that be enabled in addition the providers above

- Microsoft.Windows.WinML - d766d9ff-112c-4dac-9247-241cf99d123f
- Microsoft.Windows.AI.MachineLearning - BCAD6AEE-C08D-4F66-828C-4C43461A033D