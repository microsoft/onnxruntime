## Guidelines

### CUDA Plugin Execution Provider

- The EP name for the CUDA Plugin EP (returned by `OrtEpDevice.EpName`) is `CUDAExecutionProvider`.
- The registration name passed to `RegisterExecutionProviderLibrary` is arbitrary and chosen by the application.
