# Windows ML Documentation Updates

This directory contains updated ONNX Runtime documentation that highlights WinML as the preferred Windows path and includes deprecation notices for DirectML.

## Files Created/Updated

### New Documentation Files

- **`execution-providers/DirectML-ExecutionProvider.md`**: DirectML documentation with deprecation notice
- **`get-started/with-windows.md`**: Windows getting started guide prioritizing WinML
- **`install/index.md`**: Installation guide with WinML as the recommended Windows option

### Updated Files

- **`WinML_principles.md`**: Updated to include WinML's relationship to ONNX Runtime

## Key Changes Implemented

1. **Deprecation Notice**: Added clear deprecation notice to DirectML Execution Provider page
2. **WinML Highlighting**: Created Windows Getting Started page that prioritizes WinML over DirectML
3. **Installation Instructions**: Updated installation guide to clearly promote WinML for Windows users
4. **Developer Guidance**: Added explanation of WinML's relationship to ONNX Runtime:
   - WinML uses the same ONNX Runtime APIs
   - It dynamically selects the best execution provider (EP) based on hardware
   - It simplifies deployment for Windows developers

## Reference Links

All documentation includes links to the official WinML overview: [Windows ML Overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)

## Integration Notes

These documentation files are structured to match the expected website structure mentioned in the issue:
- `/docs/execution-providers/DirectML-ExecutionProvider.html`
- `/docs/get-started/with-windows.html`  
- `/docs/install/#winml-installs`

The documentation is written in Markdown format for easy integration with the documentation website build process.