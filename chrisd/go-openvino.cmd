mkdir c:\models\in
copy "C:\Users\chrisd\OneDrive - Microsoft\psapi-models\PSD1.quant.onnx" C:\models\in\PSD1.quant.onnx

rmdir C:\models\out /s /q
mkdir C:\models\out

echo "List available EP devices"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe --list_ep_devices

echo "Step 1: Create cache file"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe -e openvino --required_device_type npu -r 1 -C "ep.context_enable|1 ep.context_embed_mode|0 ep.context_file_path|C:\models\out\PSD1.quant.onnx.cache.onnx" -I c:\models\in\PSD1.quant.onnx

echo "Step 2: Use cache file"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe -e openvino --required_device_type npu -t 10 -I C:\models\out\PSD1.quant.onnx.cache.onnx

echo "Done"
