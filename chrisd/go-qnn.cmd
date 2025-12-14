mkdir c:\models\in
copy "C:\Users\chrisd\OneDrive - Microsoft\psapi-models\PSD1.quant.onnx" C:\models\in\PSD1.quant.onnx

rmdir C:\models\out /s /q
mkdir C:\models\out

echo "List available EP devices"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe --list_ep_devices

echo "[extreme_power_saver] Step 1: Create cache file"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe -e qnn --required_device_type npu -i "htp_performance_mode|extreme_power_saver soc_model|60 htp_graph_finalization_optimization_mode|3" -C "ep.context_enable|1 ep.context_file_path|C:\models\out\PSD1.quant.onnx.cache.onnx" -r 1 -I "C:\models\in\PSD1.quant.onnx"

echo "[extreme_power_saver] Step 2: Use cache file"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe -e qnn --required_device_type npu -i "htp_performance_mode|extreme_power_saver soc_model|60 htp_graph_finalization_optimization_mode|3" -C "ep.context_enable|1" -t 10 -I "C:\models\out\PSD1.quant.onnx.cache.onnx"

rmdir C:\models\out /s /q
mkdir C:\models\out

echo "[burst] Step 1: Create cache file"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe -e qnn --required_device_type npu -i "htp_performance_mode|burst soc_model|60 htp_graph_finalization_optimization_mode|3" -C "ep.context_enable|1 ep.context_file_path|C:\models\out\PSD1.quant.onnx.cache.onnx" -r 1 -I "C:\models\in\PSD1.quant.onnx"

echo "[burst] Step 2: Use cache file"
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test.exe -e qnn --required_device_type npu -i "htp_performance_mode|burst soc_model|60 htp_graph_finalization_optimization_mode|3" -C "ep.context_enable|1" -t 10 -I "C:\models\out\PSD1.quant.onnx.cache.onnx"
