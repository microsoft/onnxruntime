mkdir c:\models\in
copy "C:\Users\chrisd\OneDrive - Microsoft\psapi-models\PSD1.quant.onnx" C:\models\in\PSD1.quant.onnx

rmdir C:\models\out /s /q
mkdir C:\models\out

echo.
echo "List available EP devices"
echo.
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test --list_ep_devices

echo.
echo "Go NVidia!"
echo.
build\RelWithDebInfo\winappsdk_onnxruntime_perf_test -e nvtensorrtrtx --required_device_type gpu -t 10 -I "C:\models\in\PSD1.quant.onnx"
echo.

echo "Done!"
