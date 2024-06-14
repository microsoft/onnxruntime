find . -name \*.yml -exec sed -i 's/onnxruntime-Win2022-GPU-A10/onnxruntime-Win2022-GPU-A10-beta/g' {} \;
find . -name \*.yml -exec sed -i 's/onnxruntime-Win-CPU-2022'/onnxruntime-Win2022-CPU-beta'/g' {} \;
find . -name \*.yml -exec sed -i 's/onnxruntime-Win2022-CPU-training-AMD/onnxruntime-Win2022-CPU-beta/g' {} \;
