export OMP_NUM_THREADS=12
echo "Multi-threads. Threads num:"
echo $OMP_NUM_THREADS

python benchmark.py -e "onnxruntime" "torch" "torchscript" --overwrite --optimize_onnx -q
python benchmark.py -e "onnxruntime" "torch" "torchscript" --overwrite --optimize_onnx

export OMP_NUM_THREADS=1
echo "Single-thread. Threads num:"
echo $OMP_NUM_THREADS

python benchmark.py -e "onnxruntime" "torch" "torchscript" --overwrite --optimize_onnx -q --single_thread
python benchmark.py -e "onnxruntime" "torch" "torchscript" --overwrite --optimize_onnx --single_thread
