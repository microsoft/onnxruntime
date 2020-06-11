echo "Multi-threads. Threads num:"
echo $OMP_NUM_THREADS

#python benchmark.py -m "bert-base-cased" "roberta-base"  -e "onnxruntime" "torch" "torchscript" --overwrite --optimize_onnx -q
python benchmark.py -m "bert-base-cased" "roberta-base"  -e "onnxruntime" "torchscript" --overwrite --optimize_onnx -q
