call run.py to calibrate, quantize and run the quantized model, e.g.:
python run.py --input_model mobilenetv2-7.onnx --output_model mobilenetv2-7.quant.onnx --calibrate_dataset ./test_images/