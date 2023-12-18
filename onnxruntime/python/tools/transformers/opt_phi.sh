#python /dev_data/wy/code/wangye/phi2_fission/onnxruntime/python/tools/transformers/optimizer.py --input /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_small.onnx --output /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_small_opt_inlined.onnx --model_type phi --use_external_data_format

# fp32 opt
python /dev_data/wy/code/wangye/phi2_fission/onnxruntime/python/tools/transformers/optimizer.py --input /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp32.onnx --output /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp32_opt.onnx --model_type phi --use_external_data_format #--float16

# fp16 opt
#python /dev_data/wy/code/wangye/phi2_fission/onnxruntime/python/tools/transformers/optimizer.py --input /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp32.onnx --output /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp16_opt.onnx --model_type phi --use_external_data_format --float16