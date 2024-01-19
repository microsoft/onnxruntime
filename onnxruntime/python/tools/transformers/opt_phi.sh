
# fp32 opt
#python /wy/code/wangye/phi2_fission/onnxruntime/python/tools/transformers/optimizer.py --input /wy/onnx_models/phi2/mlflow_model_folder/data/onnx_models/phi-2_decoder_fp32.onnx --output /wy/onnx_models/phi2/mlflow_model_folder/data/onnx_models/cpu_fp32/phi-2_decoder_fp32_cpu_opt.onnx --model_type phi --use_external_data_format #--float16

# gqa
python /wy/code/wangye/phi2_fission/onnxruntime/python/tools/transformers/optimizer.py --input /wy/onnx_models/phi2/mlflow_model_folder/data/onnx_models/phi-2_decoder_fp32.onnx --output /wy/onnx_models/phi2/mlflow_model_folder/data/onnx_models/cuda/phi-2_decoder_cuda_opt.onnx --model_type phi --use_external_data_format --float16

# fp16 opt
#python /wy/code/wangye/phi2_fission/onnxruntime/python/tools/transformers/optimizer.py --input /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp32.onnx --output /wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp16_opt.onnx --model_type phi --use_external_data_format --float16