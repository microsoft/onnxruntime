#!/bin/bash
set -e
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer InceptionV3/Predictions/Reshape_1 --model_name inception_v3 --opset 7 8 9 10 11
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer InceptionV4/Logits/Predictions --model_name inception_v4 --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer InceptionV2/Predictions/Reshape_1 --model_name inception_v2 --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer InceptionV1/Logits/Predictions/Reshape_1 --model_name inception_v1 --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer resnet_v1_50/predictions/Reshape_1 --model_name resnet_v1_50 --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer resnet_v1_101/predictions/Reshape_1 --model_name resnet_v1_101 --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer resnet_v1_152/predictions/Reshape_1 --model_name resnet_v1_152 --opset 7 8 9 10 11

python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer resnet_v2_50/predictions/Reshape_1 --model_name resnet_v2_50 --opset 7 8 9 10 11
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer resnet_v2_101/predictions/Reshape_1 --model_name resnet_v2_101 --opset 7 8 9 10 11
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer resnet_v2_152/predictions/Reshape_1 --model_name resnet_v2_152 --opset 7 8 9 10 11

python3 gen_output.py --input_height 331 --input_width 331 --input_layer input --output_layer final_layer/predictions --model_name nasnet_large --opset 7 8 9 10 11
python3 gen_output.py --input_height 331 --input_width 331 --input_layer input --output_layer final_layer/predictions --model_name pnasnet_large --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer final_layer/predictions --model_name nasnet_mobile --opset 7 8 9 10 11

python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer InceptionResnetV2/Logits/Predictions --model_name inception_resnet_v2 --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer MobilenetV1/Predictions/Reshape_1 --model_name mobilenet_v1_1.0_224  --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer MobilenetV2/Predictions/Reshape_1 --model_name mobilenet_v2_1.4_224 --opset 7 8 9 10 11
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer MobilenetV2/Predictions/Reshape_1 --model_name mobilenet_v2_1.0_224 --opset 7 8 9 10 11
