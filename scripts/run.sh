#!/bin/bash
set -e
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer InceptionV3/Predictions/Reshape_1 --model_name inception_v3 --opset 15
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer InceptionV4/Logits/Predictions --model_name inception_v4 --opset 15
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer InceptionV2/Predictions/Reshape_1 --model_name inception_v2 --opset 15
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer InceptionV1/Logits/Predictions/Reshape_1 --model_name inception_v1 --opset 15
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer resnet_v1_50/predictions/Reshape_1 --model_name resnet_v1_50 --opset 15
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer resnet_v1_101/predictions/Reshape_1 --model_name resnet_v1_101 --opset 15
python3 gen_output.py --input_height 224 --input_width 224 --input_layer input --output_layer resnet_v1_152/predictions/Reshape_1 --model_name resnet_v1_152 --opset 15

python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer resnet_v2_50/predictions/Reshape_1 --model_name resnet_v2_50 --opset 15
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer resnet_v2_101/predictions/Reshape_1 --model_name resnet_v2_101 --opset 15
python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer resnet_v2_152/predictions/Reshape_1 --model_name resnet_v2_152 --opset 15

python3 gen_output.py --input_height 299 --input_width 299 --input_layer input --output_layer InceptionResnetV2/Logits/Predictions --model_name inception_resnet_v2 --opset 15
