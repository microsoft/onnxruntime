#!/bin/bash

source $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/_basics/_env_variables.sh
cp $PHILLY_SCRIPT_ROOT"bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm.onnx" /code/bert.onnx 
bash $PHILLY_DATA_DIRECTORY/$PHILLY_VC/pengwa/profile/scripts-ort/_basics/_machine_info.sh

sleep 2d
