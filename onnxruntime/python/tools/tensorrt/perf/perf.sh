#!/bin/bash

SYMBOLIC_SHAPE_INFER="/home/hcsuser/repos/onnxruntime/onnxruntime/python/tools/symbolic_shape_infer.py"

# many models 
if [ "$1" == "many-models" ]
then
    FAILING_MODEL_LIST="/home/hcsuser/perf/failing_model_list.json"
    python3 benchmark.py -r validate -m /home/hcsuser/mount/ -s $SYMBOLIC_SHAPE_INFER -o result/"$1" -e $FAILING_MODEL_LIST
    python3 benchmark.py -r benchmark -i random -t 10 -m /home/hcsuser/mount -s $SYMBOLIC_SHAPE_INFER -o result/"$1" -e $FAILING_MODEL_LIST
fi

# ONNX model zoo
if [ "$1" == "onnx-zoo-models" ]
then
    MODEL_LIST="model_list.json"
    #python3 benchmark.py -r validate -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
    python3 benchmark_wrapper.py -r validate -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
    #python3 benchmark.py -r benchmark -i random -t 10 -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
    python3 benchmark_wrapper.py -r benchmark -t 10 -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
fi

# 1P models 
if [ "$1" == "partner-models" ]
then
    MODEL_LIST="/home/hcsuser/perf/partner_model_list.json"
    python3 benchmark.py -r validate -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
    python3 benchmark.py -r benchmark -i random -t 10 -m $MODEL_LIST -s $SYMBOLIC_SHAPE_INFER -o result/"$1"
fi

