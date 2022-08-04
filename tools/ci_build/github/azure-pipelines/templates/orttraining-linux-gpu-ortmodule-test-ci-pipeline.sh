#!/bin/bash

REQDIR="requirements_torch1.11.0_cu11.3";

if [ $1 == "Manual" ]; then
    REQDIR="requirements_torch_nightly"
fi

python3 -m pip uninstall -y -r /onnxruntime_src/tools/ci_build/github/linux/docker/scripts/training/requirements.txt
python3 -m pip install -r /onnxruntime_src/tools/ci_build/github/linux/docker/scripts/training/ortmodule/stage1/$REQDIR/requirements.txt
python3 -m pip install -r /onnxruntime_src/tools/ci_build/github/linux/docker/scripts/training/ortmodule/stage2/requirements.txt
rm -rf /build/onnxruntime/ && python3 -m pip install /build/dist/onnxruntime*.whl && python3 -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
/build/launch_test.py --cmd_line_with_args 'python orttraining_ortmodule_tests.py --mnist /mnist --bert_data /bert_data/hf_data/glue_data/CoLA/original/raw --transformers_cache /hf_models_cache/huggingface/transformers' --cwd /build;
