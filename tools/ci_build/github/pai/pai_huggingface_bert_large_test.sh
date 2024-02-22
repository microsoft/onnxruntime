#!/bin/bash

set -ex

usage() { echo "Usage: $0 [-v <ROCm version>]" 1>&2; exit 1; }

while getopts "v:" parameter_Option
do case "${parameter_Option}"
in
v) ROCM_VERSION=${OPTARG};;
*) usage ;;
esac
done

MI200_DEVICE_NUMBERS=$(rocm-smi --showproductname | grep -c "MI250" | xargs)

if [ "$MI200_DEVICE_NUMBERS" -gt "0" ]; then
  RESULT_FILE=ci-mi200.huggingface.bert-large-rocm${ROCM_VERSION}.json
else
  RESULT_FILE=ci-mi100.huggingface.bert-large-rocm${ROCM_VERSION}.json
fi

python \
  /stage/huggingface-transformers/examples/pytorch/language-modeling/run_mlm.py \
  --model_name_or_path bert-large-uncased \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --do_train \
  --max_steps 260 \
  --logging_steps 20 \
  --output_dir ./test-mlm-bbu \
  --overwrite_output_dir \
  --per_device_train_batch_size 8 \
  --fp16 \
  --dataloader_num_workers 1 \
  --ort \
  --skip_memory_metrics

cat ci-pipeline-actual.json

python /onnxruntime_src/orttraining/tools/ci_test/compare_huggingface.py \
  ci-pipeline-actual.json \
  /onnxruntime_src/orttraining/tools/ci_test/results/"$RESULT_FILE"
