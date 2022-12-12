# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This measures the performance of OnnxRuntime, PyTorch and TorchScript on transformer models.
# Please install PyTorch (see https://pytorch.org/) before running this benchmark. Like the following:
# GPU:   conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
# CPU:   conda install pytorch torchvision cpuonly -c pytorch

# Batch Sizes and Sequence Lengths
batch_sizes="8"
input_length="10"
output_length="50"

#batch_sizes="1 4"
#input_length="10"
#output_length="10"


#pip uninstall --yes onnxruntime-gpu
#pip install onnxruntime-gpu
#
#for b in $batch_sizes
#do
#  for i in $input_length
#  do
#    for o in $output_length
#    do
#      echo "*************************************************"
#      echo Run Benchmark: ${b} ${i} ${o}
#      total_output=$((i + o))
#      echo python convert_generation.py -m gpt2 --output gpt2_beam_search.bm4_onnx -p fp16 --use_gpu --total_runs 100 --min_length ${total_output} --max_length ${total_output} --batch_size ${b} --input_length ${i} --disable_parity
#      python convert_generation.py -m gpt2 --output gpt2_beam_search.bm4_onnx -p fp16 --use_gpu --total_runs 100 --min_length ${total_output} --max_length ${total_output} --batch_size ${b} --input_length ${i} --disable_parity
#    done
#  done
#done

#pip uninstall --yes onnxruntime_gpu
#pip install ./onnxruntime_gpu-1.14.0-cp38-cp38-linux_x86_64.whl

#for b in $batch_sizes
#do
#  for i in $input_length
#  do
#    for o in $output_length
#    do
#      echo "*************************************************"
#      echo Run Benchmark: ${b} ${i} ${o}
#      total_output=$((i + o))
#      echo python convert_generation.py -m gpt2 --output gpt2_beam_search.bm4_onnx -p fp16 --use_gpu --total_runs 100 --min_length ${total_output} --max_length ${total_output} --batch_size ${b} --input_length ${i} --disable_parity --pad_vocab_size --enable_last_token_logits_matmul 
#      python convert_generation.py -m gpt2 --output gpt2_beam_search.bm4_onnx -p fp16 --use_gpu --total_runs 100 --min_length ${total_output} --max_length ${total_output} --batch_size ${b} --input_length ${i} --disable_parity --pad_vocab_size --enable_last_token_logits_matmul 
#    done
#  done
#done

python convert_generation.py -m gpt2 --output gpt2_beam_search.bm4_onnx -p fp16 --use_gpu --total_runs 100 --min_length 60 --max_length 60 --batch_size 8 --input_length 10 --disable_parity --pad_vocab_size --enable_last_token_logits_matmul --num_beams 1 --num_return_sequences 1