# BERT Model Optimization Tool Overview

This tool showcases how to fuse a BERT ONNX model either exported from PyTorch or converted from TensorFlow, and generates an optimized model to run faster with OnnxRuntime.

Note that OnnxRuntime can fuse the Bert ONNX model exported from PyTorch automatically. You don't need this tool to fuse the model. It is only required for Bert Model converted from Tensorflow. 

## Export a BERT model from PyTorch
For example, after using https://github.com/huggingface/transformers/tree/master/examples/run_glue.py to train a BERT model in PyTorch 1.3, you can use the following function to export ONNX model. 

Please specify do_constant_folding=True. That's required for this tool.

```python
def export_onnx(args, model, output_path):
    model.eval() # set the model to inference mode
    device = torch.device("cpu")
    model.to(device)
    dummy_input0 = torch.LongTensor(args.eval_batch_size, args.max_seq_length).fill_(1).to(device)
    dummy_input1 = torch.LongTensor(args.eval_batch_size, args.max_seq_length).fill_(1).to(device)
    dummy_input2 = torch.LongTensor(args.eval_batch_size, args.max_seq_length).fill_(0).to(device)
    dummy_input = (dummy_input0, dummy_input1, dummy_input2)
    torch.onnx.export(model,                     # model being run
                      dummy_input,               # model input (or a tuple for multiple inputs)
                      output_path,               # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ["input_ids", "input_mask", "segment_ids"],
                      output_names = ["output"],
                      dynamic_axes={'input_ids' : {0 : 'batch_size'},    # variable length axes
                                    'input_mask' : {0 : 'batch_size'},
                                    'segment_ids' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
```

## Convert a BERT model from Tensorflow

The tf2onnx and keras2onnx tools can be used to convert model that trained by Tensorflow.

For Keras2onnx, please refere to its [example script](https://github.com/onnx/keras-onnx/blob/master/applications/nightly_build/test_transformers.py).

For tf2onnx, please refer to this notebook: https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb


## Model Optimization

Example of using the script optimizer.py to convert a BERT-large model to run in V100 GPU:
```console
python optimizer.py --input original_model.onnx --output optimized_model_gpu.onnx --num_heads 16 --hidden_size 1024 --input_int32 --float16
```

### Options

See below for description of some options of optimizer.py:

- **input**: input model path
- **output**: output model path
- **model_type**: (*defaul: bert*)
    There are 4 model types: *bert* (exported by PyTorch), *bert_tf* (BERT exported by tf2onnx), *bert_keras* (BERT exported by keras2onnx) and *gpt2* (exported by PyTorch) respectively.
- **num_heads**: (*default: 12*)
    Number of attention heads. BERT-base and BERT-large has 12 and 16 respectively.
- **hidden_size**: (*default: 768*)
    BERT-base and BERT-large has 768 and 1024 hidden nodes respectively.
- **input_int32**: (*optional*)
    Exported model ususally uses int64 tensor as input. If this flag is specified, int32 tensors will be used as input, and it could avoid un-necessary Cast nodes and get better performance.
- **float16**: (*optional*)
    By default, model uses float32 in computation. If this flag is specified, half-precision float will be used. This option is recommended for NVidia GPU with Tensor Core like V100 and T4. For older GPUs, float32 is likely faster.
- **verbose**: (*optional*)
    Print verbose information when this flag is specified.

### Supported Models

Right now, this tool assumes input model has 3 inputs for input IDs, segment IDs, and attention mask. A model with less or addtional inputs might not be optimized.

Most optimizations require exact match of a subgraph. That means this tool could only support similar models with such subgraphs. Any layout change in subgraph might cause optimization not working. Note that different training or export tool (including different versions) might get different graph layouts.

Here is list of models from [Huggingface Transformers](https://github.com/huggingface/transformers/) that have been tested using this tool:
- **BertForSequenceClassification** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py) exported by PyTorch 1.2-1.4 using opset version 10 or 11.
- **BertForQuestionAnswering** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py) exported by PyTorch 1.2-1.4 using opset version 10 or 11.
- **TFBertForSequenceClassification** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_tf_glue.py) exported by keras2onnx installed from its master source.
- **TFBertForQuestionAnswering** exported by keras2onnx installed from its master source.
- **GPT2Model** exported by PyTorch 1.4 using opset version 10 or 11.
- **GPT2LMHeadModel** exported by PyTorch 1.4 using opset version 10 or 11.
If your model is not in the list, the optimized model might not work. You are welcome to update the scripts to support new models.

## Model Verification

When a BERT model is optimized, some optimization uses approximation in calculation so the output might be slightly different. It is recommended to use your evaluation set to measure the precision and recall. We expect the accuracy shall be on par after optimization.

If your BERT model has three inputs, a script compare_bert_results.py can be used to do a quick verification. The tool will generate some fake input data, and compare results from both the original and optimized models. If outputs are all close, it is safe to use the optimized model.

Example of verifying models optimized for CPU and GPU:

```console
pip install onnxruntime
python compare_bert_results.py --baseline_model original_model.onnx --optimized_model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128 --samples 100

pip uninstall onnxruntime
pip install onnxruntime-gpu
python compare_bert_results.py --baseline_model original_model.onnx --optimized_model optimized_model_gpu.onnx --batch_size 1 --sequence_length 128 --samples 100 --use_gpu
```

To use onnxruntime-gpu, it is required to install CUDA and cuDNN and add their bin directories to PATH environment variable.

## Performance Test (Python)

bert_perf_test.py can be used to check the model inference performance of python API. Below are examples:

```console
pip install onnxruntime
python bert_perf_test.py --model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128 --samples 100 --test_times 10 --inclusive

pip uninstall onnxruntime
pip install onnxruntime-gpu
python bert_perf_test.py --model optimized_model_gpu.onnx --batch_size 1 --sequence_length 128 --samples 100 --test_times 10 --use_gpu --inclusive
```

After test is finished, a file like perf_results_CPU_B1_S128_<date_time>.txt or perf_results_GPU_B1_S128_<date_time>.txt will be output to the model directory.

## Performance Test (C API)

First, we need generate some test data. Please make sure there is no sub-directories on the directory of onnx model.

Here is an example:
```console
python bert_test_data.py --model bert.onnx --batch_size 1 --sequence_length 32 --samples 100 --output_dir .
```

You can go to root of this git repository, and build onnxruntime_perf_test.exe from source to test performance of C API. Example commands in Windows:
```console
build.bat --config RelWithDebInfo --enable_lto --use_openmp --build_shared_lib --parallel --cmake_generator "Visual Studio 16 2019"
Set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
Set OMP_WAIT_POLICY=PASSIVE
build\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime_perf_test.exe -e cpu -r 100 -s -o 2 bert.onnx output.txt
```
