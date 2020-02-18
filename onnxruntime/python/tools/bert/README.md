# BERT Model Optimization Tool Overview

This tool showcases how to fuse a BERT ONNX model either exported from PyTorch or converted from TensorFlow, and generates an optimized model to run faster with OnnxRuntime.

Note that OnnxRuntime can fuse the Bert ONNX model exported from PyTorch automatically. You don't need this tool to fuse the model. It is only required for Bert Model converted from Tensorflow. 

## Export an BERT model from PyTorch
For example, after using https://github.com/huggingface/transformers to train a BERT model in PyTorch 1.3, you can use the following function to export ONNX model. 

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
                      dynamic_axes={'input_ids' : {0 : 'batch_size'},    # variable lenght axes
                                    'input_mask' : {0 : 'batch_size'},
                                    'segment_ids' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
```

## Convert an BERT model from Tensorflow

The tf2onnx and keras2onnx tools can be used to convert model that trained by Tensorflow.

For Keras2onnx, please refere to its [example script](https://github.com/onnx/keras-onnx/blob/master/applications/nightly_build/test_transformers.py).

For tf2onnx, please refer to this notebook: https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb


## Model Optimization

Example of using the script bert_model_optimization.py to convert a BERT-large model to run in V100 GPU:
```console
python bert_model_optimization.py --input input_model.onnx --output optimized_model.onnx --num_heads 24 --hidden_size 1024 --sequence_length 128 --input_int32 --float16
```

## Options

See below for description of all the options:

- **input**: input model path
- **output**: output model path
- **model_type**: (*defaul: bert*)
    There are 3 model types: *bert*, *bert_tf* and *bert_keras* for models exported by PyTorch, tf2onnx and keras2onnx respectively.
- **num_heads**: (*default: 12*)
    Number of attention heads, like 24 for BERT-large model.
- **hidden_size**: (*default: 768*)
- **sequence_length**: (*default: 128*)
    Maximum sequence length.
- **input_int32**: (*optional*)
    Exported model ususally uses int64 tensor as input. If this flag is specified, int32 tensors will be used as input, and it could avoid un-necessary Cast nodes and get better performance.
- **gpu_only**: (*optional*)
    Specify the option if running on GPU only.
- **float16**: (*optional*)
    By default, model uses float32 in computation. If this flag is specified, half-precision float will be used. This option is recommended for NVidia GPU with Tensor Core like V100 and T4. For older GPUs, float32 is likely faster.
- **verbose**: (*optional*)
    Print verbose information when this flag is specified.

## Supported Models

Right now, this tool assumes input model has 3 inputs for input IDs, segment IDs, and attention mask. A model with less or addtional inputs might not be optimized.

Most optimizations require exact match of a subgraph. That means this tool could only support similar models with such subgraphs. Any layout change in subgraph might cause optimization not working. Note that different training or export tool (including different versions) might get different graph layouts.

Here is list of models that have been tested using this tool:
- **BertForSequenceClassification** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py) exported by PyTorch 1.2-1.4 using opset version 10 or 11.
- **BertForQuestionAnswering** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py) exported by PyTorch 1.2-1.4 using opset version 10 or 11.
- **TFBertForSequenceClassification** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_tf_glue.py) exported by keras2onnx 1.6.0.

If your model is not in the list, the optimized model might not work. You are welcome to update the scripts to support new models.
