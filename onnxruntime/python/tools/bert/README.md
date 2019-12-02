# BERT Model Optimization Tool Overview

This tool converts a BERT ONNX model exported from PyTorch, and generates an optimized model to run faster in NVidia GPU.

Currently, this script **cannot** process BERT models exported from Tensorflow since the graph has some difference.

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
## Model Optimization

Example of using the script bert_model_optimization.py to convert a BERT-large model to run in V100 GPU:
```console
python bert_model_optimization.py --input input_model.onnx --output optimized_model.onnx --num_heads 24 --hidden_size 1024 --sequence_length 128 --input_int32 --float16
```

## Options

See below for description of all the options:

- **input**: input model path
- **output**: output model path
- **num_heads**: (*default: 12*)
    Number of attention heads, like 24 for BERT-large model.
- **hidden_size**: (*default: 768*)
- **sequence_length**: (*default: 128*)
    Maximum sequence length.
- **input_int32**: (*optional*)
    Exported model ususally uses int64 tensor as input. If this flag is specified, int32 tensors will be used as input, and it could avoid un-necessary Cast nodes and get better performance.
- **float16**: (*optional*)
    By default, model uses float32 in computation. If this flag is specified, half-precision float will be used. This option is recommended for NVidia GPU with Tensor Core like V100 and T4. For older GPUs, float32 is likely faster.