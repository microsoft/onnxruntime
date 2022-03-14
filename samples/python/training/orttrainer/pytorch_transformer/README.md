# TransformerModel example

This example was adapted from Pytorch's [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) tutorial

## Requirements

* PyTorch 1.6+
* TorchText 0.6+
* ONNX Runtime 1.5+

## Running PyTorch version

```bash
python pt_train.py
```

## Running ONNX Runtime version

```bash
python ort_train.py
```

## Optional arguments

| Argument          | Description                                             | Default   |
| :---------------- | :-----------------------------------------------------: | --------: |
| --batch-size      | input batch size for training                           | 20        |
| --test-batch-size | input batch size for testing                            | 20        |
| --epochs          | number of epochs to train                               | 2         |
| --lr              | learning rate                                           | 0.001     |
| --no-cuda         | disables CUDA training                                  | False     |
| --seed            | random seed                                             | 1         |
| --log-interval    | how many batches to wait before logging training status | 200       |
