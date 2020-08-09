import torch
from onnxruntime.capi.ort_trainer import IODescription, ModelDescription


def my_loss(x, target):
    x = x.view(-1, 28785)
    return torch.nn.CrossEntropyLoss()(x, target)


def transformer_model_description(bptt=35, batch_size=20, ntokens=28785):
    model_desc = {'inputs':  [('input1', [bptt, batch_size]),
                              ('label', [bptt, batch_size, ntokens],)],
                  'outputs': [('loss', [], True),
                              ('predictions', [bptt, batch_size, ntokens])]}
    return model_desc


def legacy_transformer_model_description(bptt=35, batch_size=20, ntokens=28785):
    input_desc = IODescription('input1', [bptt, batch_size], torch.float32)
    label_desc = IODescription('label', [bptt, batch_size, ntokens], torch.int64)
    loss_desc = IODescription('loss', [], torch.float32)
    predictions_desc = IODescription('predictions', [bptt, batch_size, ntokens], torch.float32)
    return ModelDescription([input_desc, label_desc],[loss_desc, predictions_desc]),\
           IODescription('__learning_rate', [1], torch.float32)
