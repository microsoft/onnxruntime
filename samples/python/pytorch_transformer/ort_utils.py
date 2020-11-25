import torch

from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription


def my_loss(x, target):
    x = x.view(-1, 28785)
    return torch.nn.CrossEntropyLoss()(x, target)


def transformer_model_description(bptt=35, batch_size=20, ntokens=28785):
    model_desc = {'inputs':  [('input1', [bptt, batch_size]),
                              ('label', [bptt * batch_size])],
                  'outputs': [('loss', [], True),
                              ('predictions', [bptt, batch_size, ntokens])]}
    return model_desc


def transformer_model_description_dynamic_axes(ntokens=28785):
    model_desc = {'inputs':  [('input1', ['bptt', 'batch_size']),
                              ('label', ['bptt_x_batch_size'])],
                  'outputs': [('loss', [], True),
                              ('predictions', ['bptt', 'batch_size', ntokens])]}
    return model_desc


def legacy_transformer_model_description(bptt=35, batch_size=20, ntokens=28785):
    input_desc = Legacy_IODescription('input1', [bptt, batch_size])
    label_desc = Legacy_IODescription('label', [bptt * batch_size])
    loss_desc = Legacy_IODescription('loss', [])
    predictions_desc = Legacy_IODescription('predictions', [bptt, batch_size, ntokens])
    return Legacy_ModelDescription([input_desc, label_desc],[loss_desc, predictions_desc]),\
           Legacy_IODescription('__learning_rate', [1])


def legacy_transformer_model_description_dynamic_axes(ntokens=28785):
    input_desc = Legacy_IODescription('input1', ['bptt', 'batch_size'])
    label_desc = Legacy_IODescription('label', ['bptt_x_batch_size'])
    loss_desc = Legacy_IODescription('loss', [])
    predictions_desc = Legacy_IODescription('predictions', ['bptt', 'batch_size', ntokens])
    return Legacy_ModelDescription([input_desc, label_desc],[loss_desc, predictions_desc]),\
           Legacy_IODescription('__learning_rate', [1])
