import torch
from onnxruntime.capi.ort_trainer import IODescription, ModelDescription


def my_loss(x, target):
    x = x.view(-1, 28785)
    return torch.nn.CrossEntropyLoss()(x, target)


def transformer_model_description():
    bptt = 35
    ntokens = 28785
    batch_size = 20

    model_desc = {'inputs':  [('input1', [bptt, batch_size]),
                              ('label', [bptt, batch_size, ntokens],)],
                  'outputs': [('loss', [], True),
                              ('predictions', [bptt, batch_size, ntokens])]}
    return model_desc


def legacy_transformer_model_description():
    input_desc = IODescription('input1', [35, 20], torch.float32)
    label_desc = IODescription('label', [35, 20, 28785], torch.int64)
    loss_desc = IODescription('loss', [], torch.float32)
    lr = 0.001
    #return ModelDescription([input_desc, label_desc], [loss_desc]), IODescription('Learning_Rate', [lr,], torch.float32)
    prediction_desc = IODescription('prediction', [35, 20, 28785], torch.float32)
    return ModelDescription([input_desc, label_desc], [loss_desc, prediction_desc]), IODescription('Learning_Rate', [lr,], torch.float32)
