import torch


def my_loss(x, target):
    x = x.view(-1, 28785)
    return torch.nn.CrossEntropyLoss()(x, target)


def transformer_model_description():
    bptt = 35
    ntokens = 28785
    batch_size = 20

    model_desc = {'inputs':  [('input1', [bptt, batch_size]),
                              ('label', [bptt, batch_size, ntokens],)],
                  'outputs': [('loss', [], True)]}
    return model_desc
