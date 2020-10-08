import torch
import torchtext
from torchtext.data.utils import get_tokenizer

def batchify(data, bsz, TEXT, device):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt=35):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def prepare_data(device='cpu', train_batch_size=20, eval_batch_size=20):
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    device = torch.device(device)

    train_data = batchify(train_txt, train_batch_size, TEXT, device)
    val_data = batchify(val_txt, eval_batch_size, TEXT, device)
    test_data = batchify(test_txt, eval_batch_size, TEXT, device)

    return train_data, val_data, test_data
