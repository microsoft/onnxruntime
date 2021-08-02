import io
import os
import torch
import torchtext
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def batchify(data, bsz, device):
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


def prepare_data(device='cpu', train_batch_size=20, eval_batch_size=20, data_dir=None):
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'

    download_path = '.data_wikitext_2_v1'
    extract_path = None
    if data_dir:
        download_path = os.path.join(data_dir, 'download')
        os.makedirs(download_path, exist_ok=True)
        download_path = os.path.join(download_path, 'wikitext-2-v1.zip')

        extract_path = os.path.join(data_dir, 'extracted')
        os.makedirs(extract_path, exist_ok=True)

    test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url, root=download_path), to_path=extract_path)
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer,
                                        iter(io.open(train_filepath,
                                                    encoding="utf8"))))

    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                        dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(iter(io.open(train_filepath, encoding="utf8")))
    val_data = data_process(iter(io.open(valid_filepath, encoding="utf8")))
    test_data = data_process(iter(io.open(test_filepath, encoding="utf8")))

    device = torch.device(device)

    train_data = batchify(train_data, train_batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    return train_data, val_data, test_data
