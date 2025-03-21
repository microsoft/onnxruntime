# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

hidden_size = 8

vocab_size = 1024
save_directory = "tiny_t5"
model_name = "google-t5/t5-small"

config = T5Config.from_pretrained(model_name)

config.num_heads = 2

if vocab_size:
    config.vocab_size = 1024

config.d_model = hidden_size
config.d_kv = hidden_size // config.num_heads
config.d_ff = hidden_size * 2
config.num_layers = 2
config.num_decoder_layers = config.num_layers

model = T5ForConditionalGeneration(config)

model.save_pretrained(save_directory)

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
tokenizer.save_pretrained(save_directory)


def update_tokenizer(sp_model_path: str, vocab_size: int):
    sp = SentencePieceProcessor()
    sp.Load(sp_model_path)

    # Export the vocabulary
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for id in range(sp.GetPieceSize()):
            piece = sp.IdToPiece(id)
            score = sp.GetScore(id)
            f.write(f"{piece}\t{score}\n")

    with open("vocab.txt", encoding="utf-8") as f:
        vocab = [line.strip().split("\t") for line in f]

    # Sort by score in descending order and select top tokens
    vocab_sorted = sorted(vocab, key=lambda x: float(x[1]), reverse=True)
    pruned_vocab = vocab_sorted[:vocab_size]

    # Write the pruned vocabulary to a new file
    with open("pruned_vocab.txt", "w", encoding="utf-8") as f:
        for piece, score in pruned_vocab:
            f.write(f"{piece}\t{score}\n")

    # Train a new SentencePiece model using the pruned vocabulary as a seed.
    # Example corpus.txt can be found by searching "corpus.txt download" in search engine.
    SentencePieceTrainer.Train(
        f"--input=corpus.txt --model_prefix=spiece --vocab_size={vocab_size} --user_defined_symbols=pruned_vocab.txt"
    )

    # Load the new model
    sp_new = SentencePieceProcessor()
    sp_new.Load("spiece.model")

    # Test encoding and decoding
    text = "This is an example sentence."
    tokens = sp_new.EncodeAsPieces(text)
    print(tokens)

    detokenized_text = sp_new.DecodePieces(tokens)
    print(detokenized_text)

    # Replace the original model.
    os.replace("spiece.model", sp_model_path)


if vocab_size:
    original_path = os.path.join(save_directory, "spiece.model")
    update_tokenizer(original_path, vocab_size)
