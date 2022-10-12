# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import time

import numpy as np
import torch
from utils import export_helper

from onnxruntime import InferenceSession, SessionOptions


def run_inference(args):
    """Run inference with BART Pytorch and ONNX models.

    Run an easy example with two models for a simple check on performance.

    Args:
        args: User input.
    """

    beam = args.num_beams
    min_length = args.min_length
    max_length = args.max_length
    repetition_penalty = args.repetition_penalty
    no_repeat_ngram_size = args.no_repeat_ngram_size

    config, tokenizer = export_helper.initialize_config(args)

    with torch.no_grad():

        model, input_data = export_helper.initialize_model(config, tokenizer, args)
        batch_num = 3
        input_data = input_data.repeat(batch_num, 1)

        print("pytorch inference ...")
        start_time = time.time()
        pred_ids = model.generate(
            input_data,
            decoder_start_token_id=tokenizer.eos_token_id,
            num_beams=beam,
            num_return_sequences=beam,
            min_length=min_length,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        time_cost = time.time() - start_time
        print("--- %s seconds ---" % (time_cost))
        for j in range(batch_num):
            for i in range(beam):
                print(
                    "batch",
                    j,
                    ": sequence:",
                    i,
                    tokenizer.decode(
                        pred_ids[j * beam + i], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ),
                )

        print("ORT inference ...")
        ort_inputs = {
            "input_ids": np.int32(input_data.cpu().numpy()),
            "max_length": np.array([max_length], dtype=np.int32),
            "min_length": np.array([min_length], dtype=np.int32),
            "num_beams": np.array([beam], dtype=np.int32),
            "num_return_sequences": np.array([beam], dtype=np.int32),
            "length_penalty": np.array([1], dtype=np.float32),
            "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
            "attention_mask": np.ones(input_data.shape).astype(np.int32),  # custom attn_mask, please change as needed
        }

        model_path = os.path.join(args.output, "model_final.onnx")
        sess_options = SessionOptions()
        sess_options.log_severity_level = 4
        sess = InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        start_time = time.time()
        out = sess.run(None, ort_inputs)
        time_cost = time.time() - start_time
        print("--- %s seconds ---" % (time_cost))
        for j in range(batch_num):
            for i in range(beam):
                print(
                    "batch",
                    j,
                    ": sequence:",
                    i,
                    tokenizer.decode(
                        torch.from_numpy(out[0][j][i]), skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ),
                )
