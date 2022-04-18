from json import decoder
import sys
sys.path.insert(0, 'dependencies/fairseq')
sys.path.insert(0, 'dependencies/Transformers/src')

import transformers
import fairseq
import time
import os
import torch
import numpy as np
from ZCodeBartBased import ZCodeBartBasedForConditionalGeneration, ZCodeBartBasedConfig, ZCodeBartBasedTokenizer
from typing import Dict, Any

from transformers.file_utils import ModelOutput
import onnxruntime
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel

torch.set_printoptions(precision=6, edgeitems=10, linewidth=300)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir')
    model_dir = '/home/wy/Zcode/babel_share/models/abs_sum/default'

    config = ZCodeBartBasedConfig.from_pretrained(model_dir)
    tokenizer = ZCodeBartBasedTokenizer(
        os.path.join(model_dir, 'sentencepiece.bpe.model'),
        os.path.join(model_dir, 'dict.src.txt'),
        os.path.join(model_dir, 'dict.tgt.txt'),
        config=config)

    # summarization
    config.extra_config["fs_args"]["max_len_b"] = 512
    config.extra_config["fs_args"]["fp16"] = False

    device_name = "cpu"
    #device_name = "cuda"
    config.use_decoder = True

    with torch.no_grad():
        model = ZCodeBartBasedForConditionalGeneration.from_pretrained(model_dir, config=config).eval()
        # model.zcode.encoders['seq2seq'].eval()
        # model.zcode.decoders['seq2seq'].eval()
        # print('decoder mode: ', model.zcode.decoders['seq2seq'].training)
        # import pdb; pdb.set_trace()
        model = model.to(device_name)

        input_text = "A drunk teenage boy had to be rescued by security after jumping into a lions' enclosure at a zoo in western India. Rahul Kumar, 17, clambered over the enclosure fence at the Kamla Nehru Zoological Park in Ahmedabad, and began running towards the animals, shouting he would 'kill them'. Mr Kumar explained afterwards that he was drunk and 'thought I'd stand a good chance' against the predators. Next level drunk: Intoxicated Rahul Kumar, 17, climbed into the lions' enclosure at a zoo in Ahmedabad and began running towards the animals shouting 'Today I kill a lion!' Mr Kumar had been sitting near the enclosure when he suddenly made a dash for the lions, surprising zoo security. The intoxicated teenager ran towards the lions, shouting: 'Today I kill a lion or a lion kills me!' A zoo spokesman said: 'Guards had earlier spotted him close to the enclosure but had no idea he was planing to enter it. 'Fortunately, there are eight moats to cross before getting to where the lions usually are and he fell into the second one, allowing guards to catch up with him and take him out. 'We then handed him over to the police.' Brave fool: Fortunately, Mr Kumar fell into a moat as he ran towards the lions and could be rescued by zoo security staff before reaching the animals (stock image) Kumar later explained: 'I don't really know why I did it. 'I was drunk and thought I'd stand a good chance.' A police spokesman said: 'He has been cautioned and will be sent for psychiatric evaluation. 'Fortunately for him, the lions were asleep and the zoo guards acted quickly enough to prevent a tragedy similar to that in Delhi.' Last year a 20-year-old man was mauled to death by a tiger in the Indian capital after climbing into its enclosure at the city zoo."
        beam = 5

        lang = '__en__'
        features = [tokenizer.convert_tokens_to_ids(lang)]
        features.extend(tokenizer.encode_plus(input_text, add_special_tokens=False, max_length=510, truncation=True)["input_ids"])
        features.append(tokenizer.eos_token_id)
        input_data = torch.LongTensor(features).unsqueeze(0).to(device_name)
        print(tokenizer.pad_token_id)
        print("pytorch inference")
        start_time = time.time()
        pred_ids = model.generate(
            input_data,
            decoder_start_token_id=tokenizer.eos_token_id,
            num_beams=beam,
            num_return_sequences=beam,
            min_length=10,
            max_length=256,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3)
        time_cost = time.time() - start_time
        print("--- %s seconds ---" % (time_cost))
        print(pred_ids[0])
        print(tokenizer.decode(pred_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))

        print("ORT inference")
        ort_inputs = {
            "input_ids": np.int32(input_data.cpu().numpy()),
            "max_length": np.array([256], dtype=np.int32),
            "min_length": np.array([10], dtype=np.int32),
            "num_beams": np.array([5], dtype=np.int32),
            "num_return_sequences": np.array([5], dtype=np.int32),
            "temperature": np.array([1], dtype=np.float32),
            "length_penalty": np.array([1], dtype=np.float32),
            "repetition_penalty": np.array([1], dtype=np.float32)
        }
        model_path ='../export/zcode_beamsearch/beam_search_zcode.onnx'
        sess_options = SessionOptions()
        sess_options.log_severity_level = 4
        sess = InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        start_time = time.time()
        out = sess.run(None, ort_inputs)
        time_cost = time.time() - start_time
        print("--- %s seconds ---" % (time_cost))
        print(out[0][0][0])
        print(tokenizer.decode(torch.from_numpy(out[0][0][0]), skip_special_tokens=True, clean_up_tokenization_spaces=False))

if __name__ == "__main__":
    main()
