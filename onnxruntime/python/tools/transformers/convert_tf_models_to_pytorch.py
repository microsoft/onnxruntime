#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import os
import glob

TFMODELS = {
    "bert_base_uncased": ("bert", "BertConfig", "", "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"),
    "bert_base_cased": ("bert", "BertConfig", "", "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip"),
    "bert_large_uncased": ("bert", "BertConfig", "", "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"),
    "albert_base": ("albert", "AlbertConfig", "", "https://storage.googleapis.com/albert_models/albert_base_v1.tar.gz"),
    "albert_large": ("albert", "AlbertConfig", "", "https://storage.googleapis.com/albert_models/albert_large_v1.tar.gz"),
    "gpt-2-117M": ("gpt2", "GPT2Config", "GPT2Model", "https://storage.googleapis.com/gpt-2/models/117M"),
    "gpt-2-124M": ("gpt2", "GPT2Config", "GPT2Model", "https://storage.googleapis.com/gpt-2/models/124M")
}

def download_tf_checkpoint(model_name, tf_models_dir="tf_models"):
    import pathlib
    base_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), tf_models_dir)
    ckpt_dir = os.path.join(base_dir, model_name)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    tf_ckpt_url = TFMODELS[model_name][3]

    import re
    import requests
    if (re.search('.zip$', tf_ckpt_url) != None):
        r = requests.get(tf_ckpt_url)
        zip_name = tf_ckpt_url.split("/")[-1]
        zip_dir = os.path.join(ckpt_dir, zip_name)
        with open(zip_dir, 'wb') as f:
            f.write(r.content)

        # unzip file
        import zipfile
        with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
            zip_ref.extractall(ckpt_dir)
            os.remove(zip_dir)

        # get prefix
        for o in os.listdir(ckpt_dir):
            folder_dir = os.path.join(ckpt_dir, o)
            break;
        unique_file_name = str(glob.glob(folder_dir + "/*data-00000-of-00001"))
        prefix = (unique_file_name.rpartition('.')[0]).split("/")[-1]

        return os.path.join(folder_dir, prefix)

    elif (re.search('.tar.gz$', tf_ckpt_url) != None):
        r = requests.get(tf_ckpt_url)
        tar_name = tf_ckpt_url.split("/")[-1]
        tar_dir = os.path.join(ckpt_dir, tar_name)
        with open(tar_dir, 'wb') as f:
            f.write(r.content)
        # untar file
        import tarfile
        with tarfile.open(tar_dir, 'r') as tar_ref:
            tar_ref.extractall(ckpt_dir)
            os.remove(tar_dir)

        # get prefix
        for o in os.listdir(ckpt_dir):
            folder_dir = os.path.join(ckpt_dir, o)
            break;
        unique_file_name = str(glob.glob(folder_dir + "/*data-00000-of-00001"))
        prefix = (unique_file_name.rpartition('.')[0]).split("/")[-1]

        return os.path.join(folder_dir, prefix)

    else:
        for filename in ['checkpoint', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta']:
            r = requests.get(tf_ckpt_url + "/" + filename)

            with open(os.path.join(ckpt_dir, filename), 'wb') as f:
                f.write(r.content)

        # get prefix
        unique_file_name = str(glob.glob(ckpt_dir + "/*data-00000-of-00001"))
        prefix = (unique_file_name.rpartition('.')[0]).split("/")[-1]

        return os.path.join(ckpt_dir, prefix)


def init_pytorch_model(model_name, tf_checkpoint_path):
    config_name = TFMODELS[model_name][1]
    config_module = __import__("transformers", fromlist=[config_name])
    model_config = getattr(config_module, config_name)

    parent_path = tf_checkpoint_path.rpartition('/')[0]
    config_path = glob.glob(parent_path + "/*config.json")
    config = model_config() if len(config_path) is 0 else model_config.from_json_file(str(config_path[0]))

    if TFMODELS[model_name][2] is "":
        from transformers import AutoModelForPreTraining
        init_model = AutoModelForPreTraining.from_config(config)
    else:
        model_categroy_name = TFMODELS[model_name][2]
        module = __import__("transformers", fromlist=[model_categroy_name])
        model_categroy = getattr(module, model_categroy_name)
        init_model = model_categroy(config)
    return config, init_model

def convert_tf_checkpoint_to_pytorch(config, init_model, tf_checkpoint_path):
    load_tf_weight_func_name = "load_tf_weights_in_" + TFMODELS[model_name][0]

    module = __import__("transformers", fromlist=[load_tf_weight_func_name])
    load_tf_weight_func = getattr(module, load_tf_weight_func_name)

    model = load_tf_weight_func(init_model, config, tf_checkpoint_path)
    model.eval()
    return model

def pipeline(model_name):
    if model_name not in TFMODELS:
        raise NotImplementedError()
    tf_checkpoint_path = download_tf_checkpoint(model_name)
    config, init_model = init_pytorch_model(model_name, tf_checkpoint_path)
    model = convert_tf_checkpoint_to_pytorch(config, init_model, tf_checkpoint_path)

if __name__ == '__main__':
    # For test
    for model_name in TFMODELS.keys():
        pipeline(model_name)
    print("finished")