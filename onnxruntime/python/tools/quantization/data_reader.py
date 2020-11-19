from onnxruntime.quantization import CalibrationDataReader 
from .processing import yolov3_preprocess_func
import onnxruntime
from argparse import Namespace

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                               TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

import os
import logging
import numpy as np
import os
import random
import sys
import time
import torch

# Setup logging level to WARN. Change it accordingly
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.INFO)  # Reduce logging

logger = logging.getLogger(__name__)


def parse_annotations(filename):
    import json
    annotations = {}
    with open(filename, 'r') as f:
        annotations = json.load(f)


    img_name_to_img_id = {}
    for image in annotations["images"]:
        file_name = image["file_name"]
        img_name_to_img_id[file_name] = image["id"]

    return img_name_to_img_id

class YoloV3DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder,
                       start_index=0,
                       size_limit=0,
                       augmented_model_path='augmented_model.onnx',
                       is_validation=False,
                       save_bbox_to_image=False,
                       annotations='./annotations/instances_val2017.json'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.start_index = start_index
        self.size_limit = size_limit
        self.is_validation = is_validation
        self.save_bbox_to_image = save_bbox_to_image
        self.annotations = annotations

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            print(session.get_inputs()[0].shape)
            width = 416
            height = 416
            nchw_data_list, filename_list, image_size_list = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nchw_data_list)

            print(nchw_data_list)

            data = []
            if self.is_validation:
                img_name_to_img_id = parse_annotations(self.annotations)
                for i in range(len(nchw_data_list)):
                    nhwc_data = nchw_data_list[i]
                    file_name = filename_list[i]
                    if self.save_bbox_to_image:
                        data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "image_id": img_name_to_img_id[file_name], "file_name": file_name})
                    else:
                        data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "image_id": img_name_to_img_id[file_name]})

            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = nchw_data_list[i]
                    file_name = filename_list[i]
                    data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
                    # self.enum_data_dicts = iter([{input_name: nhwc_data, "image_shape": arr} for nhwc_data in nchw_data_list])

            self.enum_data_dicts = iter(data)

            # annotations = './annotations/instances_val2017.json'
            # img_name_to_img_id = parse_annotations(annotations)
            # data = []
            # for i in range(len(nchw_data_list)):
                # nhwc_data = nchw_data_list[i]
                # file_name = filename_list[i]
                # if self.is_validation:
                    # data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "file_name": file_name, "id": img_name_to_img_id[file_name]})
                # else:
                    # data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
            # self.enum_data_dicts = iter(data)

        return next(self.enum_data_dicts, None)

class YoloV3VisionDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder,
                       start_index=0,
                       size_limit=0,
                       augmented_model_path='augmented_model.onnx',
                       is_validation=False,
                       save_bbox_to_image=False,
                       annotations='./annotations/instances_val2017.json'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.start_index = start_index
        self.size_limit = size_limit
        self.is_validation = is_validation
        self.save_bbox_to_image = save_bbox_to_image
        self.annotations = annotations

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            print(session.get_inputs()[0].shape)
            width = 512 
            height = 288 
            nchw_data_list, filename_list, _ = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.size_limit)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nchw_data_list)

            print(nchw_data_list)

            data = []
            if self.is_validation:
                img_name_to_img_id = parse_annotations(self.annotations)
                for i in range(len(nchw_data_list)):
                    nhwc_data = nchw_data_list[i]
                    file_name = filename_list[i]
                    if self.save_bbox_to_image:
                        data.append({input_name: nhwc_data, "image_id": img_name_to_img_id[file_name], "file_name": file_name})
                    else:
                        data.append({input_name: nhwc_data, "image_id": img_name_to_img_id[file_name]})

            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = nchw_data_list[i]
                    file_name = filename_list[i]
                    data.append({input_name: nhwc_data})
                    # self.enum_data_dicts = iter([{input_name: nhwc_data, "image_shape": arr} for nhwc_data in nchw_data_list])

            self.enum_data_dicts = iter(data)

        return next(self.enum_data_dicts, None)

class BertDataReader(CalibrationDataReader):
    def __init__(self, augmented_model_path='bert_augmented_model.onnx', is_validation=False):
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.is_validation = is_validation

        if not self.check_prerequisites():
            return 

        configs = Namespace()

        # The output directory for the fine-tuned model, $OUT_DIR.
        configs.output_dir = "./MRPC/"

        # The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
        configs.data_dir = "./glue_data/MRPC"

        # The model name or path for the pre-trained model.
        configs.model_name_or_path = "bert-base-uncased"
        # The maximum length of an input sequence
        configs.max_seq_length = 128

        # Prepare GLUE task.
        configs.task_name = "MRPC".lower()
        configs.processor = processors[configs.task_name]()
        configs.output_mode = output_modes[configs.task_name]
        configs.label_list = configs.processor.get_labels()
        configs.model_type = "bert".lower()
        configs.do_lower_case = True

        # Set the device, batch size, topology, and caching flags.
        configs.device = "cpu"
        configs.eval_batch_size = 1
        configs.n_gpu = 0
        configs.local_rank = -1
        configs.overwrite_cache = False

        self.configs = configs


        # define the tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            configs.output_dir, do_lower_case=configs.do_lower_case)
        self.tokenizer = tokenizer

    def check_prerequisites(self):
        if not os.path.exists('glue_data'):
            print("Please download GLUE data with following commands:\n\
                   wget https://raw.githubusercontent.com/huggingface/transformers/f98ef14d161d7bcdc9808b5ec399981481411cc1/utils/download_glue_data.py\n\
                   python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'")
            return False
        return True

    def load_and_cache_examples(self, args, task, tokenizer, evaluate=False):

        processor = self.configs.processor
        output_mode = self.configs.output_mode


        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = processors[task]()
        output_mode = output_modes[task]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
            features = convert_examples_to_features(examples,
                                                    tokenizer,
                                                    label_list=label_list,
                                                    max_length=args.max_seq_length,
                                                    output_mode=output_mode,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset


    def get_next(self):

        if self.preprocess_flag:
            self.preprocess_flag = False

            configs = self.configs
            tokenizer = self.tokenizer

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            eval_task_names = ("mnli", "mnli-mm") if configs.task_name == "mnli" else (configs.task_name,)
            eval_outputs_dirs = (configs.output_dir, configs.output_dir + '-MM') if configs.task_name == "mnli" else (configs.output_dir,)

            results = {}
            data = []
            for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
                eval_dataset = self.load_and_cache_examples(configs, eval_task, tokenizer, evaluate=True)

                if not os.path.exists(eval_output_dir) and configs.local_rank in [-1, 0]:
                    os.makedirs(eval_output_dir)

                # Note that DistributedSampler samples randomly
                eval_sampler = SequentialSampler(eval_dataset) if configs.local_rank == -1 else DistributedSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=configs.eval_batch_size)

                # multi-gpu eval
                if configs.n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                logger.info("  Num examples = %d", len(eval_dataset))
                logger.info("  Batch size = %d", configs.eval_batch_size)
                #eval_loss = 0.0
                #nb_eval_steps = 0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.detach().cpu().numpy() for t in batch)
                    ort_inputs = {
                                        'input_ids':  batch[0],
                                        'input_mask': batch[1],
                                        'segment_ids': batch[2]
                                    }
                    data.append(ort_inputs)

            self.enum_data_dicts = iter(data)
        return next(self.enum_data_dicts, None)
