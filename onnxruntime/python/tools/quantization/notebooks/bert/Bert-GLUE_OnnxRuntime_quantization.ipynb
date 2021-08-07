{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Copyright (c) Microsoft Corporation. All rights reserved.  \n",
    "Licensed under the MIT License."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Bert Quantization with ONNX Runtime on CPU"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this tutorial, we will load a fine tuned [HuggingFace BERT](https://huggingface.co/transformers/) model trained with [PyTorch](https://pytorch.org/) for [Microsoft Research Paraphrase Corpus (MRPC)](https://www.microsoft.com/en-us/download/details.aspx?id=52398) task , convert the model to ONNX, and then quantize PyTorch and ONNX model respectively. Finally, we will demonstrate the performance, accuracy and model size of the quantized PyTorch and OnnxRuntime model in the [General Language Understanding Evaluation benchmark (GLUE)](https://gluebenchmark.com/)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 0. Prerequisites ##\n",
    "\n",
    "If you have Jupyter Notebook, you can run this notebook directly with it. You may need to install or upgrade [PyTorch](https://pytorch.org/), [OnnxRuntime](https://microsoft.github.io/onnxruntime/), [transformers](https://huggingface.co/transformers/) and other required packages.\n",
    "\n",
    "Otherwise, you can setup a new environment. First, install [AnaConda](https://www.anaconda.com/distribution/). Then open an AnaConda prompt window and run the following commands:\n",
    "\n",
    "```console\n",
    "conda create -n cpu_env python=3.8\n",
    "conda activate cpu_env\n",
    "conda install jupyter\n",
    "jupyter notebook\n",
    "```\n",
    "The last command will launch Jupyter Notebook and we can open this notebook in browser to continue."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 0.1 Install packages\n",
    "Let's install nessasary packages to start the tutorial. We will install PyTorch 1.8, OnnxRuntime 1.7, latest ONNX, OnnxRuntime-tools, transformers, and sklearn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in links: https://download.pytorch.org/whl/torch_stable.html\n",
      "Requirement already up-to-date: install in /home/yufeng/anaconda3/lib/python3.8/site-packages (1.3.4)\n",
      "Requirement already up-to-date: torch==1.8.1+cpu in /home/yufeng/anaconda3/lib/python3.8/site-packages (1.8.1+cpu)\n",
      "Requirement already up-to-date: torchvision==0.9.1+cpu in /home/yufeng/anaconda3/lib/python3.8/site-packages (0.9.1+cpu)\n",
      "Requirement already up-to-date: torchaudio===0.8.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (0.8.1)\n",
      "Requirement already satisfied, skipping upgrade: numpy in /home/yufeng/anaconda3/lib/python3.8/site-packages (from torch==1.8.1+cpu) (1.19.2)\n",
      "Requirement already satisfied, skipping upgrade: typing-extensions in /home/yufeng/anaconda3/lib/python3.8/site-packages (from torch==1.8.1+cpu) (3.7.4.3)\n",
      "Requirement already satisfied, skipping upgrade: pillow>=4.1.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from torchvision==0.9.1+cpu) (8.0.1)\n",
      "Requirement already up-to-date: onnxruntime==1.7.0 in /home/yufeng/anaconda3/lib/python3.8/site-packages (1.7.0)\n",
      "Requirement already satisfied, skipping upgrade: numpy>=1.16.6 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime==1.7.0) (1.19.2)\n",
      "Requirement already satisfied, skipping upgrade: protobuf in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime==1.7.0) (3.15.6)\n",
      "Requirement already satisfied, skipping upgrade: six>=1.9 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from protobuf->onnxruntime==1.7.0) (1.15.0)\n",
      "Requirement already up-to-date: onnxruntime-tools in /home/yufeng/anaconda3/lib/python3.8/site-packages (1.6.0)\n",
      "Requirement already satisfied, skipping upgrade: psutil in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime-tools) (5.7.2)\n",
      "Requirement already satisfied, skipping upgrade: numpy in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime-tools) (1.19.2)\n",
      "Requirement already satisfied, skipping upgrade: packaging in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime-tools) (20.4)\n",
      "Requirement already satisfied, skipping upgrade: coloredlogs in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime-tools) (15.0)\n",
      "Requirement already satisfied, skipping upgrade: onnx in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime-tools) (1.8.1)\n",
      "Requirement already satisfied, skipping upgrade: py-cpuinfo in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime-tools) (7.0.0)\n",
      "Requirement already satisfied, skipping upgrade: py3nvml in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnxruntime-tools) (0.2.6)\n",
      "Requirement already satisfied, skipping upgrade: pyparsing>=2.0.2 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from packaging->onnxruntime-tools) (2.4.7)\n",
      "Requirement already satisfied, skipping upgrade: six in /home/yufeng/anaconda3/lib/python3.8/site-packages (from packaging->onnxruntime-tools) (1.15.0)\n",
      "Requirement already satisfied, skipping upgrade: humanfriendly>=9.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from coloredlogs->onnxruntime-tools) (9.1)\n",
      "Requirement already satisfied, skipping upgrade: typing-extensions>=3.6.2.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnx->onnxruntime-tools) (3.7.4.3)\n",
      "Requirement already satisfied, skipping upgrade: protobuf in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnx->onnxruntime-tools) (3.15.6)\n",
      "Requirement already satisfied, skipping upgrade: xmltodict in /home/yufeng/anaconda3/lib/python3.8/site-packages (from py3nvml->onnxruntime-tools) (0.12.0)\n",
      "Requirement already up-to-date: transformers in /home/yufeng/anaconda3/lib/python3.8/site-packages (4.4.2)\n",
      "Requirement already satisfied, skipping upgrade: tqdm>=4.27 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (4.50.2)\n",
      "Requirement already satisfied, skipping upgrade: tokenizers<0.11,>=0.10.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (0.10.1)\n",
      "Requirement already satisfied, skipping upgrade: regex!=2019.12.17 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (2020.10.15)\n",
      "Requirement already satisfied, skipping upgrade: numpy>=1.17 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (1.19.2)\n",
      "Requirement already satisfied, skipping upgrade: packaging in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (20.4)\n",
      "Requirement already satisfied, skipping upgrade: filelock in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (3.0.12)\n",
      "Requirement already satisfied, skipping upgrade: sacremoses in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (0.0.43)\n",
      "Requirement already satisfied, skipping upgrade: requests in /home/yufeng/anaconda3/lib/python3.8/site-packages (from transformers) (2.24.0)\n",
      "Requirement already satisfied, skipping upgrade: pyparsing>=2.0.2 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from packaging->transformers) (2.4.7)\n",
      "Requirement already satisfied, skipping upgrade: six in /home/yufeng/anaconda3/lib/python3.8/site-packages (from packaging->transformers) (1.15.0)\n",
      "Requirement already satisfied, skipping upgrade: click in /home/yufeng/anaconda3/lib/python3.8/site-packages (from sacremoses->transformers) (7.1.2)\n",
      "Requirement already satisfied, skipping upgrade: joblib in /home/yufeng/anaconda3/lib/python3.8/site-packages (from sacremoses->transformers) (0.17.0)\n",
      "Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from requests->transformers) (1.25.11)\n",
      "Requirement already satisfied, skipping upgrade: chardet<4,>=3.0.2 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from requests->transformers) (3.0.4)\n",
      "Requirement already satisfied, skipping upgrade: idna<3,>=2.5 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from requests->transformers) (2.10)\n",
      "Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from requests->transformers) (2020.6.20)\n",
      "Requirement already up-to-date: onnx in /home/yufeng/anaconda3/lib/python3.8/site-packages (1.8.1)\n",
      "Requirement already up-to-date: sklearn in /home/yufeng/anaconda3/lib/python3.8/site-packages (0.0)\n",
      "Requirement already satisfied, skipping upgrade: numpy>=1.16.6 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnx) (1.19.2)\n",
      "Requirement already satisfied, skipping upgrade: six in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnx) (1.15.0)\n",
      "Requirement already satisfied, skipping upgrade: protobuf in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnx) (3.15.6)\n",
      "Requirement already satisfied, skipping upgrade: typing-extensions>=3.6.2.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from onnx) (3.7.4.3)\n",
      "Requirement already satisfied, skipping upgrade: scikit-learn in /home/yufeng/anaconda3/lib/python3.8/site-packages (from sklearn) (0.23.2)\n",
      "Requirement already satisfied, skipping upgrade: joblib>=0.11 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from scikit-learn->sklearn) (0.17.0)\n",
      "Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from scikit-learn->sklearn) (2.1.0)\n",
      "Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /home/yufeng/anaconda3/lib/python3.8/site-packages (from scikit-learn->sklearn) (1.5.2)\n"
     ]
    }
   ],
   "source": [
    "# Install or upgrade PyTorch 1.8.0 and OnnxRuntime 1.7.0 for CPU-only.\n",
    "import sys\n",
    "!{sys.executable} -m pip install --upgrade install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html\n",
    "!{sys.executable} -m pip install --upgrade onnxruntime==1.7.0\n",
    "!{sys.executable} -m pip install --upgrade onnxruntime-tools\n",
    "\n",
    "# Install other packages used in this notebook.\n",
    "!{sys.executable} -m pip install --upgrade transformers\n",
    "!{sys.executable} -m pip install --upgrade onnx sklearn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 0.2 Download GLUE data and Fine-tune BERT model for MPRC task\n",
    "HuggingFace [text-classification examples]( https://github.com/huggingface/transformers/tree/master/examples/text-classification) shows details on how to fine-tune a MPRC tack with GLUE data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Firstly, Let's download the GLUE data with download_glue_data.py [script](https://github.com/huggingface/transformers/blob/master/utils/download_glue_data.py) and unpack it to directory glue_data under current directory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--2021-03-25 20:48:36--  https://raw.githubusercontent.com/huggingface/transformers/f98ef14d161d7bcdc9808b5ec399981481411cc1/utils/download_glue_data.py\n",
      "Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.108.133, 185.199.111.133, ...\n",
      "Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 8209 (8.0K) [text/plain]\n",
      "Saving to: ‘download_glue_data.py.2’\n",
      "\n",
      "download_glue_data. 100%[===================>]   8.02K  --.-KB/s    in 0s      \n",
      "\n",
      "2021-03-25 20:48:36 (79.6 MB/s) - ‘download_glue_data.py.2’ saved [8209/8209]\n",
      "\n",
      "Processing MRPC...\n",
      "Local MRPC data not specified, downloading data from https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt\n",
      "\tCompleted!\n",
      "cached_dev_bert-base-uncased_128_mrpc  msr_paraphrase_test.txt\t train.tsv\n",
      "dev.tsv\t\t\t\t       msr_paraphrase_train.txt\n",
      "dev_ids.tsv\t\t\t       test.tsv\n",
      "  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n",
      "                                 Dload  Upload   Total   Spent    Left  Speed\n",
      "100  386M  100  386M    0     0   224M      0  0:00:01  0:00:01 --:--:--  224M\n",
      "Archive:  MPRC.zip\n"
     ]
    }
   ],
   "source": [
    "!wget https://raw.githubusercontent.com/huggingface/transformers/f98ef14d161d7bcdc9808b5ec399981481411cc1/utils/download_glue_data.py\n",
    "!python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'\n",
    "!ls glue_data/MRPC\n",
    "!curl https://download.pytorch.org/tutorial/MRPC.zip --output MPRC.zip\n",
    "!unzip -n MPRC.zip"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Next, we can fine-tune the model based on the [MRPC example](https://github.com/huggingface/transformers/tree/master/examples/text-classification#mrpc) with command like:\n",
    "\n",
    "`\n",
    "export GLUE_DIR=./glue_data\n",
    "export TASK_NAME=MRPC\n",
    "export OUT_DIR=./$TASK_NAME/\n",
    "python ./run_glue.py \\\n",
    "    --model_type bert \\\n",
    "    --model_name_or_path bert-base-uncased \\\n",
    "    --task_name $TASK_NAME \\\n",
    "    --do_train \\\n",
    "    --do_eval \\\n",
    "    --do_lower_case \\\n",
    "    --data_dir $GLUE_DIR/$TASK_NAME \\\n",
    "    --max_seq_length 128 \\\n",
    "    --per_gpu_eval_batch_size=8   \\\n",
    "    --per_gpu_train_batch_size=8   \\\n",
    "    --learning_rate 2e-5 \\\n",
    "    --num_train_epochs 3.0 \\\n",
    "    --save_steps 100000 \\\n",
    "    --output_dir $OUT_DIR\n",
    "`\n",
    "\n",
    "In order to save time, we download the fine-tuned BERT model for MRPC task by PyTorch from:https://download.pytorch.org/tutorial/MRPC.zip."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n",
      "                                 Dload  Upload   Total   Spent    Left  Speed\n",
      "100  386M  100  386M    0     0   190M      0  0:00:02  0:00:02 --:--:--  190M\n",
      "Archive:  MPRC.zip\n"
     ]
    }
   ],
   "source": [
    "!curl https://download.pytorch.org/tutorial/MRPC.zip --output MPRC.zip\n",
    "!unzip -n MPRC.zip"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.Load and quantize model with PyTorch\n",
    "\n",
    "In this section, we will load the fine-tuned model with PyTorch, quantize it and measure the performance. We reused the code from PyTorch's [BERT quantization blog](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html) for this section."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.1 Import modules and set global configurations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this step, we import the necessary PyTorch, transformers and other necessary modules for the tutorial, and then set up the global configurations, like data & model folder, GLUE task settings, thread settings, warning settings and etc."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.8.1+cpu\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/yufeng/anaconda3/lib/python3.8/site-packages/transformers/data/processors/glue.py:175: FutureWarning: This processor will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py\n",
      "  warnings.warn(DEPRECATION_WARNING.format(\"processor\"), FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "from __future__ import absolute_import, division, print_function\n",
    "\n",
    "\n",
    "import logging\n",
    "import numpy as np\n",
    "import os\n",
    "import random\n",
    "import sys\n",
    "import time\n",
    "import torch\n",
    "\n",
    "from argparse import Namespace\n",
    "from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,\n",
    "                              TensorDataset)\n",
    "from tqdm import tqdm\n",
    "from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)\n",
    "from transformers import glue_compute_metrics as compute_metrics\n",
    "from transformers import glue_output_modes as output_modes\n",
    "from transformers import glue_processors as processors\n",
    "from transformers import glue_convert_examples_to_features as convert_examples_to_features\n",
    "\n",
    "# Setup warnings\n",
    "import warnings\n",
    "warnings.filterwarnings(\n",
    "    action='ignore',\n",
    "    category=DeprecationWarning,\n",
    "    module=r'.*'\n",
    ")\n",
    "warnings.filterwarnings(\n",
    "    action='default',\n",
    "    module=r'torch.quantization'\n",
    ")\n",
    "\n",
    "# Setup logging level to WARN. Change it accordingly\n",
    "logger = logging.getLogger(__name__)\n",
    "logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',\n",
    "                    datefmt = '%m/%d/%Y %H:%M:%S',\n",
    "                    level = logging.WARN)\n",
    "\n",
    "#logging.getLogger(\"transformers.modeling_utils\").setLevel(\n",
    "#    logging.WARN)  # Reduce logging\n",
    "\n",
    "print(torch.__version__)\n",
    "\n",
    "\n",
    "configs = Namespace()\n",
    "\n",
    "# The output directory for the fine-tuned model, $OUT_DIR.\n",
    "configs.output_dir = \"./MRPC/\"\n",
    "\n",
    "# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.\n",
    "configs.data_dir = \"./glue_data/MRPC\"\n",
    "\n",
    "# The model name or path for the pre-trained model.\n",
    "configs.model_name_or_path = \"bert-base-uncased\"\n",
    "# The maximum length of an input sequence\n",
    "configs.max_seq_length = 128\n",
    "\n",
    "# Prepare GLUE task.\n",
    "configs.task_name = \"MRPC\".lower()\n",
    "configs.processor = processors[configs.task_name]()\n",
    "configs.output_mode = output_modes[configs.task_name]\n",
    "configs.label_list = configs.processor.get_labels()\n",
    "configs.model_type = \"bert\".lower()\n",
    "configs.do_lower_case = True\n",
    "\n",
    "# Set the device, batch size, topology, and caching flags.\n",
    "configs.device = \"cpu\"\n",
    "configs.eval_batch_size = 1\n",
    "configs.n_gpu = 0\n",
    "configs.local_rank = -1\n",
    "configs.overwrite_cache = False\n",
    "\n",
    "\n",
    "# Set random seed for reproducibility.\n",
    "def set_seed(seed):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "set_seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.2 Load and quantize the fine-tuned BERT model with PyTorch \n",
    "In this step, we load the fine-tuned BERT model, and quantize it with PyTorch's dynamic quantization. And show the model size comparison between full precision and quantized model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Size (MB): 417.72998905181885\n",
      "Size (MB): 173.0945405960083\n"
     ]
    }
   ],
   "source": [
    "# load model\n",
    "model = BertForSequenceClassification.from_pretrained(configs.output_dir)\n",
    "model.to(configs.device)\n",
    "\n",
    "# quantize model\n",
    "quantized_model = torch.quantization.quantize_dynamic(\n",
    "    model, {torch.nn.Linear}, dtype=torch.qint8\n",
    ")\n",
    "\n",
    "#print(quantized_model)\n",
    "\n",
    "def print_size_of_model(model):\n",
    "    torch.save(model.state_dict(), \"temp.p\")\n",
    "    print('Size (MB):', os.path.getsize(\"temp.p\")/(1024*1024))\n",
    "    os.remove('temp.p')\n",
    "\n",
    "print_size_of_model(model)\n",
    "print_size_of_model(quantized_model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.3 Evaluate the accuracy and performance of PyTorch quantization\n",
    "This section reused the tokenize and evaluation function from [Huggingface](https://github.com/huggingface/transformers/blob/45e26125de1b9fbae46837856b1f518a4b56eb65/examples/movement-pruning/masked_run_glue.py)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating:   0%|          | 1/408 [00:00<00:49,  8.28it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Evaluating PyTorch full precision accuracy and performance:\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating: 100%|██████████| 408/408 [00:52<00:00,  7.85it/s]\n",
      "/home/yufeng/anaconda3/lib/python3.8/site-packages/transformers/data/metrics/__init__.py:66: FutureWarning: This metric will be removed from the library soon, metrics should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py\n",
      "  warnings.warn(DEPRECATION_WARNING, FutureWarning)\n",
      "/home/yufeng/anaconda3/lib/python3.8/site-packages/transformers/data/metrics/__init__.py:42: FutureWarning: This metric will be removed from the library soon, metrics should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py\n",
      "  warnings.warn(DEPRECATION_WARNING, FutureWarning)\n",
      "/home/yufeng/anaconda3/lib/python3.8/site-packages/transformers/data/metrics/__init__.py:36: FutureWarning: This metric will be removed from the library soon, metrics should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py\n",
      "  warnings.warn(DEPRECATION_WARNING, FutureWarning)\n",
      "Evaluating:   0%|          | 2/408 [00:00<00:28, 14.33it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'acc': 0.8602941176470589, 'f1': 0.9018932874354562, 'acc_and_f1': 0.8810937025412575}\n",
      "Evaluate total time (seconds): 52.0\n",
      "Evaluating PyTorch quantization accuracy and performance:\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating: 100%|██████████| 408/408 [00:26<00:00, 15.61it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'acc': 0.8504901960784313, 'f1': 0.8942807625649914, 'acc_and_f1': 0.8723854793217114}\n",
      "Evaluate total time (seconds): 26.2\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# coding=utf-8\n",
    "# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.\n",
    "# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.\n",
    "#\n",
    "# Licensed under the Apache License, Version 2.0 (the \"License\");\n",
    "# you may not use this file except in compliance with the License.\n",
    "# You may obtain a copy of the License at\n",
    "#\n",
    "#     http://www.apache.org/licenses/LICENSE-2.0\n",
    "#\n",
    "# Unless required by applicable law or agreed to in writing, software\n",
    "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
    "# See the License for the specific language governing permissions and\n",
    "# limitations under the License.\n",
    "\n",
    "def evaluate(args, model, tokenizer, prefix=\"\"):\n",
    "    # Loop to handle MNLI double evaluation (matched, mis-matched)\n",
    "    eval_task_names = (\"mnli\", \"mnli-mm\") if args.task_name == \"mnli\" else (args.task_name,)\n",
    "    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == \"mnli\" else (args.output_dir,)\n",
    "\n",
    "    results = {}\n",
    "    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):\n",
    "        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)\n",
    "\n",
    "        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:\n",
    "            os.makedirs(eval_output_dir)\n",
    "\n",
    "        # Note that DistributedSampler samples randomly\n",
    "        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)\n",
    "        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)\n",
    "\n",
    "        # multi-gpu eval\n",
    "        if args.n_gpu > 1:\n",
    "            model = torch.nn.DataParallel(model)\n",
    "\n",
    "        # Eval!\n",
    "        logger.info(\"***** Running evaluation {} *****\".format(prefix))\n",
    "        logger.info(\"  Num examples = %d\", len(eval_dataset))\n",
    "        logger.info(\"  Batch size = %d\", args.eval_batch_size)\n",
    "        eval_loss = 0.0\n",
    "        nb_eval_steps = 0\n",
    "        preds = None\n",
    "        out_label_ids = None\n",
    "        for batch in tqdm(eval_dataloader, desc=\"Evaluating\"):\n",
    "            model.eval()\n",
    "            batch = tuple(t.to(args.device) for t in batch)\n",
    "\n",
    "            with torch.no_grad():\n",
    "                inputs = {'input_ids':      batch[0],\n",
    "                          'attention_mask': batch[1],\n",
    "                          'labels':         batch[3]}\n",
    "                if args.model_type != 'distilbert':\n",
    "                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids\n",
    "                outputs = model(**inputs)\n",
    "                tmp_eval_loss, logits = outputs[:2]\n",
    "\n",
    "                eval_loss += tmp_eval_loss.mean().item()\n",
    "            nb_eval_steps += 1\n",
    "            if preds is None:\n",
    "                preds = logits.detach().cpu().numpy()\n",
    "                out_label_ids = inputs['labels'].detach().cpu().numpy()\n",
    "            else:\n",
    "                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)\n",
    "                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)\n",
    "\n",
    "        eval_loss = eval_loss / nb_eval_steps\n",
    "        if args.output_mode == \"classification\":\n",
    "            preds = np.argmax(preds, axis=1)\n",
    "        elif args.output_mode == \"regression\":\n",
    "            preds = np.squeeze(preds)\n",
    "        result = compute_metrics(eval_task, preds, out_label_ids)\n",
    "        results.update(result)\n",
    "\n",
    "        output_eval_file = os.path.join(eval_output_dir, prefix, \"eval_results.txt\")\n",
    "        with open(output_eval_file, \"w\") as writer:\n",
    "            logger.info(\"***** Eval results {} *****\".format(prefix))\n",
    "            for key in sorted(result.keys()):\n",
    "                logger.info(\"  %s = %s\", key, str(result[key]))\n",
    "                writer.write(\"%s = %s\\n\" % (key, str(result[key])))\n",
    "\n",
    "    return results\n",
    "\n",
    "\n",
    "def load_and_cache_examples(args, task, tokenizer, evaluate=False):\n",
    "    if args.local_rank not in [-1, 0] and not evaluate:\n",
    "        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache\n",
    "\n",
    "    processor = processors[task]()\n",
    "    output_mode = output_modes[task]\n",
    "    # Load data features from cache or dataset file\n",
    "    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(\n",
    "        'dev' if evaluate else 'train',\n",
    "        list(filter(None, args.model_name_or_path.split('/'))).pop(),\n",
    "        str(args.max_seq_length),\n",
    "        str(task)))\n",
    "    if os.path.exists(cached_features_file) and not args.overwrite_cache:\n",
    "        logger.info(\"Loading features from cached file %s\", cached_features_file)\n",
    "        features = torch.load(cached_features_file)\n",
    "    else:\n",
    "        logger.info(\"Creating features from dataset file at %s\", args.data_dir)\n",
    "        label_list = processor.get_labels()\n",
    "        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:\n",
    "            # HACK(label indices are swapped in RoBERTa pretrained model)\n",
    "            label_list[1], label_list[2] = label_list[2], label_list[1]\n",
    "        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)\n",
    "        features = convert_examples_to_features(examples,\n",
    "                                                tokenizer,\n",
    "                                                label_list=label_list,\n",
    "                                                max_length=args.max_seq_length,\n",
    "                                                output_mode=output_mode,\n",
    "        )\n",
    "        if args.local_rank in [-1, 0]:\n",
    "            logger.info(\"Saving features into cached file %s\", cached_features_file)\n",
    "            torch.save(features, cached_features_file)\n",
    "\n",
    "    if args.local_rank == 0 and not evaluate:\n",
    "        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache\n",
    "\n",
    "    # Convert to Tensors and build dataset\n",
    "    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)\n",
    "    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)\n",
    "    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)\n",
    "    if output_mode == \"classification\":\n",
    "        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)\n",
    "    elif output_mode == \"regression\":\n",
    "        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)\n",
    "\n",
    "    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)\n",
    "    return dataset\n",
    "\n",
    "def time_model_evaluation(model, configs, tokenizer):\n",
    "    eval_start_time = time.time()\n",
    "    result = evaluate(configs, model, tokenizer, prefix=\"\")\n",
    "    eval_end_time = time.time()\n",
    "    eval_duration_time = eval_end_time - eval_start_time\n",
    "    print(result)\n",
    "    print(\"Evaluate total time (seconds): {0:.1f}\".format(eval_duration_time))\n",
    "\n",
    "# define the tokenizer\n",
    "tokenizer = BertTokenizer.from_pretrained(\n",
    "    configs.output_dir, do_lower_case=configs.do_lower_case)\n",
    "    \n",
    "# Evaluate the original FP32 BERT model\n",
    "print('Evaluating PyTorch full precision accuracy and performance:')\n",
    "time_model_evaluation(model, configs, tokenizer)\n",
    "\n",
    "# Evaluate the INT8 BERT model after the dynamic quantization\n",
    "print('Evaluating PyTorch quantization accuracy and performance:')\n",
    "time_model_evaluation(quantized_model, configs, tokenizer)\n",
    "\n",
    "# Serialize the quantized model\n",
    "quantized_output_dir = configs.output_dir + \"quantized/\"\n",
    "if not os.path.exists(quantized_output_dir):\n",
    "    os.makedirs(quantized_output_dir)\n",
    "    quantized_model.save_pretrained(quantized_output_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Quantization and Inference with ORT ##\n",
    "In this section, we will demonstrate how to export the PyTorch model to ONNX, quantize the exported ONNX model, and infererence the quantized model with ONNXRuntime."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 Export to ONNX model and optimize with ONNXRuntime-tools\n",
    "This step will export the PyTorch model to ONNX and then optimize the ONNX model with [ONNXRuntime-tools](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers), which is an offline optimizer tool for transformers based models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/yufeng/anaconda3/lib/python3.8/site-packages/transformers/modeling_utils.py:1791: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  assert all(\n"
     ]
    }
   ],
   "source": [
    "import onnxruntime\n",
    "\n",
    "def export_onnx_model(args, model, tokenizer, onnx_model_path):\n",
    "    with torch.no_grad():\n",
    "        inputs = {'input_ids':      torch.ones(1,128, dtype=torch.int64),\n",
    "                    'attention_mask': torch.ones(1,128, dtype=torch.int64),\n",
    "                    'token_type_ids': torch.ones(1,128, dtype=torch.int64)}\n",
    "        outputs = model(**inputs)\n",
    "\n",
    "        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}\n",
    "        torch.onnx.export(model,                                            # model being run\n",
    "                    (inputs['input_ids'],                             # model input (or a tuple for multiple inputs)\n",
    "                    inputs['attention_mask'], \n",
    "                    inputs['token_type_ids']),                                         # model input (or a tuple for multiple inputs)\n",
    "                    onnx_model_path,                                # where to save the model (can be a file or file-like object)\n",
    "                    opset_version=11,                                 # the ONNX version to export the model to\n",
    "                    do_constant_folding=True,                         # whether to execute constant folding for optimization\n",
    "                    input_names=['input_ids',                         # the model's input names\n",
    "                                'input_mask', \n",
    "                                'segment_ids'],\n",
    "                    output_names=['output'],                    # the model's output names\n",
    "                    dynamic_axes={'input_ids': symbolic_names,        # variable length axes\n",
    "                                'input_mask' : symbolic_names,\n",
    "                                'segment_ids' : symbolic_names})\n",
    "        logger.info(\"ONNX Model exported to {0}\".format(onnx_model_path))\n",
    "\n",
    "export_onnx_model(configs, model, tokenizer, \"bert.onnx\")\n",
    "\n",
    "# optimize transformer-based models with onnxruntime-tools\n",
    "from onnxruntime_tools import optimizer\n",
    "from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions\n",
    "\n",
    "# disable embedding layer norm optimization for better model size reduction\n",
    "opt_options = BertOptimizationOptions('bert')\n",
    "opt_options.enable_embed_layer_norm = False\n",
    "\n",
    "opt_model = optimizer.optimize_model(\n",
    "    'bert.onnx',\n",
    "    'bert', \n",
    "    num_heads=12,\n",
    "    hidden_size=768,\n",
    "    optimization_options=opt_options)\n",
    "opt_model.save_model_to_file('bert.opt.onnx')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 Quantize ONNX model\n",
    "We will call [onnxruntime.quantization.quantize](https://github.com/microsoft/onnxruntime/blob/fe0b2b2abd494b7ff14c00c0f2c51e0ccf2a3094/onnxruntime/python/tools/quantization/README.md) to apply quantization on the HuggingFace BERT model. It supports dynamic quantization with IntegerOps and static quantization with QLinearOps. For activation ONNXRuntime supports only uint8 format for now, and for weight ONNXRuntime supports both int8 and uint8 format.\n",
    "\n",
    "We apply dynamic quantization for BERT model and use int8 for weight."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ONNX full precision model size (MB): 417.66954708099365\n",
      "ONNX quantized model size (MB): 104.80786514282227\n"
     ]
    }
   ],
   "source": [
    "def quantize_onnx_model(onnx_model_path, quantized_model_path):\n",
    "    from onnxruntime.quantization import quantize_dynamic, QuantType\n",
    "    import onnx\n",
    "    onnx_opt_model = onnx.load(onnx_model_path)\n",
    "    quantize_dynamic(onnx_model_path,\n",
    "                     quantized_model_path,\n",
    "                     weight_type=QuantType.QInt8)\n",
    "\n",
    "    logger.info(f\"quantized model saved to:{quantized_model_path}\")\n",
    "\n",
    "quantize_onnx_model('bert.opt.onnx', 'bert.opt.quant.onnx')\n",
    "\n",
    "print('ONNX full precision model size (MB):', os.path.getsize(\"bert.opt.onnx\")/(1024*1024))\n",
    "print('ONNX quantized model size (MB):', os.path.getsize(\"bert.opt.quant.onnx\")/(1024*1024))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3 Evaluate ONNX quantization performance and accuracy\n",
    "\n",
    "In this step, we will evalute OnnxRuntime quantization with GLUE data set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "Evaluating:   0%|          | 0/408 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Evaluating ONNXRuntime full precision accuracy and performance:\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating: 100%|██████████| 408/408 [00:40<00:00, 10.04it/s]\n",
      "Evaluating:   0%|          | 0/408 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'acc': 0.8602941176470589, 'f1': 0.9018932874354562, 'acc_and_f1': 0.8810937025412575}\n",
      "Evaluate total time (seconds): 41.0\n",
      "Evaluating ONNXRuntime quantization accuracy and performance:\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluating: 100%|██████████| 408/408 [00:18<00:00, 21.84it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'acc': 0.8529411764705882, 'f1': 0.8986486486486487, 'acc_and_f1': 0.8757949125596185}\n",
      "Evaluate total time (seconds): 18.8\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "def evaluate_onnx(args, model_path, tokenizer, prefix=\"\"):\n",
    "\n",
    "    sess_options = onnxruntime.SessionOptions()\n",
    "    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL\n",
    "    session = onnxruntime.InferenceSession(model_path, sess_options)\n",
    "\n",
    "    # Loop to handle MNLI double evaluation (matched, mis-matched)\n",
    "    eval_task_names = (\"mnli\", \"mnli-mm\") if args.task_name == \"mnli\" else (args.task_name,)\n",
    "    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == \"mnli\" else (args.output_dir,)\n",
    "\n",
    "    results = {}\n",
    "    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):\n",
    "        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)\n",
    "\n",
    "        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:\n",
    "            os.makedirs(eval_output_dir)\n",
    "\n",
    "        # Note that DistributedSampler samples randomly\n",
    "        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)\n",
    "        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)\n",
    "\n",
    "        # multi-gpu eval\n",
    "        if args.n_gpu > 1:\n",
    "            model = torch.nn.DataParallel(model)\n",
    "\n",
    "        # Eval!\n",
    "        logger.info(\"***** Running evaluation {} *****\".format(prefix))\n",
    "        logger.info(\"  Num examples = %d\", len(eval_dataset))\n",
    "        logger.info(\"  Batch size = %d\", args.eval_batch_size)\n",
    "        #eval_loss = 0.0\n",
    "        #nb_eval_steps = 0\n",
    "        preds = None\n",
    "        out_label_ids = None\n",
    "        for batch in tqdm(eval_dataloader, desc=\"Evaluating\"):\n",
    "            batch = tuple(t.detach().cpu().numpy() for t in batch)\n",
    "            ort_inputs = {\n",
    "                                'input_ids':  batch[0],\n",
    "                                'input_mask': batch[1],\n",
    "                                'segment_ids': batch[2]\n",
    "                            }\n",
    "            logits = np.reshape(session.run(None, ort_inputs), (-1,2))\n",
    "            if preds is None:\n",
    "                preds = logits\n",
    "                #print(preds.shape)\n",
    "                out_label_ids = batch[3]\n",
    "            else:\n",
    "                preds = np.append(preds, logits, axis=0)\n",
    "                out_label_ids = np.append(out_label_ids, batch[3], axis=0)\n",
    "\n",
    "        #print(preds.shap)\n",
    "        #eval_loss = eval_loss / nb_eval_steps\n",
    "        if args.output_mode == \"classification\":\n",
    "            preds = np.argmax(preds, axis=1)\n",
    "        elif args.output_mode == \"regression\":\n",
    "            preds = np.squeeze(preds)\n",
    "        #print(preds)\n",
    "        #print(out_label_ids)\n",
    "        result = compute_metrics(eval_task, preds, out_label_ids)\n",
    "        results.update(result)\n",
    "\n",
    "        output_eval_file = os.path.join(eval_output_dir, prefix + \"_eval_results.txt\")\n",
    "        with open(output_eval_file, \"w\") as writer:\n",
    "            logger.info(\"***** Eval results {} *****\".format(prefix))\n",
    "            for key in sorted(result.keys()):\n",
    "                logger.info(\"  %s = %s\", key, str(result[key]))\n",
    "                writer.write(\"%s = %s\\n\" % (key, str(result[key])))\n",
    "\n",
    "    return results\n",
    "\n",
    "\n",
    "def time_ort_model_evaluation(model_path, configs, tokenizer, prefix=\"\"):\n",
    "    eval_start_time = time.time()\n",
    "    result = evaluate_onnx(configs, model_path, tokenizer, prefix=prefix)\n",
    "    eval_end_time = time.time()\n",
    "    eval_duration_time = eval_end_time - eval_start_time\n",
    "    print(result)\n",
    "    print(\"Evaluate total time (seconds): {0:.1f}\".format(eval_duration_time))\n",
    "\n",
    "print('Evaluating ONNXRuntime full precision accuracy and performance:')\n",
    "time_ort_model_evaluation('bert.opt.onnx', configs, tokenizer, \"onnx.opt\")\n",
    "    \n",
    "print('Evaluating ONNXRuntime quantization accuracy and performance:')\n",
    "time_ort_model_evaluation('bert.opt.quant.onnx', configs, tokenizer, \"onnx.opt.quant\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3 Summary\n",
    "In this tutorial, we demonstrated how to quantize a fine-tuned BERT model for MPRC task on GLUE data set. Let's summarize the main metrics of quantization.\n",
    "\n",
    "### Model Size\n",
    "PyTorch quantizes torch.nn.Linear modules only and reduce the model from 438 MB to 173 MB. OnnxRuntime quantizes not only Linear(MatMul), but also the embedding layer. It achieves almost the ideal model size reduction with quantization.\n",
    "\n",
    "| Engine | Full Precision(MB) | Quantized(MB) |\n",
    "| --- | --- | --- |\n",
    "| PyTorch 1.8 | 417.7 | 173.1 |\n",
    "| ORT 1.7 | 417.7 | 104.5 |\n",
    "\n",
    "### Accuracy\n",
    "OnnxRuntime achieves a little bit better accuracy and F1 score, even though it has small model size.\n",
    "\n",
    "| Metrics | Full Precision | PyTorch 1.8 Quantization | ORT 1.7 Quantization |\n",
    "| --- | --- | --- | --- |\n",
    "| Accuracy | 0.86029 | 0.85049 | 0.85294 |\n",
    "| F1 score | 0.90189 | 0.89428 | 0.89865 |\n",
    "| Acc and F1 | 0.88109 | 0.87239 | 0.87579 |\n",
    "\n",
    "### Performance\n",
    "\n",
    "The evaluation data set has 408 sample. Table below shows the performance on **Azure VM: Standard E4ds_v4 (4 vcpus, 32 GiB memory)**. Comparing with PyTorch full precision, PyTorch quantization achieves ~2x speedup, and ORT quantization achieves ~1.73x speedup. And ORT quantization can achieve ~2.77x speedup, comparing with PyTorch quantization. \n",
    "You can run the [benchmark.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/benchmark.py) for comparison on more models.\n",
    "\n",
    "|Engine | Full Precision Latency(s) | Quantized(s) |\n",
    "| --- | --- | --- |\n",
    "| PyTorch 1.8 | 52.0 | 26.2 |\n",
    "| ORT 1.7 | 41.0 | 18.8 |"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "base"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
