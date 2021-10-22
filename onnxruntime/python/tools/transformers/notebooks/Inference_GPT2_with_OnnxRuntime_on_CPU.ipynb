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
    "# Inference PyTorch GPT2 Model with ONNX Runtime on CPU\n",
    "\n",
    "In this tutorial, you'll be introduced to how to load a GPT2 model from PyTorch, convert it to ONNX, and inference it using ONNX Runtime using IO Binding. Note that past state is used to get better performance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prerequisites ##\n",
    "\n",
    "If you have Jupyter Notebook, you may directly run this notebook. We will use pip to install or upgrade [PyTorch](https://pytorch.org/), [OnnxRuntime](https://microsoft.github.io/onnxruntime/) and other required packages.\n",
    "\n",
    "Otherwise, you can setup a new environment. First, we install [AnaConda](https://www.anaconda.com/distribution/). Then open an AnaConda prompt window and run the following commands:\n",
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
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in links: https://download.pytorch.org/whl/torch_stable.html\n",
      "Requirement already up-to-date: torch==1.6.0+cpu in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (1.6.0+cpu)\n",
      "Requirement already up-to-date: torchvision==0.7.0+cpu in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (0.7.0+cpu)\n",
      "Requirement already satisfied, skipping upgrade: future in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from torch==1.6.0+cpu) (0.18.2)\n",
      "Requirement already satisfied, skipping upgrade: numpy in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from torch==1.6.0+cpu) (1.18.1)\n",
      "Requirement already satisfied, skipping upgrade: pillow>=4.1.1 in c:\\users\\tianl\\appdata\\roaming\\python\\python36\\site-packages (from torchvision==0.7.0+cpu) (7.0.0)\n",
      "Requirement already satisfied: onnxruntime==1.5.1 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (1.5.1)\n",
      "Requirement already satisfied: numpy>=1.16.6 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from onnxruntime==1.5.1) (1.18.1)\n",
      "Requirement already satisfied: protobuf in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from onnxruntime==1.5.1) (3.11.3)\n",
      "Requirement already satisfied: six>=1.9 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from protobuf->onnxruntime==1.5.1) (1.14.0)\n",
      "Requirement already satisfied: setuptools in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from protobuf->onnxruntime==1.5.1) (45.2.0.post20200210)\n",
      "Requirement already satisfied: transformers==3.0.2 in d:\\git\\transformers\\src (3.0.2)\n",
      "Requirement already satisfied: numpy in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (1.18.1)\n",
      "Requirement already satisfied: tokenizers==0.8.1.rc2 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (0.8.1rc2)\n",
      "Requirement already satisfied: packaging in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (20.1)\n",
      "Requirement already satisfied: filelock in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (3.0.12)\n",
      "Requirement already satisfied: requests in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (2.23.0)\n",
      "Requirement already satisfied: tqdm>=4.27 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (4.43.0)\n",
      "Requirement already satisfied: regex!=2019.12.17 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (2020.2.20)\n",
      "Requirement already satisfied: sentencepiece!=0.1.92 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (0.1.85)\n",
      "Requirement already satisfied: sacremoses in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (0.0.38)\n",
      "Requirement already satisfied: dataclasses in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from transformers==3.0.2) (0.7)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from packaging->transformers==3.0.2) (2.4.6)\n",
      "Requirement already satisfied: six in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from packaging->transformers==3.0.2) (1.14.0)\n",
      "Requirement already satisfied: idna<3,>=2.5 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from requests->transformers==3.0.2) (2.9)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from requests->transformers==3.0.2) (3.0.4)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from requests->transformers==3.0.2) (2020.4.5.1)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from requests->transformers==3.0.2) (1.25.8)\n",
      "Requirement already satisfied: joblib in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from sacremoses->transformers==3.0.2) (0.14.1)\n",
      "Requirement already satisfied: click in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from sacremoses->transformers==3.0.2) (7.0)\n",
      "Requirement already satisfied: onnx in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (1.7.0)\n",
      "Requirement already satisfied: psutil in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (5.7.0)\n",
      "Requirement already satisfied: pytz in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (2019.3)\n",
      "Requirement already satisfied: pandas in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (1.0.1)\n",
      "Requirement already satisfied: py-cpuinfo in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (5.0.0)\n",
      "Requirement already satisfied: py3nvml in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (0.2.5)\n",
      "Requirement already satisfied: netron in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (3.9.6)\n",
      "Requirement already satisfied: typing-extensions>=3.6.2.1 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from onnx) (3.7.4.1)\n",
      "Requirement already satisfied: six in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from onnx) (1.14.0)\n",
      "Requirement already satisfied: protobuf in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from onnx) (3.11.3)\n",
      "Requirement already satisfied: numpy in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from onnx) (1.18.1)\n",
      "Requirement already satisfied: python-dateutil>=2.6.1 in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from pandas) (2.8.1)\n",
      "Requirement already satisfied: xmltodict in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from py3nvml) (0.12.0)\n",
      "Requirement already satisfied: setuptools in d:\\anaconda3\\envs\\cpu_env\\lib\\site-packages (from protobuf->onnx) (45.2.0.post20200210)\n"
     ]
    }
   ],
   "source": [
    "# Install PyTorch 1.6.0 and OnnxRuntime 1.5.1 for CPU-only.\n",
    "import sys\n",
    "if sys.platform == 'darwin': # Mac\n",
    "    !{sys.executable} -m pip install --upgrade torch torchvision\n",
    "else:\n",
    "    !{sys.executable} -m pip install --upgrade torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html\n",
    "!{sys.executable} -m pip install onnxruntime==1.5.1\n",
    "\n",
    "# Install other packages used in this notebook.\n",
    "!{sys.executable} -m pip install transformers==3.0.2\n",
    "!{sys.executable} -m pip install onnx onnxconverter_common psutil pytz pandas py-cpuinfo py3nvml netron"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "# Create a cache directory to store pretrained model.\n",
    "cache_dir = os.path.join(\".\", \"cache_models\")\n",
    "if not os.path.exists(cache_dir):\n",
    "    os.makedirs(cache_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Convert GPT2 model from PyTorch to ONNX ##\n",
    "\n",
    "We have a script [convert_to_onnx.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/convert_to_onnx.py) that could help you to convert GPT2 with past state to ONNX. \n",
    "\n",
    "The script accepts a pretrained model name or path of a checkpoint directory as input, and converts the model to ONNX. It also verifies that the ONNX model could generate same input as the pytorch model. The usage is like \n",
    "```\n",
    "python -m onnxruntime.transformers.convert_to_onnx -m model_name_or_path --output gpt2.onnx -o -p fp32|fp16|int8\n",
    "```\n",
    "The -p option can be used to choose the precision: fp32 (float32), fp16 (mixed precision) or int8 (quantization). The -o option will generate optimized model, which is required for fp16 or int8.\n",
    "\n",
    "Here we use a pretrained model as example:"
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
      "GPT2Config {\n",
      "  \"activation_function\": \"gelu_new\",\n",
      "  \"architectures\": [\n",
      "    \"GPT2LMHeadModel\"\n",
      "  ],\n",
      "  \"attn_pdrop\": 0.1,\n",
      "  \"bos_token_id\": 50256,\n",
      "  \"embd_pdrop\": 0.1,\n",
      "  \"eos_token_id\": 50256,\n",
      "  \"initializer_range\": 0.02,\n",
      "  \"layer_norm_epsilon\": 1e-05,\n",
      "  \"model_type\": \"gpt2\",\n",
      "  \"n_ctx\": 1024,\n",
      "  \"n_embd\": 768,\n",
      "  \"n_head\": 12,\n",
      "  \"n_inner\": null,\n",
      "  \"n_layer\": 12,\n",
      "  \"n_positions\": 1024,\n",
      "  \"resid_pdrop\": 0.1,\n",
      "  \"summary_activation\": null,\n",
      "  \"summary_first_dropout\": 0.1,\n",
      "  \"summary_proj_to_labels\": true,\n",
      "  \"summary_type\": \"cls_index\",\n",
      "  \"summary_use_proj\": true,\n",
      "  \"task_specific_params\": {\n",
      "    \"text-generation\": {\n",
      "      \"do_sample\": true,\n",
      "      \"max_length\": 50\n",
      "    }\n",
      "  },\n",
      "  \"vocab_size\": 50257\n",
      "}\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel\n",
    "from transformers import AutoConfig\n",
    "import torch\n",
    "\n",
    "model_name_or_path = \"gpt2\"\n",
    "config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)\n",
    "model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)\n",
    "device = torch.device(\"cpu\")\n",
    "model.eval().to(device)\n",
    "\n",
    "print(model.config)\n",
    "\n",
    "num_attention_heads = model.config.n_head\n",
    "hidden_size = model.config.n_embd\n",
    "num_layer = model.config.n_layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:714: FutureWarning: The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.\n",
      "  FutureWarning,\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:560: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  assert batch_size > 0, \"batch_size has to be defined and > 0\"\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:166: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  w = w / (float(v.size(-1)) ** 0.5)\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:171: TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  mask = self.bias[:, :, ns - nd : ns, :ns]\n"
     ]
    }
   ],
   "source": [
    "onnx_model_path = \"gpt2.onnx\"\n",
    "Gpt2Helper.export_onnx(model, device, onnx_model_path) # add parameter use_external_data_format=True when model size > 2 GB"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## PyTorch Inference using Huggingface Transformers##\n",
    "\n",
    "In the following, we will use an example input to get the output from PyTorch for comparison purpose.\n",
    "For the first inference, there is no any past state. We can prepare empty state for input."
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
      "input_ids tensor([[50256, 50256, 50256, 50256, 13466,  7541,   287, 15489,  1989],\n",
      "        [ 1456,   318,   281,  1672,   286,   308,   457,    17,  2746]])\n",
      "attention_mask tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1.],\n",
      "        [1., 1., 1., 1., 1., 1., 1., 1., 1.]])\n",
      "position_ids tensor([[0, 0, 0, 0, 0, 1, 2, 3, 4],\n",
      "        [0, 1, 2, 3, 4, 5, 6, 7, 8]])\n"
     ]
    }
   ],
   "source": [
    "from transformers import AutoTokenizer\n",
    "\n",
    "EXAMPLE_Text = ['best hotel in bay area', 'here is an example of gpt2 model']\n",
    "\n",
    "def get_tokenizer(model_name_or_path, cache_dir):\n",
    "    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)\n",
    "    tokenizer.padding_side = \"left\"\n",
    "    tokenizer.pad_token = tokenizer.eos_token\n",
    "    #okenizer.add_special_tokens({'pad_token': '[PAD]'})\n",
    "    return tokenizer\n",
    "\n",
    "def get_example_inputs(prompt_text=EXAMPLE_Text):    \n",
    "    tokenizer = get_tokenizer(model_name_or_path, cache_dir)\n",
    "    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)\n",
    "\n",
    "    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)\n",
    "    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)\n",
    "    position_ids = (attention_mask.long().cumsum(-1) - 1)\n",
    "    position_ids.masked_fill_(position_ids < 0, 0)\n",
    "\n",
    "    #Empty Past State for generating first word\n",
    "    empty_past = []\n",
    "    batch_size = input_ids.size(0)\n",
    "    sequence_length = input_ids.size(1)\n",
    "    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]\n",
    "    for i in range(num_layer):\n",
    "        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))\n",
    "       \n",
    "    return input_ids, attention_mask, position_ids, empty_past\n",
    "\n",
    "\n",
    "from transformers import GPT2LMHeadModel\n",
    "torch_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)\n",
    "device = torch.device(\"cpu\")\n",
    "torch_model.eval().to(device)\n",
    "\n",
    "input_ids, attention_mask, position_ids, empty_past = get_example_inputs()\n",
    "print(\"input_ids\", input_ids)\n",
    "print(\"attention_mask\", attention_mask)\n",
    "print(\"position_ids\", position_ids)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "with torch.no_grad():\n",
    "    torch_output = torch_model(input_ids, past=empty_past, attention_mask=attention_mask, position_ids=position_ids)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ONNX Runtime Inference ##\n",
    "\n",
    "We can use ONNX Runtime to inference. The inputs are dictionary with name and numpy array as value, and the output is list of numpy array. Note that both input and output are in CPU. When you run the inference in GPU, it will involve data copy between CPU and GPU for input and output.\n",
    "\n",
    "Let's create an inference session for ONNX Runtime given the exported ONNX model, and see the output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import onnxruntime\n",
    "import numpy\n",
    "\n",
    "input_ids, attention_mask, position_ids, empty_past = get_example_inputs()\n",
    "\n",
    "onnx_model_path = \"gpt2.onnx\"\n",
    "session = onnxruntime.InferenceSession(onnx_model_path)\n",
    "ort_inputs = {'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),\n",
    "              'attention_mask' : numpy.ascontiguousarray(attention_mask.cpu().numpy()),\n",
    "              'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy())\n",
    "             }\n",
    "for i, past_i in enumerate(empty_past):\n",
    "    ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())\n",
    "ort_outputs = session.run(None, ort_inputs)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can compare the outputs from PyTorch and ONNX Runtime. Logits are very close (max difference is 1E-4)."
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
      "max logits diff (ignored padding) tensor(6.8665e-05)\n"
     ]
    }
   ],
   "source": [
    "logits_masked_diff = (torch_output[0] - ort_outputs[0]) * attention_mask.unsqueeze(2)\n",
    "max_logits_diff = logits_masked_diff.abs().max()\n",
    "print(\"max logits diff (ignored padding)\", max_logits_diff)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ONNX Runtime Inference with IO Binding ##\n",
    "\n",
    "To avoid data copy for input and output, ONNX Runtime also supports IO Binding. User could provide some buffer for input and outputs. For GPU inference, the buffer can be in GPU to reduce memory copy between CPU and GPU. This is helpful for high performance inference in GPU. For GPT-2, IO Binding might help the performance when batch size or (past) sequence length is large."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):\n",
    "    output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),\n",
    "                                                 past_sequence_length=past[0].size(3),\n",
    "                                                 sequence_length=input_ids.size(1),\n",
    "                                                 config=config)\n",
    "    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device)\n",
    "\n",
    "    io_binding = Gpt2Helper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past,\n",
    "                                               output_buffers, output_shapes)\n",
    "    session.run_with_iobinding(io_binding)\n",
    "\n",
    "    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes,\n",
    "                                                            return_numpy=False)\n",
    "    return outputs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that the result is exactly same with/without IO Binding:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "IO Binding result is good\n"
     ]
    }
   ],
   "source": [
    "input_ids, attention_mask, position_ids, empty_past = get_example_inputs()\n",
    "outputs = inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, empty_past)\n",
    "for i in range(len(outputs)):\n",
    "    assert torch.eq(outputs[i], torch.from_numpy(ort_outputs[i])).all()\n",
    "print(\"IO Binding result is good\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Batch Text Generation ##\n",
    "\n",
    "Here is an example for text generation using ONNX Runtime or PyTorch. For ONNX Runtime, IO Binding is used for better performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test_generation(tokenizer, input_text, ort_session=None, num_tokens_to_produce = 30):\n",
    "    use_onnxruntime = (ort_session is not None)\n",
    "    print(\"Text generation using\", \"OnnxRuntime\" if use_onnxruntime else \"PyTorch\", \"...\")\n",
    "    eos_token_id = tokenizer.eos_token_id\n",
    "    \n",
    "    input_ids, attention_mask, position_ids, past = get_example_inputs(input_text)\n",
    "    batch_size = input_ids.size(0)\n",
    "\n",
    "    has_eos = torch.zeros(batch_size, dtype=torch.bool)\n",
    "\n",
    "    all_token_ids = input_ids.clone()\n",
    "\n",
    "    for step in range(num_tokens_to_produce):\n",
    "        if ort_session is not None:\n",
    "            outputs = inference_with_io_binding(ort_session, config, input_ids, position_ids, attention_mask, past)\n",
    "        else:\n",
    "            outputs = torch_model(input_ids, attention_mask=attention_mask, position_ids=position_ids, past=past)  \n",
    "\n",
    "        next_token_logits = outputs[0][:, -1, :]\n",
    "        # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.\n",
    "        next_tokens = torch.argmax(next_token_logits, dim=-1)\n",
    "\n",
    "        has_eos = has_eos | (next_tokens == eos_token_id)\n",
    "        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)\n",
    "        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)\n",
    "\n",
    "        # Update input_ids, attention_mask, position_ids and past\n",
    "        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)    \n",
    "        position_ids = (position_ids[:,-1] + 1).reshape(batch_size,1)\n",
    "        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)    \n",
    "\n",
    "        past = []\n",
    "        if not use_onnxruntime:\n",
    "            past = list(outputs[1]) # past in torch output is tuple\n",
    "        else:\n",
    "            for i in range(num_layer):\n",
    "                past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[i + 1].clone().detach()\n",
    "                past.append(past_i.to(device))\n",
    "\n",
    "        if torch.all(has_eos):\n",
    "            break\n",
    "\n",
    "    for i, output in enumerate(all_token_ids):\n",
    "        print(\"------------\")\n",
    "        print(tokenizer.decode(output, skip_special_tokens=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Text generation using OnnxRuntime ...\n",
      "------------\n",
      "best hotel in bay area.\n",
      "\n",
      "The hotel is located in the historic Bayview neighborhood of San Francisco.\n",
      "\n",
      "The hotel is open daily from 9 a.m.\n",
      "------------\n",
      "here is an example of gpt2 model.\n",
      "\n",
      "The gpt2 model is a simple, but powerful, way to generate a GPT2-like data structure. It is a\n"
     ]
    }
   ],
   "source": [
    "tokenizer = get_tokenizer(model_name_or_path, cache_dir)\n",
    "input_text = EXAMPLE_Text\n",
    "test_generation(tokenizer, input_text, ort_session=session)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we use PyTorch to run again and we can see that the result is exactly same."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Text generation using PyTorch ...\n",
      "------------\n",
      "best hotel in bay area.\n",
      "\n",
      "The hotel is located in the historic Bayview neighborhood of San Francisco.\n",
      "\n",
      "The hotel is open daily from 9 a.m.\n",
      "------------\n",
      "here is an example of gpt2 model.\n",
      "\n",
      "The gpt2 model is a simple, but powerful, way to generate a GPT2-like data structure. It is a\n"
     ]
    }
   ],
   "source": [
    "test_generation(tokenizer, input_text)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Int8 Quantization ##\n",
    "Next, we will apply dynamic quantization to the model. We optimize the model before quantization to get better performance.\n",
    "\n",
    "Note that text generation result from fp32 and int8 models could be quite different. User shall evaluate the precision metric for your application for both fp32 and int8 models. If the quality of int8 model result is acceptable, you will be glad to find that it is faster than fp32 model in inference. \n",
    "\n",
    "Note that you can leverage [quantization aware training (QAT)](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) for accuracy improvement if needed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Warning: onnxruntime.quantization.quantize is deprecated.\n",
      "         Please use quantize_static for static quantization, quantize_dynamic for dynamic quantization.\n"
     ]
    }
   ],
   "source": [
    "from onnxruntime.transformers.quantize_helper import QuantizeHelper\n",
    "\n",
    "optimized_fp32_model_path = \"gpt2_fp32.onnx\"\n",
    "quantized_int8_model_path = \"gpt2_int8.onnx\"\n",
    "Gpt2Helper.optimize_onnx(\"gpt2.onnx\", optimized_fp32_model_path, False, model.config.num_attention_heads, model.config.hidden_size)\n",
    "QuantizeHelper.quantize_onnx_model(optimized_fp32_model_path, quantized_int8_model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Text generation using OnnxRuntime ...\n",
      "------------\n",
      "bert model optimization, and the NLP model is a generalizable and robust model.\n"
     ]
    }
   ],
   "source": [
    "session_int8 = onnxruntime.InferenceSession(quantized_int8_model_path)\n",
    "input_text = ['bert model optimization']\n",
    "test_generation(tokenizer, input_text, ort_session=session_int8, num_tokens_to_produce=14)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Benchmark ##\n",
    "There is a tool benchmark_gpt2.py, which can be used to measure the performance of GPT-2 by PyTorch, ONNX Runtime without/with IO Binding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ATen/Parallel:\n",
      "\tat::get_num_threads() : 12\n",
      "\tat::get_num_interop_threads() : 6\n",
      "OpenMP 2019\n",
      "\tomp_get_max_threads() : 12\n",
      "Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191125 for Intel(R) 64 architecture applications\n",
      "\tmkl_get_max_threads() : 12\n",
      "Intel(R) MKL-DNN v1.5.0 (Git Hash e2ac1fac44c5078ca927cb9b90e1b3066a0b2ed0)\n",
      "std::thread::hardware_concurrency() : 12\n",
      "Environment variables:\n",
      "\tOMP_NUM_THREADS : [not set]\n",
      "\tMKL_NUM_THREADS : [not set]\n",
      "ATen parallel backend: OpenMP\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2020-09-30 18:44:40.720277: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library cudart64_101.dll\n",
      "Arguments:Namespace(batch_sizes=[1], cache_dir='.\\\\cache_models', include_copy_output_latency=False, model_class='GPT2LMHeadModel', model_name_or_path='gpt2', onnx_dir='.\\\\onnx_models', optimize_onnx=True, past_sequence_lengths=[8, 16, 32, 64, 128, 256], precision=<Precision.FLOAT32: 'fp32'>, result_csv=None, test_times=100, thread_num=-1, torchscript=False, use_gpu=False, validate_onnx=False, verbose=False)\n",
      "PyTorch Version:1.6.0+cpu\n",
      "Transformers Version:3.0.2\n",
      "Onnxruntime Version:1.5.1\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:714: FutureWarning: The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.\n",
      "  FutureWarning,\n",
      "Shapes: input_ids=torch.Size([1, 1]) past=torch.Size([2, 1, 12, 1, 64]) output=torch.Size([1, 1, 50257]) present=torch.Size([2, 1, 12, 2, 64])\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:560: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  assert batch_size > 0, \"batch_size has to be defined and > 0\"\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:166: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  w = w / (float(v.size(-1)) ** 0.5)\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:171: TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  mask = self.bias[:, :, ns - nd : ns, :ns]\n",
      "Fused LayerNormalization count: 25\n",
      "Fused FastGelu count: 12\n",
      "Fused Attention(with past) count: 12\n",
      "Graph pruned: 0 inputs, 0 outputs and 741 nodes are removed\n",
      "Graph pruned: 0 inputs, 0 outputs and 312 nodes are removed\n",
      "postprocess: remove Reshape count:48\n",
      "Fused FastGelu(add bias) count: 12\n",
      "opset verion: 11\n",
      "Output model to .\\onnx_models\\gpt2_past_fp32.onnx\n",
      "batch_size=1, past_sequence_length=8, torch_latency=40.68, ort_latency=24.07, ort_io_latency=24.03\n",
      "batch_size=1, past_sequence_length=16, torch_latency=40.87, ort_latency=23.14, ort_io_latency=22.27\n",
      "batch_size=1, past_sequence_length=32, torch_latency=41.36, ort_latency=23.74, ort_io_latency=23.05\n",
      "batch_size=1, past_sequence_length=64, torch_latency=42.97, ort_latency=26.25, ort_io_latency=23.64\n",
      "batch_size=1, past_sequence_length=128, torch_latency=44.30, ort_latency=30.48, ort_io_latency=25.85\n",
      "batch_size=1, past_sequence_length=256, torch_latency=54.77, ort_latency=40.60, ort_io_latency=28.20\n",
      "Results are saved to file benchmark_result_20200930-184558.csv\n"
     ]
    }
   ],
   "source": [
    "!{sys.executable} -m onnxruntime.transformers.benchmark_gpt2 -m gpt2 -o"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ATen/Parallel:\n",
      "\tat::get_num_threads() : 12\n",
      "\tat::get_num_interop_threads() : 6\n",
      "OpenMP 2019\n",
      "\tomp_get_max_threads() : 12\n",
      "Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191125 for Intel(R) 64 architecture applications\n",
      "\tmkl_get_max_threads() : 12\n",
      "Intel(R) MKL-DNN v1.5.0 (Git Hash e2ac1fac44c5078ca927cb9b90e1b3066a0b2ed0)\n",
      "std::thread::hardware_concurrency() : 12\n",
      "Environment variables:\n",
      "\tOMP_NUM_THREADS : [not set]\n",
      "\tMKL_NUM_THREADS : [not set]\n",
      "ATen parallel backend: OpenMP\n",
      "\n",
      "Warning: onnxruntime.quantization.quantize is deprecated.\n",
      "         Please use quantize_static for static quantization, quantize_dynamic for dynamic quantization.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2020-09-30 18:47:09.756025: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library cudart64_101.dll\n",
      "Arguments:Namespace(batch_sizes=[1], cache_dir='.\\\\cache_models', include_copy_output_latency=False, model_class='GPT2LMHeadModel', model_name_or_path='gpt2', onnx_dir='.\\\\onnx_models', optimize_onnx=True, past_sequence_lengths=[8, 16, 32, 64, 128, 256], precision=<Precision.INT8: 'int8'>, result_csv=None, test_times=100, thread_num=-1, torchscript=False, use_gpu=False, validate_onnx=False, verbose=False)\n",
      "PyTorch Version:1.6.0+cpu\n",
      "Transformers Version:3.0.2\n",
      "Onnxruntime Version:1.5.1\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:714: FutureWarning: The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.\n",
      "  FutureWarning,\n",
      "Shapes: input_ids=torch.Size([1, 1]) past=torch.Size([2, 1, 12, 1, 64]) output=torch.Size([1, 1, 50257]) present=torch.Size([2, 1, 12, 2, 64])\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:560: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  assert batch_size > 0, \"batch_size has to be defined and > 0\"\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:166: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  w = w / (float(v.size(-1)) ** 0.5)\n",
      "d:\\git\\transformers\\src\\transformers\\modeling_gpt2.py:171: TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  mask = self.bias[:, :, ns - nd : ns, :ns]\n",
      "Fused LayerNormalization count: 25\n",
      "Fused FastGelu count: 12\n",
      "Fused Attention(with past) count: 12\n",
      "Graph pruned: 0 inputs, 0 outputs and 741 nodes are removed\n",
      "Graph pruned: 0 inputs, 0 outputs and 312 nodes are removed\n",
      "postprocess: remove Reshape count:48\n",
      "Fused FastGelu(add bias) count: 12\n",
      "opset verion: 11\n",
      "Output model to .\\onnx_models\\gpt2_past_int8.onnx\n",
      "quantizing model...\n",
      "Size of full precision ONNX model(MB):621.9615631103516\n",
      "quantized model saved to:.\\onnx_models\\gpt2_past_int8.onnx\n",
      "Size of quantized ONNX model(MB):155.89412593841553\n",
      "Size of full precision Torch model(MB):486.7606954574585\n",
      "Size of quantized Torch model(MB):280.60562801361084\n",
      "finished quantizing model\n",
      "batch_size=1, past_sequence_length=8, torch_latency=19.50, ort_latency=11.35, ort_io_latency=11.24\n",
      "batch_size=1, past_sequence_length=16, torch_latency=20.13, ort_latency=11.53, ort_io_latency=10.24\n",
      "batch_size=1, past_sequence_length=32, torch_latency=20.54, ort_latency=12.05, ort_io_latency=11.97\n",
      "batch_size=1, past_sequence_length=64, torch_latency=21.29, ort_latency=13.90, ort_io_latency=12.15\n",
      "batch_size=1, past_sequence_length=128, torch_latency=23.40, ort_latency=19.22, ort_io_latency=13.96\n",
      "batch_size=1, past_sequence_length=256, torch_latency=30.26, ort_latency=29.05, ort_io_latency=16.77\n",
      "Results are saved to file benchmark_result_20200930-184855.csv\n"
     ]
    }
   ],
   "source": [
    "!{sys.executable} -m onnxruntime.transformers.benchmark_gpt2 -m gpt2 -o --precision int8"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that quantized model has significant speed up (close to 2x).\n",
    "\n",
    "### Test Environment ###\n",
    "The following is the hardware of the test machine, and software version:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\n",
      "  \"gpu\": {\n",
      "    \"driver_version\": \"451.67\",\n",
      "    \"devices\": [\n",
      "      {\n",
      "        \"memory_total\": 8589934592,\n",
      "        \"memory_available\": 8480882688,\n",
      "        \"name\": \"GeForce GTX 1070\"\n",
      "      }\n",
      "    ]\n",
      "  },\n",
      "  \"cpu\": {\n",
      "    \"brand\": \"Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz\",\n",
      "    \"cores\": 6,\n",
      "    \"logical_cores\": 12,\n",
      "    \"hz\": \"3.1920 GHz\",\n",
      "    \"l2_cache\": \"1536 KB\",\n",
      "    \"flags\": [\n",
      "      \"3dnow\",\n",
      "      \"3dnowprefetch\",\n",
      "      \"abm\",\n",
      "      \"acpi\",\n",
      "      \"adx\",\n",
      "      \"aes\",\n",
      "      \"apic\",\n",
      "      \"avx\",\n",
      "      \"avx2\",\n",
      "      \"bmi1\",\n",
      "      \"bmi2\",\n",
      "      \"clflush\",\n",
      "      \"clflushopt\",\n",
      "      \"cmov\",\n",
      "      \"cx16\",\n",
      "      \"cx8\",\n",
      "      \"de\",\n",
      "      \"dtes64\",\n",
      "      \"dts\",\n",
      "      \"erms\",\n",
      "      \"est\",\n",
      "      \"f16c\",\n",
      "      \"fma\",\n",
      "      \"fpu\",\n",
      "      \"fxsr\",\n",
      "      \"hle\",\n",
      "      \"ht\",\n",
      "      \"hypervisor\",\n",
      "      \"ia64\",\n",
      "      \"invpcid\",\n",
      "      \"lahf_lm\",\n",
      "      \"mca\",\n",
      "      \"mce\",\n",
      "      \"mmx\",\n",
      "      \"movbe\",\n",
      "      \"mpx\",\n",
      "      \"msr\",\n",
      "      \"mtrr\",\n",
      "      \"osxsave\",\n",
      "      \"pae\",\n",
      "      \"pat\",\n",
      "      \"pbe\",\n",
      "      \"pcid\",\n",
      "      \"pclmulqdq\",\n",
      "      \"pdcm\",\n",
      "      \"pge\",\n",
      "      \"pni\",\n",
      "      \"popcnt\",\n",
      "      \"pse\",\n",
      "      \"pse36\",\n",
      "      \"rdrnd\",\n",
      "      \"rdseed\",\n",
      "      \"rtm\",\n",
      "      \"sep\",\n",
      "      \"serial\",\n",
      "      \"sgx\",\n",
      "      \"sgx_lc\",\n",
      "      \"smap\",\n",
      "      \"smep\",\n",
      "      \"ss\",\n",
      "      \"sse\",\n",
      "      \"sse2\",\n",
      "      \"sse4_1\",\n",
      "      \"sse4_2\",\n",
      "      \"ssse3\",\n",
      "      \"tm\",\n",
      "      \"tm2\",\n",
      "      \"tsc\",\n",
      "      \"vme\",\n",
      "      \"x2apic\",\n",
      "      \"xsave\",\n",
      "      \"xtpr\"\n",
      "    ],\n",
      "    \"processor\": \"Intel64 Family 6 Model 158 Stepping 10, GenuineIntel\"\n",
      "  },\n",
      "  \"memory\": {\n",
      "    \"total\": 16971276288,\n",
      "    \"available\": 6431543296\n",
      "  },\n",
      "  \"python\": \"3.6.10.final.0 (64 bit)\",\n",
      "  \"os\": \"Windows-10-10.0.19041-SP0\",\n",
      "  \"onnxruntime\": {\n",
      "    \"version\": \"1.5.1\",\n",
      "    \"support_gpu\": false\n",
      "  },\n",
      "  \"onnxruntime_tools\": {\n",
      "    \"version\": \"1.4.4\"\n",
      "  },\n",
      "  \"pytorch\": {\n",
      "    \"version\": \"1.6.0+cpu\",\n",
      "    \"support_gpu\": false,\n",
      "    \"cuda\": null\n",
      "  },\n",
      "  \"tensorflow\": {\n",
      "    \"version\": \"2.3.0\",\n",
      "    \"git_version\": \"v2.3.0-rc2-23-gb36436b087\",\n",
      "    \"support_gpu\": true\n",
      "  }\n",
      "}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2020-09-30 18:49:40.600527: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library cudart64_101.dll\n"
     ]
    }
   ],
   "source": [
    "!{sys.executable} -m onnxruntime.transformers.machine_info --silent"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "cpu_env",
   "language": "python",
   "name": "cpu_env"
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
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
