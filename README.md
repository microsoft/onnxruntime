## README.md
<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

**ONNX Runtime is a cross-platform inference and training machine-learning accelerator**.

**ONNX Runtime inference** can enable faster customer experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc. ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-inferencing)

**ONNX Runtime training** can accelerate the model training time on multi-node NVIDIA GPUs for transformer models with a one-line addition for existing PyTorch training scripts. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-training)
Accelerate BERT pre-training with ONNX Runtime
This example uses ONNX Runtime to pre-train the BERT PyTorch model maintained at https://github.com/NVIDIA/DeepLearningExamples.

You can run the training in Azure Machine Learning or on an Azure VM with NVIDIA GPU.

Setup
Clone this repo

git clone https://github.com/microsoft/onnxruntime-training-examples.git
cd onnxruntime-training-examples
Clone download code and model

git clone --no-checkout https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/
git checkout 4733603577080dbd1bdcd51864f31e45d5196704
cd ..
Create working directory

mkdir -p workspace
mv DeepLearningExamples/PyTorch/LanguageModeling/BERT/ workspace
rm -rf DeepLearningExamples
cp -r ./nvidia-bert/ort_addon/* workspace/BERT
cd workspace
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor/
git checkout e4abb4cbd019b0257824ee47c23dd163919b731b
cd ../../ 
Download and prepare data
The following are a minimal set of instructions to download and process one of the datasets used for BERT pre-training.

To include additional datasets, and for more details, refer to DeepLearningExamples.

Note that the datasets used for BERT pre-training need a large amount of disk space. After processing, the data should be made available for training. Due to the large size of the data copy, we recommend that you execute the steps below in the training environment itself or in an environment from where data transfer to training environment will be fast and efficient.

Check pre-requisites

Python 3.6
Natural Language Toolkit (NLTK) python3-pip install nltk
Download and prepare Wikicorpus training data in HDF5 format.

export BERT_PREP_WORKING_DIR=./workspace/BERT/data/

# Download google_pretrained_weights
python ./workspace/BERT/data/bertPrep.py --action download --dataset google_pretrained_weights

# Download wikicorpus_en via wget
mkdir -p ./workspace/BERT/data/download/wikicorpus_en
cd ./workspace/BERT/data/download/wikicorpus_en
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -dv enwiki-latest-pages-articles.xml.bz2
mv enwiki-latest-pages-articles.xml wikicorpus_en.xml
cd ../../../../..

# Fix path issue to use BERT_PREP_WORKING_DIR as prefix for path instead of hard-coded prefix
sed -i "s/path_to_wikiextractor_in_container = '/path_to_wikiextractor_in_container = './g" ./workspace/BERT/data/bertPrep.py

# Format text files
python ./workspace/BERT/data/bertPrep.py --action text_formatting --dataset wikicorpus_en

# Shard text files
python ./workspace/BERT/data/bertPrep.py --action sharding --dataset wikicorpus_en

# Fix path to workspace to allow running outside of the docker container
sed -i "s/python \/workspace\/bert/python .\/workspace\/BERT/g" ./workspace/BERT/data/bertPrep.py

# Create HDF5 files Phase 1
python ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 128 \
  --max_predictions_per_seq 20 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

# Create HDF5 files Phase 2
python ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 512 \
--max_predictions_per_seq 80 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1
Make data accessible for training

After completing the steps above, data in hdf5 format will be available at the following locations:

Phase 1 data: ./workspace/BERT/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/train
Phase 2 data: ./workspace/BERT/data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/train
Below instructions refer to these hdf5 data files as the data to make accessible to training process.

BERT pre-training with ONNX Runtime in Azure Machine Learning
Data Transfer

Transfer training data to Azure blob storage
To transfer the data to an Azure blob storage using Azure CLI, use command:

az storage blob upload-batch --account-name <storage-name> -d <container-name> -s ./workspace/BERT/data
Register the blob container as a data store
Mount the data store in the compute targets used for training
Please refer to the storage guidance for details on using Azure storage account for training in Azure Machine Learning.

Execute pre-training

The BERT pre-training job in Azure Machine Learning can be launched using either of these environments:

Azure Machine Learning Compute Instance to run the Jupyter notebook.
Azure Machine Learning SDK
You will need a GPU optimized compute target - either NCv3 or NDv2 series, to execute this pre-training job.

Execute the steps in the Python notebook azureml-notebooks/run-pretraining.ipynb within your environment. If you have a local setup to run an Azure ML notebook, you could run the steps in the notebook in that environment. Otherwise, a compute instance in Azure Machine Learning could be created and used to run the steps.

BERT pre-training with ONNX Runtime directly on ND40rs_v2 (or similar NVIDIA capable Azure VM)
Check pre-requisites

CUDA 10.2
Docker
NVIDIA docker toolkit
Build the ONNX Runtime Docker image

Build the onnxruntime wheel from source into a Docker image.

cd nvidia-bert/docker
bash build.sh
cd ../..
Tag this image onnxruntime-pytorch-for-bert`
To build and install the onnxruntime wheel on the host machine, follow steps here

Set correct paths to training data for docker image.

Edit nvidia-bert/docker/launch.sh.

...
-v <replace-with-path-to-phase1-hdf5-training-data>:/data/128
-v <replace-with-path-to-phase2-hdf5-training-data>:/data/512
...
The two directories must contain the hdf5 training files.

Set the number of GPUs and per GPU limit.

Edit workspace/BERT/scripts/run_pretraining_ort.sh.

num_gpus=${4:-8}
gpu_memory_limit_gb=${26:-"32"}
Modify other training parameters as needed.

Edit workspace/BERT/scripts/run_pretraining_ort.sh.

seed=${12:-42}

accumulate_gradients=${10:-"true"}
deepspeed_zero_stage=${27:-"false"}

train_batch_size=${1:-16320}
learning_rate=${2:-"6e-3"}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-7038}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-340}

train_batch_size_phase2=${17:-8160}
learning_rate_phase2=${18:-"4e-3"}
warmup_proportion_phase2=${19:-"0.128"}
train_steps_phase2=${20:-1563}
gradient_accumulation_steps_phase2=${11:-1020}
The above defaults are tuned for an Azure NC24rs_v3.

The training batch size refers to the number of samples a single GPU sees before weights are updated. The training is performed over local and global steps. A local step refers to a single backpropagation execution on the model to calculate its gradient. These gradients are accumulated every local step until weights are updated in a global step. The microbatch size is samples a single GPU sees in a single backpropagation execution step. The microbatch size will be the training batch size divided by gradient accumulation steps.

Note: The effective batch size will be (number of GPUs) x train_batch_size (per GPU). In general we recommend setting the effective batch size to ~64,000 for phase 1 and ~32,000 for phase 2. The number of gradient accumulation steps should be minimized without overflowing the GPU memory (i.e. maximizes microbatch size).

Consult Parameters section by NVIDIA for additional details.

Launch interactive container.

cd workspace/BERT
bash ../../nvidia-bert/docker/launch.sh
Launch pre-training run

bash /workspace/bert/scripts/run_pretraining_ort.sh
If you get memory errors, try reducing the batch size or enabling the partition optimizer flag.

Fine-tuning
For fine-tuning tasks, follow model_evaluation.md

## Get Started

http://onnxruntime.ai/
[Overview](https://www.onnxruntime.ai/docs/)
[Tutorials](https://www.onnxruntime.ai/docs/tutorials/)
[Inferencing](https://www.onnxruntime.ai/docs/tutorials/inferencing/)
[Training](https://www.onnxruntime.ai/docs/tutorials/training/)
[How To](https://www.onnxruntime.ai/docs/how-to)
[Install](https://www.onnxruntime.ai/docs/how-to/install.html)
[Build](https://www.onnxruntime.ai/docs/how-to/build/)
[Tune performance](https://www.onnxruntime.ai/docs/how-to/tune-performance.html)
[Quantize models](https://www.onnxruntime.ai/docs/how-to/quantization.html)
[Deploy on mobile](https://www.onnxruntime.ai/docs/how-to/deploy-on-mobile.html)
[Use custom ops](https://www.onnxruntime.ai/docs/how-to/add-custom-op.html)
[Add a new EP](https://www.onnxruntime.ai/docs/how-to/add-execution-provider.html)
[Reference](https://www.onnxruntime.ai/docs/reference)
[API documentation](https://www.onnxruntime.ai/docs/reference/api/)
[Execution Providers](https://www.onnxruntime.ai/docs/reference/execution-providers/)
[Releases and servicing](https://www.onnxruntime.ai/docs/reference/releases-servicing.html)
[Citing](https://www.onnxruntime.ai/docs/reference/citing.html)
[Additional resources](https://www.onnxruntime.ai/docs/resources/)

## Build Pipeline Status
|System|CPU|GPU|EPs|
|---|---|---|---|
|Windows|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20TensorRT%20CI%20Pipeline?label=Windows+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=47)|
|Linux|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20Minimal%20Build%20E2E%20CI%20Pipeline?label=Linux+CPU+Minimal+Build)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=64)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20x64%20NoContribops%20CI%20Pipeline?label=Linux+CPU+x64+No+Contrib+Ops)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=110)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/centos7_cpu?label=Linux+CentOS7)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=78)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-ci-pipeline?label=Linux+CPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=86)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20TensorRT%20CI%20Pipeline?label=Linux+GPU+TensorRT)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=45)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-distributed?label=Distributed+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=140)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/orttraining-linux-gpu-ci-pipeline?label=Linux+GPU+Training)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=84)|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20NUPHAR%20CI%20Pipeline?label=Linux+NUPHAR)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=110)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20OpenVINO%20CI%20Pipeline%20v2?label=Linux+OpenVINO)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=108)|
|Mac|[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)<br>[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20NoContribops%20CI%20Pipeline?label=MacOS+NoContribops)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=65)|||
|Android|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Android%20CI%20Pipeline?label=Android)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=53)|
|iOS|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/iOS%20CI%20Pipeline?label=iOS)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=134)|
|WebAssembly|||[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20WebAssembly%20CI%20Pipeline?label=WASM)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=161)|


## Data/Telemetry

Windows distributions of this project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

## Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For feature requests or bug reports, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

For general discussion or questions, please use [Github Discussions](https://github.com/microsoft/onnxruntime/discussions).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

This project is licensed under the [MIT License](LICENSE).
