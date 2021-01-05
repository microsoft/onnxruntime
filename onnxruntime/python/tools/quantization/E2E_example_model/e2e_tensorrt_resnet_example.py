import os
import onnx
import glob
import scipy.io
import numpy as np
from onnxruntime.quantization import get_calibrator, ImageNetDataReader, ImageClassificationEvaluator, generate_calibration_table, write_calibration_table

def get_prediction_evaluation(model_path, input_name, batch_size, dataset_path, dataset_offset, dataset_size, synset_id, providers=["TensorrtExecutionProvider"]):
    data_reader = ImageNetDataReader(dataset_path, start_index=dataset_offset, end_index=dataset_offset+dataset_size, stride=dataset_size, batch_size=batch_size, model_path=model_path, input_name=input_name)
    evaluator = ImageClassificationEvaluator(model_path, synset_id, data_reader, providers=providers)
    evaluator.predict()
    result = evaluator.get_result()
    evaluator.evaluate(result)

def get_calibration_table(model_path, input_name, batch_size, augmented_model_path, dataset_path, calibration_dataset_size, use_existing_augmented_model=False, use_existing_calibration_table=False):
    calibrator = get_calibrator(model_path, None, augmented_model_path=augmented_model_path)    
    start_index = 0
    stride=calibration_dataset_size
    for i in range(0, calibration_dataset_size, stride):
        data_reader = ImageNetDataReader(dataset_path,start_index=start_index, end_index=start_index+stride, stride=stride, batch_size=batch_size, model_path=augmented_model_path, input_name=input_name)
        calibrator.set_data_reader(data_reader)
        generate_calibration_table(calibrator, model_path, augmented_model_path, False, data_reader)
        start_index += stride
    write_calibration_table(calibrator.get_calibration_cache())

def get_synset_id(dataset_path, offset, dataset_size):
    ilsvrc2012_meta = scipy.io.loadmat(dataset_path + "/devkit/data/meta.mat")
    id_to_synset = {}    
    for i in range(1000):
        id = int(ilsvrc2012_meta["synsets"][i,0][0][0][0])
        id_to_synset[id] = ilsvrc2012_meta["synsets"][i,0][1][0]
    
    synset_to_id = {}
    file = open(dataset_path + "/synset_words.txt","r")
    index = 0
    for line in file:
        parts = line.split(" ")
        synset_to_id[parts[0]] = index
        index = index + 1
    file.close()
  
    file = open(dataset_path + "/devkit/data/ILSVRC2012_validation_ground_truth.txt","r")
    id = file.read().strip().split("\n")
    id = list(map(int, id))
    file.close()
    
    image_names = os.listdir(dataset_path + "/val")
    image_names.sort()
    image_names = image_names[offset : offset + dataset_size]
    seq_num = []
    for file in image_names:
        seq_num.append(int(file.split("_")[-1].split(".")[0]))
    id = np.array([id[index-1] for index in seq_num])       
    synset_id = np.array([synset_to_id[id_to_synset[index]] for index in id])

    # one-hot encoding
    synset_id_onehot = np.zeros((len(synset_id), 1000), dtype=np.float32)
    for i, id in enumerate(synset_id):
        synset_id_onehot[i, id] = 1.0
    return synset_id_onehot

def convert_model_batch_to_dynamic(model_path):
    model = onnx.load(model_path)
    input = model.graph.input
    input_name = input[0].name
    shape = input[0].type.tensor_type.shape
    dim = shape.dim
    if isinstance(dim[0].dim_value, int):
        dim[0].dim_param = 'N'
        model = onnx.shape_inference.infer_shapes(model)        
        model_name = model_path.split(".")
        model_path = model_name[0] + "_dynamic.onnx"
        onnx.save(model, model_path)
    return [model_path, input_name]

def get_dataset_size(dataset_path, calibration_dataset_size):
    total_dataset_size = len(os.listdir(dataset_path + "/val"))        
    if calibration_dataset_size > total_dataset_size:
        print("Warning: calibration data size is bigger than available dataset. Will assign half of the dataset for calibration")
        calibration_dataset_size = total_dataset_size // 2
    calibration_dataset_size =  (calibration_dataset_size // batch_size) * batch_size
    if calibration_dataset_size == 0:
        print("Warning: No dataset is assigned for calibration. Please use bigger dataset")

    prediction_dataset_size = ((total_dataset_size - calibration_dataset_size) // batch_size) * batch_size
    if prediction_dataset_size <= 0:
        print("Warning: No dataset is assigned for evaluation. Please use bigger dataset")
    return [calibration_dataset_size, prediction_dataset_size] 

if __name__ == '__main__':
    '''
    TensorRT EP INT8 Inference on Resnet model

    The script is using ILSVRC2012 ImageNet dataset for calibration and prediction.
    Please prepare the dataset as below, 
    1. Create dataset folder 'ILSVRC2012' in workspace.
    2. Download ILSVRC2012 validation dataset and development kit from http://www.image-net.org/challenges/LSVRC/2012/downloads.
    3. Extract validation dataset JPEG files to 'ILSVRC2012/val'.
    4. Extract development kit to 'ILSVRC2012/devkit'. Two files are used in the development kit, 'ILSVRC2012_validation_ground_truth.txt' and 'meta.mat'.
    5. Download 'synset_words.txt' from https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt into 'ILSVRC2012/'.
    
    Please download Resnet50 model from ONNX model zoo https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v2-7.tar.gz
    Untar the model into the workspace
    '''

    # Dataset settings
    model_path = "./resnet50-v2-7.onnx"
    ilsvrc2012_dataset_path = "./ILSVRC2012"   
    augmented_model_path = "./augmented_model.onnx"
    batch_size = 20
    calibration_dataset_size = 1000

    # INT8 calibration setting    
    calibration_table_generation_enable = True # Enable/Disable INT8 calibration

    # TensorRT EP INT8 settings     
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1" # Enable FP16 precision    
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1" # Enable INT8 precision 
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers" # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1" # Enable engine caching
    execution_provider = ["TensorrtExecutionProvider"]
    
    # Convert static batch to dynamic batch
    [new_model_path, input_name] = convert_model_batch_to_dynamic(model_path)

    # Get calibration and prediction dataset size
    [calibration_dataset_size, prediction_dataset_size] = get_dataset_size(ilsvrc2012_dataset_path, calibration_dataset_size)

    # Generate calibration table for INT8 quantization
    if calibration_table_generation_enable:
        get_calibration_table(new_model_path, input_name, batch_size, augmented_model_path, ilsvrc2012_dataset_path, calibration_dataset_size)

    # Generate class id map from imagenet data set
    synset_id = get_synset_id(ilsvrc2012_dataset_path, calibration_dataset_size, prediction_dataset_size)

    # Run prediction on Tensorrt EP    
    get_prediction_evaluation(new_model_path, input_name, batch_size, ilsvrc2012_dataset_path, calibration_dataset_size, prediction_dataset_size, synset_id, providers=execution_provider)
