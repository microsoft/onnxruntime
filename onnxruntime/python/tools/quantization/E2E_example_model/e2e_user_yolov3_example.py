import os
from onnxruntime.quantization import get_calibrator, YoloV3DataReader, YoloV3VisionDataReader, YoloV3Evaluator, YoloV3VisionEvaluator, generate_calibration_table, write_calibration_table
from dataset_utils import *

def get_prediction_evaluation(model_path, validation_dataset, providers):
    data_reader = YoloV3DataReader(validation_dataset, stride=1000, batch_size=1, model_path=model_path, is_evaluation=True)
    evaluator = YoloV3Evaluator(model_path, data_reader, providers=providers)

    # data_reader = YoloV3VisionDataReader(validation_dataset, width=608, height=384, stride=1000, batch_size=1, model_path=model_path, is_evaluation=True)
    # evaluator = YoloV3VisionEvaluator(model_path, data_reader, width=608, height=384, providers=providers)

    evaluator.predict()
    result = evaluator.get_result()

    annotations = './annotations/instances_val2017.json'
    # annotations = './annotations/instances_val2017_person.json'
    print(result)
    evaluator.evaluate(result, annotations)


def get_calibration_table(model_path, augmented_model_path, calibration_dataset):

    calibrator = get_calibrator(model_path, None, augmented_model_path=augmented_model_path)
    
    # DataReader can handle dataset with batch or serial processing depends on its implementation 
    # Following examples show two different ways to generate calibration table 

    '''
    1. Use serial processing
    
    We can use only one DataReader to do serial processing, however,
    some machines don't have sufficient memory to hold all dataset images and all intermediate output.
    So let multiple DataReader do handle different stride of dataset one by one.
    DataReader will use serial processing when batch_size is 1.
    '''

    total_data_size = len(os.listdir(calibration_dataset)) 
    start_index = 0
    stride=2000
    for i in range(0, total_data_size, stride):
        data_reader = YoloV3DataReader(calibration_dataset,start_index=start_index, end_index=start_index+stride, stride=stride, batch_size=1, model_path=augmented_model_path)
        calibrator.set_data_reader(data_reader)
        generate_calibration_table(calibrator, model_path, augmented_model_path, False, data_reader)
        start_index += stride


    '''
    2. Use batch processing (much faster)
    
    Batch processing requires less memory for intermediate output, therefore let only one DataReader to handle dataset in batch. 
    However, if encountering OOM, we can make multiple DataReader to do the job just like serial processing does. 
    DataReader will use batch processing when batch_size > 1.
    '''

    # data_reader = YoloV3DataReader(calibration_dataset, stride=1000, batch_size=20, model_path=augmented_model_path)
    # data_reader = YoloV3VisionDataReader(calibration_dataset, width=512, height=288, stride=1000, batch_size=20, model_path=augmented_model_path)
    # data_reader = YoloV3VisionDataReader(calibration_dataset, width=608, height=384, stride=1000, batch_size=20, model_path=augmented_model_path)
    # calibrator.set_data_reader(data_reader)
    # generate_calibration_table(calibrator, model_path, augmented_model_path, True, data_reader)

    write_calibration_table(calibrator.get_calibration_cache())
    print('calibration table generated and saved.')

if __name__ == '__main__':

    model_path = 'yolov3_new.onnx'
    # model_path = 'yolov3_288x512_batch_nms.onnx'
    # model_path = 'yolov3_384x608_batch_nms.onnx'

    augmented_model_path = 'augmented_model.onnx'

    calibration_dataset = './test2017'

    validation_dataset = './val2017'
    # validation_dataset = './val2017person'

    get_calibration_table(model_path, augmented_model_path, calibration_dataset)
    get_prediction_evaluation(model_path, validation_dataset, ["TensorrtExecutionProvider"])
