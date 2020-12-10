import os
from onnxruntime.quantization import get_calibrator, YoloV3DataReader, YoloV3VisionDataReader, YoloV3Evaluator, YoloV3VisionEvaluator, generate_calibration_table
from dataset_utils import *

def get_prediction_evaluation(model_path, validation_dataset, providers):
    image_list = os.listdir(validation_dataset)
    stride = 1000 
    
    evaluator = None
    results = []

    # Some machines don't have sufficient memory to hold all dataset at once. So handle it by batch/stride.
    # For each stride, data_reader can handle them with batch or serial processing depends on data reader implementation 
    for i in range(0, len(image_list), stride):
        print("Total %s images\nStart to process from %s with stride %s ..." % (str(len(image_list)), str(i), str(stride)))
        dr = YoloV3DataReader(validation_dataset, model_path=model_path, start_index=i, size_limit=stride, batch_size=20, is_evaluation=True)
        evaluator = YoloV3Evaluator(model_path, dr, providers=providers)

        # dr = YoloV3VisionDataReader(validation_dataset, width=512, height=288, model_path=model_path, start_index=i, size_limit=stride, batch_size=20, is_evaluation=True)
        # evaluator = YoloV3VisionEvaluator(model_path, dr, width=512, height=288, providers=providers)

        # dr = YoloV3VisionDataReader(validation_dataset, width=608, height=384, model_path=model_path, start_index=i, size_limit=stride, batch_size=20, is_evaluation=True)
        # evaluator = YoloV3VisionEvaluator(model_path, dr, width=608, height=384, providers=providers)

        evaluator.predict()
        results += evaluator.get_result()

    print("Total %s bounding boxes." % (len(results)))
        
    if evaluator:
        annotations = './annotations/instances_val2017.json'
        # annotations = './annotations/instances_val2017_person.json'
        print(results)
        evaluator.evaluate(results, annotations)


def get_calibration_table(model_path, augmented_model_path, calibration_dataset):
    data_reader = YoloV3DataReader(calibration_dataset, model_path=augmented_model_path)
    # data_reader = YoloV3VisionDataReader(calibration_dataset, width=512, height=288, model_path=augmented_model_path)
    # data_reader = YoloV3VisionDataReader(calibration_dataset, width=608, height=384, model_path=augmented_model_path)

    generate_calibration_table(model_path, augmented_model_path, data_reader, calibration_dataset=calibration_dataset, stride=1000, batch_size=20)


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
