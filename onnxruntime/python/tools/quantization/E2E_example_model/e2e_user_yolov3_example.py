import os
from onnxruntime.quantization import get_calibrator, YoloV3DataReader, YoloV3VisionDataReader, YoloV3Evaluator, YoloV3VisionEvaluator, generate_calibration_table
from dataset_utils import *

def get_prediction_evaluation(model_path, validation_dataset, providers):
    # Some machines don't have sufficient memory to hold all dataset at once. So handle it by batch/stride.
    # For each stride, data_reader can handle them with batch or serial processing depends on data reader implementation 
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
    # Some machines don't have sufficient memory to hold all dataset at once. So handle it by batch/stride.
    # For each stride, data_reader can handle them with batch or serial processing depends on data reader implementation 
    data_reader = YoloV3DataReader(calibration_dataset, stride=1000, batch_size=1, model_path=augmented_model_path)

    # data_reader = YoloV3VisionDataReader(calibration_dataset, width=512, height=288, stride=1000, batch_size=20, model_path=augmented_model_path)
    # data_reader = YoloV3VisionDataReader(calibration_dataset, width=608, height=384, stride=1000, batch_size=20, model_path=augmented_model_path)

    generate_calibration_table(model_path, augmented_model_path, data_reader)


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
