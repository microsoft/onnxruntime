import os
from onnxruntime.quantization import BertDataReader, generate_calibration_table

def get_calibration_table(model_path, augmented_model_path):
    data_reader = BertDataReader(model_path)
    generate_calibration_table(model_path, augmented_model_path, data_reader)

def get_prediction_evaluation(model_path):
    data_reader = BertDataReader(model_path)
    data_reader.evaluate()


if __name__ == '__main__':

    model_path = 'bert.shape.onnx'
    augmented_model_path = 'augmented_bert.shape.onnx'

    get_calibration_table(model_path, augmented_model_path)
    get_prediction_evaluation(model_path)


