import os
from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibrationMethod
from data_reader import YoloV3DataReader, YoloV3VariantDataReader
from preprocessing import yolov3_preprocess_func, yolov3_preprocess_func_2, yolov3_variant_preprocess_func, yolov3_variant_preprocess_func_2
from evaluate import YoloV3Evaluator, YoloV3VariantEvaluator,YoloV3Variant2Evaluator, YoloV3Variant3Evaluator


def get_calibration_table(model_path, augmented_model_path, calibration_dataset):

    calibrator = create_calibrator(model_path, None, augmented_model_path=augmented_model_path)

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
    stride = 2000
    for i in range(0, total_data_size, stride):
        data_reader = YoloV3DataReader(calibration_dataset,
                                       start_index=start_index,
                                       end_index=start_index + stride,
                                       stride=stride,
                                       batch_size=1,
                                       model_path=augmented_model_path)
        calibrator.collect_data(data_reader)
        start_index += stride
    '''
    2. Use batch processing (much faster)
    
    Batch processing requires less memory for intermediate output, therefore let only one DataReader to handle dataset in batch. 
    However, if encountering OOM, we can make multiple DataReader to do the job just like serial processing does. 
    DataReader will use batch processing when batch_size > 1.
    '''

    # data_reader = YoloV3DataReader(calibration_dataset, stride=1000, batch_size=20, model_path=augmented_model_path)
    # calibrator.collect_data(data_reader)

    write_calibration_table(calibrator.compute_range())
    print('calibration table generated and saved.')


def get_prediction_evaluation(model_path, validation_dataset, providers):
    data_reader = YoloV3DataReader(validation_dataset,
                                   stride=1000,
                                   batch_size=1,
                                   model_path=model_path,
                                   is_evaluation=True)
    evaluator = YoloV3Evaluator(model_path, data_reader, providers=providers)

    evaluator.predict()
    result = evaluator.get_result()

    annotations = './annotations/instances_val2017.json'
    evaluator.evaluate(result, annotations)


def get_calibration_table_yolov3_variant(model_path, augmented_model_path, calibration_dataset):

    calibrator = create_calibrator(model_path, [], augmented_model_path=augmented_model_path, calibrate_method=CalibrationMethod.Entropy)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])

    # DataReader can handle dataset with batch or serial processing depends on its implementation
    # Following examples show two different ways to generate calibration table
    '''
    1. Use serial processing
    
    We can use only one data reader to do serial processing, however,
    some machines don't have sufficient memory to hold all dataset images and all intermediate output.
    So let multiple data readers to handle different stride of dataset one by one.
    DataReader will use serial processing when batch_size is 1.
    '''

    width = 608
    height = 608

    total_data_size = len(os.listdir(calibration_dataset))
    start_index = 0
    stride = 20 
    batch_size = 1
    for i in range(0, total_data_size, stride):
        data_reader = YoloV3VariantDataReader(calibration_dataset,
                                              width=width,
                                              height=height,
                                              start_index=start_index,
                                              end_index=start_index + stride,
                                              stride=stride,
                                              batch_size=batch_size,
                                              model_path=augmented_model_path)
        calibrator.collect_data(data_reader)
        start_index += stride
    '''
    2. Use batch processing (much faster)
    
    Batch processing requires less memory for intermediate output, therefore let only one data reader to handle dataset in batch. 
    However, if encountering OOM, we can make multiple data reader to do the job just like serial processing does. 
    DataReader will use batch processing when batch_size > 1.
    '''

    # batch_size = 20
    # stride=1000
    # data_reader = YoloV3VariantDataReader(calibration_dataset,
                                          # width=width,
                                          # height=height,
                                          # stride=stride,
                                          # batch_size=batch_size,
                                          # model_path=augmented_model_path)
    # calibrator.collect_data(data_reader)

    write_calibration_table(calibrator.compute_range())
    print('calibration table generated and saved.')


def get_prediction_evaluation_yolov3_variant(model_path, validation_dataset, providers):
    width = 608 
    height = 608 
    evaluator = YoloV3VariantEvaluator(model_path, None, width=width, height=height, providers=providers)

    total_data_size = len(os.listdir(validation_dataset)) 
    start_index = 0
    stride=1000
    batch_size = 1
    for i in range(0, total_data_size, stride):
        data_reader = YoloV3VariantDataReader(validation_dataset,
                                              width=width,
                                              height=height,
                                              start_index=start_index,
                                              end_index=start_index+stride,
                                              stride=stride,
                                              batch_size=batch_size,
                                              model_path=model_path,
                                              is_evaluation=True)

        evaluator.set_data_reader(data_reader)
        evaluator.predict()
        start_index += stride


    result = evaluator.get_result()
    annotations = './annotations/instances_val2017.json'
    evaluator.evaluate(result, annotations)


if __name__ == '__main__':
    '''
    TensorRT EP INT8 Inference on Yolov3 model.

    The script is using subset of COCO 2017 Train images as calibration and COCO 2017 Val images as evaluation.
    1. Please create workspace folders 'train2017/calib' and 'val2017'.
    2. Download 2017 Val dataset: http://images.cocodataset.org/zips/val2017.zip
    3. Download 2017 Val and Train annotations from http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    4. Run following script to download subset of COCO 2017 Train images and save them to 'train2017/calib':
        python3 coco_filter.py -i annotations/instances_train2017.json -f train2017 -c all 

        (Reference and modify from https://github.com/immersive-limit/coco-manager)
    5. Download Yolov3 model:
        (i) ONNX model zoo yolov3: https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx 
        (ii) yolov3 variants: https://github.com/jkjung-avt/tensorrt_demos.git
    '''

    augmented_model_path = 'augmented_model.onnx'
    calibration_dataset = './train2017/calib'
    validation_dataset = './val2017'
    is_onnx_model_zoo_yolov3 = False 

    # TensorRT EP INT8 settings
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers"  # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
    execution_provider = ["TensorrtExecutionProvider"]

    if is_onnx_model_zoo_yolov3:
        model_path = 'yolov3.onnx'
        get_calibration_table(model_path, augmented_model_path, calibration_dataset)
        get_prediction_evaluation(model_path, validation_dataset, execution_provider)
    else:
        # Yolov3 variants from here
        # https://github.com/jkjung-avt/tensorrt_demos.git
        model_path = 'yolov3-608.onnx'
        get_calibration_table_yolov3_variant(model_path, augmented_model_path, calibration_dataset)
        get_prediction_evaluation_yolov3_variant(model_path, validation_dataset, execution_provider)
