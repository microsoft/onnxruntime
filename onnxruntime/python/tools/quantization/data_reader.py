from onnxruntime.quantization import CalibrationDataReader 
from .processing import yolov3_preprocess_func
import onnxruntime

class YoloV3OnnxModelZooDataReader(CalibrationDataReader):
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
