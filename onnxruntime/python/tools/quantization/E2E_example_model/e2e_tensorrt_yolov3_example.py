import os
import sys
import numpy as np
from PIL import Image
import cv2
import onnxruntime
from onnxruntime.quantization import get_calibrator, CalibrationDataReader, generate_calibration_table, write_calibration_table

def yolov3_preprocess_func(images_folder, height, width, start_index=0, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    # this function is from yolo3.utils.letterbox_image
    # https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/utils.py
    def letterbox_image(image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    image_names = os.listdir(images_folder)
    if start_index >= len(image_names):
        return np.asanyarray([]), np.asanyarray([]), np.asanyarray([])
    elif size_limit > 0 and len(image_names) >= size_limit:
        end_index = start_index + size_limit
        if end_index > len(image_names):
            end_index = len(image_names)

        batch_filenames = [image_names[i] for i in range(start_index, end_index)]
    else:
        batch_filenames = image_names


    unconcatenated_batch_data = []
    image_size_list = []

    print(batch_filenames)
    print("size: %s" % str(len(batch_filenames)))

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        img = Image.open(image_filepath) 
        model_image_size = (height, width)
        boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        unconcatenated_batch_data.append(image_data)
        image_size_list.append(np.array([img.size[1], img.size[0]], dtype=np.float32).reshape(1, 2))

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list

def yolov3_vision_preprocess_func(images_folder, height, width, start_index=0, size_limit=0):

    def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = new_shape
            ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)



    image_names = os.listdir(images_folder)
    if start_index >= len(image_names):
        return np.asanyarray([]), np.asanyarray([]), np.asanyarray([])
    elif size_limit > 0 and len(image_names) >= size_limit:
        end_index = start_index + size_limit
        if end_index > len(image_names):
            end_index = len(image_names)

        batch_filenames = [image_names[i] for i in range(start_index, end_index)]
    else:
        batch_filenames = image_names


    unconcatenated_batch_data = []
    image_size_list = []

    print(batch_filenames)
    print("size: %s" % str(len(batch_filenames)))

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        img0 = cv2.imread(image_filepath)
        img = letterbox(img0, new_shape=(height, width), auto = False)[0]
        img = img[:,:,::-1].transpose(2,0,1)
        img = np.expand_dims(img, axis = 0)
        img = np.repeat(img, 1, axis = 0)

        img = img.astype('float32')/255.0

        unconcatenated_batch_data.append(img)
        image_size_list.append(img0.shape[0:2]) # img.shape is h, w, c

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list

def parse_annotations(filename):
    import json
    annotations = {}
    with open(filename, 'r') as f:
        annotations = json.load(f)


    img_name_to_img_id = {}
    for image in annotations["images"]:
        file_name = image["file_name"]
        img_name_to_img_id[file_name] = image["id"]

    return img_name_to_img_id

class ObejctDetectionDataReader(CalibrationDataReader):
    def __init__(self, model_path='augmented_model.onnx'):
        self.model_path = model_path
        self.preprocess_flag = None
        self.start_index = 0 
        self.end_index = 0 
        self.stride = 1 
        self.batch_size = 1 
        self.enum_data_dicts = iter([])

    def get_batch_size(self):
        return self.batch_size

class YoloV3DataReader(ObejctDetectionDataReader):
    def __init__(self, calibration_image_folder,
                       width=416,
                       height=416,
                       start_index=0,
                       end_index=0,
                       stride=1,
                       batch_size=1,
                       model_path='augmented_model.onnx',
                       is_evaluation=False,
                       annotations='./annotations/instances_val2017.json'):
        ObejctDetectionDataReader.__init__(self, model_path)
        self.image_folder = calibration_image_folder
        self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.width = width
        self.height = height
        self.start_index = start_index
        self.end_index = len(os.listdir(calibration_image_folder)) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1 # stride must > 0
        self.batch_size = batch_size
        self.is_evaluation = is_evaluation

        self.input_name = 'input_1'
        self.img_name_to_img_id = parse_annotations(annotations)

    def get_dataset_size(self):
        return len(os.listdir(self.image_folder))

    def get_input_name(self):
        if self.input_name:
            return
        session = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.start_index < self.end_index:
            if self.batch_size == 1:
                data = self.load_serial()
            else:
                data = self.load_batches()

            self.start_index += self.stride
            self.enum_data_dicts = iter(data)

            return next(self.enum_data_dicts, None)
        else:
            return None

    def load_serial(self):
        width = self.width 
        height = self.width 
        nchw_data_list, filename_list, image_size_list = yolov3_preprocess_func(self.image_folder, height, width, self.start_index, self.stride)
        input_name = self.input_name

        print("Start from index %s ..." % (str(self.start_index)))
        data = []
        if self.is_evaluation:
            img_name_to_img_id = self.img_name_to_img_id 
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_shape": image_size_list[i], "image_id": img_name_to_img_id[file_name]})

        else:
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
        return data

    def load_batches(self):
        width = self.width 
        height = self.height 
        batch_size = self.batch_size
        stride = self.stride
        input_name = self.input_name

        for index in range(0, stride, batch_size):
            start_index = self.start_index + index 
            print("Load batch from index %s ..." % (str(start_index)))
            nchw_data_list, filename_list, image_size_list = yolov3_preprocess_func(self.image_folder, height, width, start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            image_id_batch = []
            batches = []
            if self.is_evaluation:
                img_name_to_img_id = self.img_name_to_img_id 
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                    img_name = filename_list[i]
                    image_id = img_name_to_img_id[img_name]
                    image_id_batch.append(image_id)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                batch_id = np.concatenate(np.expand_dims(image_id_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data, "image_id": batch_id, "image_shape": np.asarray([[416, 416]], dtype=np.float32)}
            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data, "image_shape": np.asarray([[416, 416]], dtype=np.float32)}

            batches.append(data)

        return batches


class YoloV3VisionDataReader(YoloV3DataReader):
    def __init__(self, calibration_image_folder,
                       width=608,
                       height=384,
                       start_index=0,
                       end_index=0,
                       stride=1,
                       batch_size=1,
                       model_path='augmented_model.onnx',
                       is_evaluation=False,
                       annotations='./annotations/instances_val2017.json'):
        YoloV3DataReader.__init__(self, calibration_image_folder, width, height, start_index, end_index, stride, batch_size, model_path, is_evaluation, annotations)
        self.input_name = 'images'

    def load_serial(self):
        width = self.width 
        height = self.height 
        input_name = self.input_name
        nchw_data_list, filename_list, image_size_list = yolov3_vision_preprocess_func(self.image_folder, height, width, self.start_index, self.stride)

        data = []
        if self.is_evaluation:
            img_name_to_img_id = self.img_name_to_img_id
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data, "image_id": img_name_to_img_id[file_name], "image_size": image_size_list[i]})

        else:
            for i in range(len(nchw_data_list)):
                nhwc_data = nchw_data_list[i]
                file_name = filename_list[i]
                data.append({input_name: nhwc_data})
        return data

    def load_batches(self):
        width = self.width 
        height = self.height 
        stride = self.stride
        batch_size = self.batch_size
        input_name = self.input_name

        batches = []
        for index in range(0, stride, batch_size):
            start_index = self.start_index + index 
            print("Load batch from index %s ..." % (str(start_index)))
            nchw_data_list, filename_list, image_size_list = yolov3_vision_preprocess_func(self.image_folder, height, width, start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            image_id_batch = []
            if self.is_evaluation:
                img_name_to_img_id = self.img_name_to_img_id
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                    img_name = filename_list[i]
                    image_id = img_name_to_img_id[img_name]
                    image_id_batch.append(image_id)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                batch_id = np.concatenate(np.expand_dims(image_id_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data, "image_size": image_size_list, "image_id": batch_id}
            else:
                for i in range(len(nchw_data_list)):
                    nhwc_data = np.squeeze(nchw_data_list[i], 0)
                    nchw_data_batch.append(nhwc_data)
                batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
                print(batch_data.shape)
                data = {input_name: batch_data}

            batches.append(data)

        return batches

class YoloV3Evaluator: 
    def __init__(self, model_path,
                       data_reader: CalibrationDataReader,
                       width=416,
                       height=416,
                       providers=["CUDAExecutionProvider"],
                       ground_truth_object_class_file="./coco-object-categories-2017.json",
                       onnx_object_class_file="./onnx_coco_classes.txt"):
        '''
        :param model_path: ONNX model to validate 
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.width = width
        self.height = height
        self.providers = providers 
        self.class_to_id = {} # object class -> id
        self.onnx_class_list = []
        self.prediction_result_list = []
        self.identical_class_map = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv"}

        f = open(onnx_object_class_file, 'r')
        lines = f.readlines()
        for c in lines:
            self.onnx_class_list.append(c.strip('\n'))

        self.generate_class_to_id(ground_truth_object_class_file)
        print(self.class_to_id)


    def generate_class_to_id(self, ground_truth_object_class_file):
        with open(ground_truth_object_class_file) as f:
            import json
            classes = json.load(f)

        for c in classes:
            self.class_to_id[c["name"]] = c["id"]

    def set_data_reader(self, data_reader):
        self.data_reader = data_reader

    def get_result(self):
        return self.prediction_result_list

    def set_bbox_prediction(self, boxes, scores, indices, is_batch, image_id, image_id_batch):
        out_boxes, out_scores, out_classes, out_batch_index = [], [], [], []

        for idx_ in indices:
            out_classes.append(idx_[1])
            out_batch_index.append(idx_[0])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])

        for i in range(len(out_classes)):
            out_class = out_classes[i]
            class_name = self.onnx_class_list[int(out_class)]
            if class_name in self.identical_class_map:
                class_name = self.identical_class_map[class_name]
            id = self.class_to_id[class_name]

            bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
            bbox_yxhw = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3]-out_boxes[i][1], out_boxes[i][2]-out_boxes[i][0]]
            bbox_yxhw_str = [str(out_boxes[i][1]), str(out_boxes[i][0]), str(out_boxes[i][3]-out_boxes[i][1]), str(out_boxes[i][2]-out_boxes[i][0])]
            score = str(out_scores[i])
            coor = np.array(bbox[:4], dtype=np.int32)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
           
            if is_batch:
                image_id = image_id_batch[out_batch_index[i]] 
            self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":bbox_yxhw, "score":out_scores[i]})

    def predict(self):
        session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)

        outputs = []

        # If you decide to run batch inference, please make sure all input images must be re-sized to the same shape.
        # Which means the bounding boxes from groun truth annotation must to be adjusted accordingly, otherwise you will get very low mAP results. 
        # Here we simply choose to run serial inference.
        if self.data_reader.get_batch_size() > 1:
            # batch inference
            print("Doing batch inference...")

            image_id_list = []
            image_id_batch = []
            while True:
                inputs = self.data_reader.get_next()
                if not inputs:
                    break
                image_id_list = inputs["image_id"]
                del inputs["image_id"]
                image_id_batch.append(image_id_list)
                outputs.append(session.run(None, inputs))

                for index in range(len(outputs)):
                    output = outputs[index]
                    boxes = output[0]
                    scores = output[1]
                    indices = output[2]

                    self.set_bbox_prediction(boxes, scores, indices, True, None, image_id_batch[index])
        else:
            # serial inference
            while True:
                inputs = self.data_reader.get_next()
                if not inputs:
                    break

                image_id = inputs["image_id"]
                del inputs["image_id"]

                output = session.run(None, inputs)

                boxes = output[0]
                scores = output[1]
                indices = output[2]

                self.set_bbox_prediction(boxes, scores, indices, False, image_id, None)

    def evaluate(self, prediction_result, annotations):
        # calling coco api
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import numpy as np
        import skimage.io as io
        import pylab
        pylab.rcParams['figure.figsize'] = (10.0, 8.0)


        annType = ['segm','bbox','keypoints']
        annType = annType[1]      #specify type here
        prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
        print('Running evaluation for *%s* results.'%(annType))

        annFile = annotations
        cocoGt=COCO(annFile)

        resFile = prediction_result 
        cocoDt=cocoGt.loadRes(resFile)

        imgIds=sorted(cocoGt.getImgIds())
        imgIds=imgIds[0:100]
        imgId = imgIds[np.random.randint(100)]


        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

class YoloV3VisionEvaluator(YoloV3Evaluator): 
    def __init__(self, model_path,
                       data_reader: CalibrationDataReader,
                       width=608,
                       height=384,
                       providers=["CUDAExecutionProvider"],
                       ground_truth_object_class_file="./coco-object-categories-2017.json",
                       onnx_object_class_file="./onnx_coco_classes.txt"):

        YoloV3Evaluator.__init__(self, model_path, data_reader,width, height, providers, ground_truth_object_class_file, onnx_object_class_file)

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            # gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1]/img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[[0, 2]] -= pad[0]  # x padding
        coords[[1, 3]] -= pad[1]  # y padding
        coords[:4] /= gain
        return coords

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[0] = x[0] - x[2] / 2  # top left x
        y[1] = x[1] - x[3] / 2  # top left y
        y[2] = x[0] + x[2] / 2  # bottom right x
        y[3] = x[1] + x[3] / 2  # bottom right y
        return y

    def set_bbox_prediction(self, bboxes, scores, image_height, image_width, image_id):

        for i in range(bboxes.shape[0]): 
            bbox = bboxes[i]
            bbox[0] *= self.width #x
            bbox[1] *= self.height #y
            bbox[2] *= self.width #w
            bbox[3] *= self.height #h
            
            img0_shape = (image_height, image_width)
            img1_shape = (self.height, self.width)
            bbox = self.xywh2xyxy(bbox)
            bbox = self.scale_coords(img1_shape, bbox, img0_shape)

            class_name = 'person' 
            if class_name in self.identical_class_map:
                class_name = self.identical_class_map[class_name]
            id = self.class_to_id[class_name]

            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]

            self.prediction_result_list.append({"image_id":int(image_id), "category_id":int(id), "bbox":list(bbox), "score":scores[i][0]})

    def predict(self):
        session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
        outputs = []

        image_id_list = []
        image_id_batch = []
        image_size_list = []
        image_size_batch = []
        while True:
            inputs = self.data_reader.get_next()
            if not inputs:
                break
            image_size_list = inputs["image_size"]
            image_id_list = inputs["image_id"]
            del inputs["image_size"]
            del inputs["image_id"]

            # in the case of batch size is 1
            if type(image_id_list) == int:
                image_size_list = [image_size_list]
                image_id_list = [image_id_list]


            image_size_batch.append(image_size_list)
            image_id_batch.append(image_id_list)
            outputs.append(session.run(None, inputs))

        for i in range(len(outputs)):
            output = outputs[i]
            for batch_i in range(self.data_reader.get_batch_size()):
                batch_idx = output[0][:,0] == batch_i
                bboxes = output[1][batch_idx,:]
                scores = output[2][batch_idx,:]

                if batch_i > len(image_size_batch[i])-1 or batch_i > len(image_id_batch[i])-1:
                    continue

                image_height = image_size_batch[i][batch_i][0]
                image_width= image_size_batch[i][batch_i][1]
                image_id = image_id_batch[i][batch_i]
                self.set_bbox_prediction(bboxes, scores, image_height, image_width, image_id)

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
    '''
    TensorRT EP INT8 Inference on Yolov3 model

    This script is using COCO dataset for calibration and evaluation.
    Please go to dataset download page https://cocodataset.org/#download and save 2017 Test images and 2017 Val images respectively.
    
    Besides classic Yolov3, this example also tackles Yolov3 variants which take differnt image dimension as model input.
    These variants only focus on detecting people, therefore the dataset should be filtered to contain only person in order for better evaluation. 
    You can reference this repo to do filtering.(https://github.com/immersive-limit/coco-manager)

    '''

    model_path = 'yolov3_new.onnx'
    # model_path = 'yolov3_288x512_batch_nms.onnx'
    # model_path = 'yolov3_384x608_batch_nms.onnx'

    augmented_model_path = 'augmented_model.onnx'

    calibration_dataset = './test2017'

    validation_dataset = './val2017'
    # validation_dataset = './val2017person'

    get_calibration_table(model_path, augmented_model_path, calibration_dataset)
    get_prediction_evaluation(model_path, validation_dataset, ["TensorrtExecutionProvider"])
