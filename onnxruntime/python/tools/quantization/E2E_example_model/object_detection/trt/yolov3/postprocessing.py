import numpy as np
class PostprocessYOLO(object):
    """Class for post-processing the three output tensors from YOLO."""

    def __init__(self,
                 yolo_masks,
                 yolo_anchors,
                 nms_threshold,
                 yolo_input_resolution,
                 category_num=80):
        """Initialize with all values that will be kept when processing
        several frames.  Assuming 3 outputs of the network in the case
        of (large) YOLO, or 2 for the Tiny YOLO.

        Keyword arguments:
        yolo_masks -- a list of 3 (or 2) three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 (or 6) two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_wh -- tuple (W, H) for the target network
        category_num -- number of output categories/classes
        """
        self.masks = yolo_masks
        self.anchors = yolo_anchors
        self.nms_threshold = nms_threshold
        self.input_wh = (yolo_input_resolution[1], yolo_input_resolution[0])
        self.category_num = category_num

    def process(self, outputs, resolution_raw, conf_th):
        """Take the YOLO outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.

        Keyword arguments:
        outputs -- outputs from a TensorRT engine in NCHW format
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        conf_th -- confidence threshold, e.g. 0.3
        """
        outputs_reshaped = list()
        for output in outputs:
            outputs_reshaped.append(self._reshape_output(output))

        boxes_xywh, categories, confidences = self._process_yolo_output(
            outputs_reshaped, resolution_raw, conf_th)

        if len(boxes_xywh) > 0:
            # convert (x, y, width, height) to (x1, y1, x2, y2)
            img_w, img_h = resolution_raw
            xx = boxes_xywh[:, 0].reshape(-1, 1)
            yy = boxes_xywh[:, 1].reshape(-1, 1)
            ww = boxes_xywh[:, 2].reshape(-1, 1)
            hh = boxes_xywh[:, 3].reshape(-1, 1)
            boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., float(img_w-1))
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., float(img_h-1))
            boxes = boxes.astype(np.int)
        else:
            boxes = np.zeros((0, 4), dtype=np.int)  # empty

        return boxes, categories, confidences

    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are CATEGORY_NUM=80 object categories:
        dim4 = (4 + 1 + self.category_num)
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def _process_yolo_output(self, outputs_reshaped, resolution_raw, conf_th):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.

        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (height,width,3,85)
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        conf_th -- confidence threshold
        """

        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence, conf_th)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        width, height = resolution_raw
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32))

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped, mask):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.

        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        def sigmoid_v(array):
            return np.reciprocal(np.exp(-array) + 1.0)

        def exponential_v(array):
            return np.exp(array)

        grid_h, grid_w, _, _ = output_reshaped.shape

        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., 0:2])
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4:5])
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_wh
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        # class confidence
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs, conf_th):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.

        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,CATEGORY_NUM)
        conf_th -- confidence threshold
        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= conf_th)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

class PostprocessYOLOWrapper(object):
    """This class encapsulates things needed to run yolo."""
    """Reference from here https://github.com/jkjung-avt/tensorrt_demos/blob/3fb15c908b155d5edc1bf098c6b8c31886cd8e8d/utils/yolo.py"""

    def _init_yolov3_postprocessor(self):
        h, w = self.input_shape
        filters = (self.category_num + 5) * 3
        if 'tiny' in self.model:
            self.output_shapes = [(1, filters, h // 32, w // 32),
                                  (1, filters, h // 16, w // 16)]
        else:
            self.output_shapes = [(1, filters, h // 32, w // 32),
                                  (1, filters, h // 16, w // 16),
                                  (1, filters, h //  8, w //  8)]
        if 'tiny' in self.model:
            postprocessor_args = {
                # A list of 2 three-dimensional tuples for the Tiny YOLO masks
                'yolo_masks': [(3, 4, 5), (0, 1, 2)],
                # A list of 6 two-dimensional tuples for the Tiny YOLO anchors
                'yolo_anchors': [(10, 14), (23, 27), (37, 58),
                                 (81, 82), (135, 169), (344, 319)],
                # Threshold for non-max suppression algorithm, float
                # value between 0 and 1
                'nms_threshold': 0.5,
                'yolo_input_resolution': self.input_shape,
                'category_num': self.category_num
            }
        else:
            postprocessor_args = {
                # A list of 3 three-dimensional tuples for the YOLO masks
                'yolo_masks': [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                # A list of 9 two-dimensional tuples for the YOLO anchors
                'yolo_anchors': [(10, 13), (16, 30), (33, 23),
                                 (30, 61), (62, 45), (59, 119),
                                 (116, 90), (156, 198), (373, 326)],
                # Threshold for non-max suppression algorithm, float
                # value between 0 and 1
                'nms_threshold': 0.5,
                'yolo_input_resolution': self.input_shape,
                'category_num': self.category_num
            }
        self.postprocessor = PostprocessYOLO(**postprocessor_args)

    def __init__(self, model, input_shape, category_num=80):
        self.model = model
        self.input_shape = input_shape
        self.category_num = category_num
        self.postprocessor = None
        self._init_yolov3_postprocessor()
