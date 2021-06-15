import os
import sys
import numpy as np
import re
from PIL import Image
import cv2
import pdb

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

def yolov3_preprocess_func_2(images_folder, height, width, start_index=0, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''

    # reference from here:
    # https://github.com/jkjung-avt/tensorrt_demos/blob/3fb15c908b155d5edc1bf098c6b8c31886cd8e8d/utils/yolo.py#L60
    def _preprocess_yolo(img, input_shape):
        """Preprocess an image before TRT YOLO inferencing.
        # Args
            img: int8 numpy array of shape (img_h, img_w, 3)
            input_shape: a tuple of (H, W)
        # Returns
            preprocessed img: float32 numpy array of shape (3, H, W)
        """
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img

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
        model_image_size = (height, width)

        img = cv2.imread(image_filepath)
        image_data = _preprocess_yolo(img, tuple(model_image_size)) 
        image_data = np.ascontiguousarray(image_data)
        image_data = np.expand_dims(image_data, 0)
        unconcatenated_batch_data.append(image_data)
        _height, _width, _ = img.shape
        # image_size_list.append(img.shape[0:2])  # img.shape is h, w, c
        image_size_list.append(np.array([img.shape[0], img.shape[1]], dtype=np.float32).reshape(1, 2))

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list

def yolov3_variant_preprocess_func(images_folder, height, width, start_index=0, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''

    # reference from here:
    # https://github.com/jkjung-avt/tensorrt_demos/blob/3fb15c908b155d5edc1bf098c6b8c31886cd8e8d/utils/yolo.py#L60
    def _preprocess_yolo(img, input_shape):
        """Preprocess an image before TRT YOLO inferencing.
        # Args
            img: int8 numpy array of shape (img_h, img_w, 3)
            input_shape: a tuple of (H, W)
        # Returns
            preprocessed img: float32 numpy array of shape (3, H, W)
        """
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img

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
        model_image_size = (height, width)

        img = cv2.imread(image_filepath)
        image_data = _preprocess_yolo(img, tuple(model_image_size))
        image_data = np.ascontiguousarray(image_data)
        image_data = np.expand_dims(image_data, 0)
        unconcatenated_batch_data.append(image_data)
        _height, _width, _ = img.shape
        image_size_list.append(img.shape[0:2])  # img.shape is h, w, c

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list


# This is for special tuned yolov3 model
def yolov3_variant_preprocess_func_2(images_folder, height, width, start_index=0, size_limit=0):
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
        img = letterbox(img0, new_shape=(height, width), auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 1, axis=0)

        img = img.astype('float32') / 255.0

        unconcatenated_batch_data.append(img)
        image_size_list.append(img0.shape[0:2])  # img.shape is h, w, c

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list
