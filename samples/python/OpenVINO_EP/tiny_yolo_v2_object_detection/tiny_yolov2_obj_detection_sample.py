'''
Copyright (C) 2021, Intel Corporation
SPDX-License-Identifier: Apache-2.0
'''

import numpy as np
import onnxruntime as rt
import cv2
import time
import os

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def softmax(x):
  scoreMatExp = np.exp(np.asarray(x))
  return scoreMatExp / scoreMatExp.sum(0)

def checkModelExtension(fp):
  # Split the extension from the path and normalise it to lowercase.
  ext = os.path.splitext(fp)[-1].lower()

  # Now we can simply use != to check for inequality, no need for wildcards.
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def checkVideoFileExtension(fp):
  # Split the extension from the path and normalise it to lowercase.
  ext = os.path.splitext(fp)[-1].lower()
  # Now we can simply use != to check for inequality, no need for wildcards.
  
  if(ext == ".mp4" or ext == ".avi" or ext == ".mov"):
    pass
  else:
    raise Exception(fp, "is an unknown file format. Use the video file ending with .mp4 or .avi or .mov formats")
  
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the video file is Invalid")

# color look up table for different classes for object detection sample
clut = [(0,0,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,128),
        (128,255,0),(128,128,0),(0,128,255),(128,0,128),
        (255,0,128),(128,0,255),(255,128,128),(128,255,128),(255,255,0),
        (255,128,128),(128,128,255),(255,128,128),(128,255,128),(128,255,128)]

# 20 labels that the tiny-yolov2 model can do the object_detection on
label = ["aeroplane","bicycle","bird","boat","bottle",
         "bus","car","cat","chair","cow","diningtable",
         "dog","horse","motorbike","person","pottedplant",
          "sheep","sofa","train","tvmonitor"]

model_file_path = "tiny_yolo_v2_zoo_model.onnx"
# TODO: You need to modify the path to the input onnx model based on where it is located on your device after downloading it from ONNX Model zoo.

# Validate model file path
checkModelExtension(model_file_path)

# Load the model
sess = rt.InferenceSession(model_file_path)

# Get the input name of the model
input_name = sess.get_inputs()[0].name

device = 'CPU_FP32'
# Set OpenVINO as the Execution provider to infer this model
sess.set_providers(['OpenVINOExecutionProvider'], [{'device_type' : device}])
'''
other 'device_type' options are: (Any hardware target can be assigned if you have the access to it)

'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16', 'VAD-F_FP32',
'HETERO:MYRIAD,CPU',  'MULTI:MYRIAD,GPU,CPU'

'''

#Path to video file has to be provided
video_file_path = "sample_demo_video.mp4"
# TODO: You need to specify the path to your own sample video based on where it is located on your device.

#validate video file input path
checkVideoFileExtension(video_file_path)

#Path to video file has to be provided
cap = cv2.VideoCapture(video_file_path)

# capturing different metrics of the image from the video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x_scale = float(width)/416.0  #In the document of tino-yolo-v2, input shape of this network is (1,3,416,416).
y_scale = float(height)/416.0

# writing the inferencing output as a video to the local disk
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_name = device + "_output.avi"
output_video = cv2.VideoWriter(output_video_name,fourcc, float(17.0), (640,360))

# capturing one frame at a time from the video feed and performing the inference
i = 0
while cap.isOpened():
        l_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)
        
        # preprocessing the input frame and reshaping it.
        #In the document of tino-yolo-v2, input shape of this network is (1,3,416,416). so we resize the model frame w.r.t that size.
        in_frame = cv2.resize(frame, (416, 416))
        X = np.asarray(in_frame)
        X = X.astype(np.float32)
        X = X.transpose(2,0,1)
        # Reshaping the input array to align with the input shape of the model
        X = X.reshape(1,3,416,416)
        
        start = time.time()
        #Running the session by passing in the input data of the model
        out = sess.run(None, {input_name: X})
        end = time.time()
        inference_time = end - start
        out = out[0][0]

        numClasses = 20
        anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

        existingLabels = {l: [] for l in label}

        #Inside this loop we compute the bounding box b for grid cell (cy, cx)
        for cy in range(0,13):
         for cx in range(0,13):
          for b in range(0,5):
            # First we read the tx, ty, width(tw), and height(th) for the bounding box from the out array, as well as the confidence score
            channel = b*(numClasses+5)
            tx = out[channel  ][cy][cx]
            ty = out[channel+1][cy][cx]
            tw = out[channel+2][cy][cx]
            th = out[channel+3][cy][cx]
            tc = out[channel+4][cy][cx]

            x = (float(cx) + sigmoid(tx))*32
            y = (float(cy) + sigmoid(ty))*32

            w = np.exp(tw) * 32 * anchors[2*b  ]
            h = np.exp(th) * 32 * anchors[2*b+1] 

            #calculating the confidence score
            confidence = sigmoid(tc) # The confidence value for the bounding box is given by tc

            classes = np.zeros(numClasses)
            for c in range(0,numClasses):
               classes[c] = out[channel + 5 +c][cy][cx]
            # we take the softmax to turn the array into a probability distribution. And then we pick the class with the largest score as the winner.
            classes = softmax(classes)
            detectedClass = classes.argmax()
            
            # Now we can compute the final score for this bounding box and we only want to keep the ones whose combined score is over a certain threshold
            if 0.45< classes[detectedClass]*confidence:
               color =clut[detectedClass]
               x = (x - w/2)*x_scale
               y = (y - h/2)*y_scale
               w *= x_scale
               h *= y_scale
               
               labelX = int((x+x+w)/2)
               labelY = int((y+y+h)/2)
               addLabel = True
               labThreshold = 40
               for point in existingLabels[label[detectedClass]]:
                  if labelX < point[0] + labThreshold and labelX > point[0] - labThreshold and \
                     labelY < point[1] + labThreshold and labelY > point[1] - labThreshold:
                     addLabel = False
               #Adding class labels to the output of the frame and also drawing a rectangular bounding box around the object detected.
               if addLabel:
                  cv2.rectangle(frame, (int(x),int(y)),(int(x+w),int(y+h)),color,2)
                  cv2.rectangle(frame, (int(x),int(y-13)),(int(x)+9*len(label[detectedClass]),int(y)),color,-1)
                  cv2.putText(frame,label[detectedClass],(int(x)+2,int(y)-3),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
                  existingLabels[label[detectedClass]].append((labelX,labelY))
               print('{} detected in frame {}'.format(label[detectedClass],i))
        output_video.write(frame)
        cv2.putText(frame,device,(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,'FPS: {}'.format(1.0/inference_time),(10,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.imshow('frame',frame)

        #Press 'q' to quit the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        print('Processed Frame {}'.format(i))
        i += 1
        l_end = time.time()
        print('Loop Time = {}'.format(l_end - l_start))
output_video.release()
cv2.destroyAllWindows()