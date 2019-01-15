# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

.. _l-example-backend-api-tensorflow:

ONNX Runtime for Keras
======================

The following demonstrates how to compute the predictions
of a pretrained deep learning model obtained from 
`keras <https://keras.io/>`_
with *onnxruntime*. The conversion requires
`keras <https://keras.io/>`_,
`tensorflow <https://www.tensorflow.org/>`_,
`sklearn-onnx <https://github.com/onnx/sklearn-onnx/>`_,
`onnxmltools <https://pypi.org/project/onnxmltools/>`_
but then only *onnxruntime* is required
to compute the predictions.
"""
import os
if not os.path.exists('dense121.onnx'):
    from keras.applications.densenet import DenseNet121
    model = DenseNet121(include_top=True, weights='imagenet')

    from onnxmltools import convert_keras
    onx = convert_keras(model, 'dense121.onnx')    
    with open("dense121.onnx", "wb") as f:
        f.write(onx.SerializeToString())

##################################
# Let's load an image (source: wikipedia).

from keras.preprocessing.image import array_to_img, img_to_array, load_img
img = load_img('Sannosawa1.jpg')
ximg = img_to_array(img)

import matplotlib.pyplot as plt
plt.imshow(ximg / 255)
plt.axis('off')

#############################################
# Let's load the model with onnxruntime.
import onnxruntime as rt

sess = rt.InferenceSession('dense121.onnx')

print("The model expects input shape:", sess.get_inputs()[0].shape)
print("image shape:", ximg.shape)

#######################################
# Let's resize the image.
from skimage.transform import resize
import numpy

ximg224 = resize(ximg / 255, (224, 224, 3), anti_aliasing=True)
ximg = ximg224[numpy.newaxis, :, :, :]
ximg = ximg.astype(numpy.float32)

print("new shape:", ximg.shape)

##################################
# Let's compute the output.

input_name = sess.get_inputs()[0].name
res = sess.run(None, {input_name: ximg})
prob = res[0]
print(prob.ravel()[:10])  # Too big to be displayed.


##################################
# Let's get more comprehensive results.

from keras.applications.densenet import decode_predictions
decoded = decode_predictions(prob)

import pandas
df = pandas.DataFrame(decoded[0], columns=["class_id", "name", "P"])
print(df)


