# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Train, convert and predict with ONNX Runtime
============================================

This example demonstrates an end to end scenario
starting with the training of a scikit-learn pipeline
which takes as inputs not a regular vector but a
dictionary ``{ int: float }`` as its first step is a
`DictVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`_.

Train a pipeline
++++++++++++++++

The first step consists in creating a dummy datasets.
"""

import pandas
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(1000, n_targets=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_dict = pandas.DataFrame(X_train[:, 1:]).T.to_dict().values()
X_test_dict = pandas.DataFrame(X_test[:, 1:]).T.to_dict().values()

####################################
# We create a pipeline.

from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.feature_extraction import DictVectorizer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

pipe = make_pipeline(DictVectorizer(sparse=False), GradientBoostingRegressor())

pipe.fit(X_train_dict, y_train)

####################################
# We compute the prediction on the test set
# and we show the confusion matrix.
from sklearn.metrics import r2_score  # noqa: E402

pred = pipe.predict(X_test_dict)
print(r2_score(y_test, pred))

####################################
# Conversion to ONNX format
# +++++++++++++++++++++++++
#
# We use module
# `sklearn-onnx <https://github.com/onnx/sklearn-onnx>`_
# to convert the model into ONNX format.

from skl2onnx import convert_sklearn  # noqa: E402
from skl2onnx.common.data_types import DictionaryType, FloatTensorType, Int64TensorType  # noqa: E402

# initial_type = [('float_input', DictionaryType(Int64TensorType([1]), FloatTensorType([])))]
initial_type = [("float_input", DictionaryType(Int64TensorType([1]), FloatTensorType([])))]
onx = convert_sklearn(pipe, initial_types=initial_type, target_opset=17)
with open("pipeline_vectorize.onnx", "wb") as f:
    f.write(onx.SerializeToString())

##################################
# We load the model with ONNX Runtime and look at
# its input and output.
import onnxruntime as rt  # noqa: E402
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument  # noqa: E402

sess = rt.InferenceSession("pipeline_vectorize.onnx", providers=rt.get_available_providers())

inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
print(f"input name='{inp.name}' and shape={inp.shape} and type={inp.type}")
print(f"output name='{out.name}' and shape={out.shape} and type={out.type}")

##################################
# We compute the predictions.
# We could do that in one call:

try:
    sess.run([out.name], {inp.name: X_test_dict})[0]
except (RuntimeError, InvalidArgument) as e:
    print(e)

#############################
# But it fails because, in case of a DictVectorizer,
# ONNX Runtime expects one observation at a time.
pred_onx = [sess.run([out.name], {inp.name: row})[0][0, 0] for row in X_test_dict]

###############################
# We compare them to the model's ones.
print(r2_score(pred, pred_onx))

#########################
# Very similar. *ONNX Runtime* uses floats instead of doubles,
# that explains the small discrepencies.
