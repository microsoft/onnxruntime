# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Metadata
========

ONNX format contains metadata related to how the
model was produced. It is useful when the model
is deployed to production to keep track of which
instance was used at a specific time.
Let's see how to do that with a simple
logistic regression model trained with
*scikit-learn* and converted with *sklearn-onnx*.
"""

from onnxruntime.datasets import get_example

example = get_example("logreg_iris.onnx")

import onnx  # noqa: E402

model = onnx.load(example)

print(f"doc_string={model.doc_string}")
print(f"domain={model.domain}")
print(f"ir_version={model.ir_version}")
print(f"metadata_props={model.metadata_props}")
print(f"model_version={model.model_version}")
print(f"producer_name={model.producer_name}")
print(f"producer_version={model.producer_version}")

#############################
# With *ONNX Runtime*:

import onnxruntime as rt  # noqa: E402

sess = rt.InferenceSession(example, providers=rt.get_available_providers())
meta = sess.get_modelmeta()

print(f"custom_metadata_map={meta.custom_metadata_map}")
print(f"description={meta.description}")
print(f"domain={meta.domain}")
print(f"graph_name={meta.graph_name}")
print(f"producer_name={meta.producer_name}")
print(f"version={meta.version}")
