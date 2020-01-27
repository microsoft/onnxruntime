
========
Tutorial
========

*ONNX Runtime* provides an easy way to run
machine learned models with high performance on CPU or GPU
without dependencies on the training framework.
Machine learning frameworks are usually optimized for
batch training rather than for prediction, which is a
more common scenario in applications, sites, and services.
At a high level, you can:

1. Train a model using your favorite framework.
2. Convert or export the model into ONNX format.
   See `ONNX Tutorials <https://github.com/onnx/tutorials>`_
   for more details.
3. Load and run the model using *ONNX Runtime*.

In this tutorial, we will briefly create a 
pipeline with *scikit-learn*, convert it into
ONNX format and run the first predictions.

.. _l-logreg-example:

Step 1: Train a model using your favorite framework
+++++++++++++++++++++++++++++++++++++++++++++++++++

We'll use the famous iris datasets.

.. runpython::
    :showcode:
    :store:
    :warningout: ImportWarning FutureWarning

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    from sklearn.linear_model import LogisticRegression
    clr = LogisticRegression()
    clr.fit(X_train, y_train)
    print(clr)

Step 2: Convert or export the model into ONNX format
++++++++++++++++++++++++++++++++++++++++++++++++++++

`ONNX <https://github.com/onnx/onnx>`_ is a format to describe
the machine learned model.
It defines a set of commonly used operators to compose models.
There are `tools <https://github.com/onnx/tutorials>`_
to convert other model formats into ONNX. Here we will use
`ONNXMLTools <https://github.com/onnx/onnxmltools>`_.

.. runpython::
    :showcode:
    :restore:
    :store:
    :warningout: ImportWarning FutureWarning

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    with open("logreg_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())

Step 3: Load and run the model using ONNX Runtime
+++++++++++++++++++++++++++++++++++++++++++++++++

We will use *ONNX Runtime* to compute the predictions 
for this machine learning model.

.. runpython::
    :showcode:
    :restore:
    :store:

    import numpy
    import onnxruntime as rt

    sess = rt.InferenceSession("logreg_iris.onnx")
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})[0]
    print(pred_onx)

The code can be changed to get one specific output
by specifying its name into a list.

.. runpython::
    :showcode:
    :restore:

    import numpy
    import onnxruntime as rt

    sess = rt.InferenceSession("logreg_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
    print(pred_onx)


