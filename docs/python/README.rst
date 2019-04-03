ONNX Runtime
============

ONNX Runtime
enables high-performance evaluation of trained machine learning (ML)
models while keeping resource usage low.
Building on Microsoft's dedication to the
`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
community, it supports traditional ML models as well
as Deep Learning algorithms in the
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_.
Documentation is available at
`Python Bindings for ONNX Runtime <https://aka.ms/onnxruntime-python>`_.

Example
-------

The following example demonstrates an end-to-end example
in a very common scenario. A model is trained with *scikit-learn*
but it has to run very fast in a optimized environment.
The model is then converted into ONNX format and ONNX Runtime
replaces *scikit-learn* to compute the predictions.

::

    # Train a model.
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)

    # Convert into ONNX format with onnxmltools
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(clr, initial_types=initial_type)
    with open("rf_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # Compute the prediction with ONNX Runtime
    import onnxruntime as rt
    import numpy
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

Changes
-------

0.3.1
^^^^^

Protobuf-lite, NuGet file fixes (patch to 0.3.0).

0.3.0
^^^^^

C-API, Linux support for Dotnet Nuget package, Cuda 9.1 support.

0.2.1
^^^^^

C-API, Linux support for Dotnet Nuget package, Cuda 10.0 support (patch to 0.2.0).

0.2.0
^^^^^

C-API, Linux support for Dotnet Nuget package, Cuda 10.0 support

0.1.5
^^^^^

GA release as part of open sourcing onnxruntime (patch to 0.1.4).

0.1.4
^^^^^

GA release as part of open sourcing onnxruntime.

0.1.3
^^^^^

Fixes a crash on machines which do not support AVX instructions.

0.1.2
^^^^^

First release on Ubuntu 16.04 for CPU and GPU with Cuda 9.1 and Cudnn 7.0,
supports runtime for deep learning models architecture such as AlexNet, ResNet,
XCeption, VGG, Inception, DenseNet, standard linear learner,
standard ensemble learners,
and transform scaler, imputer.
