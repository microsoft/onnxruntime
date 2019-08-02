
Python Bindings for ONNX Runtime
================================

ONNX Runtime enables high-performance evaluation of trained machine learning (ML)
models while keeping resource usage low.
Building on Microsoft's dedication to the
`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
community, it supports traditional ML models as well
as Deep Learning algorithms in the
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_.

.. toctree::
    :maxdepth: 1

    tutorial
    api_summary
    auto_examples/index

:ref:`genindex`

The core library is implemented in C++.
*ONNX Runtime* is available on
PyPi for Linux Ubuntu 16.04, Python 3.5+ for both
`CPU <https://pypi.org/project/onnxruntime/>`_ and
`GPU <https://pypi.org/project/onnxruntime-gpu/>`_.
Please see `system requirements <https://github.com/Microsoft/onnxruntime#system-requirements>`_ before installating the packages.
This example demonstrates a simple prediction for an
`ONNX-ML format <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_
model. The following file ``model.onnx`` is taken from
github `onnx...test_sigmoid <https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node/test_sigmoid>`_.

.. runpython::
    :showcode:

    import numpy
    import onnxruntime as rt
    sess = rt.InferenceSession("model.onnx")
    input_name = sess.get_inputs()[0].name
    X = numpy.random.random((3,4,5)).astype(numpy.float32)
    pred_onnx = sess.run(None, {input_name: X})
    print(pred_onnx)
