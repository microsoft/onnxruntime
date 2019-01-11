

.. only:: md

    Gallery of examples
    ===================
    
    The first series of examples briefly goes into the main
    feature *ONNX Runtime* implements. Each of them run in a 
    few seconds and relies on machine learned models
    trained with `scikit-learn <http://scikit-learn.org/stable/>`_.
    
    .. toctree::
        :maxdepth: 1
        :caption: Contents:
        
        auto_examples/plot_load_and_predict
        auto_examples/plot_common_errors
        auto_examples/plot_train_convert_predict
        auto_examples/plot_pipeline
        auto_examples/plot_backend
        auto_examples/plot_convert_pipeline_vectorizer
        auto_examples/plot_metadata
        auto_examples/plot_profiling

    The second series is about deep learning.
    Once converted to *ONNX*, the predictions can be
    computed with *onnxruntime* without having any
    dependencies on the framework used to train the model.

    .. toctree::
        :maxdepth: 1
        :caption: Contents:
        
        auto_examples/plot_dl_keras
