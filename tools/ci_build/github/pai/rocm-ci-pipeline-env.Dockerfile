FROM rocm/pytorch:rocm5.1.1_ubuntu20.04_py3.7_pytorch_1.10.0

WORKDIR /stage

# rocm-ci branch contains instrumentation needed for loss curves and perf
RUN git clone https://github.com/microsoft/huggingface-transformers.git &&\
      cd huggingface-transformers &&\
      git checkout rocm-ci &&\
      pip install -e .

RUN pip install \
      numpy \
      onnx \
      cerberus \
      sympy \
      h5py \
      datasets==1.9.0 \
      requests \
      sacrebleu==1.5.1 \
      sacremoses \
      scipy \
      scikit-learn \
      sklearn \
      tokenizers \
      sentencepiece

RUN pip install torch-ort --no-dependencies
ENV ORTMODULE_ONNX_OPSET_VERSION=14
