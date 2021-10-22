FROM rocm/pytorch:rocm4.3.1_ubuntu18.04_py3.6_pytorch_1.9.0

WORKDIR /stage

# from rocm/pytorch's image, work around ucx's dlopen replacement conflicting with shared provider
RUN cd /opt/mpi_install/ucx/build &&\
      make clean &&\
      ../contrib/configure-release --prefix=/opt/ucx --without-rocm &&\
      make -j $(nproc) &&\
      make install

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
