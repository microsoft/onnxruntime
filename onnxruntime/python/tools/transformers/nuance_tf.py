import os
import psutil
import time
import numpy as np
import tensorflow as tf
print(f"Tensorflow Version: {tf.__version__}")

def get_input(batch_size = 1, seq_length = 256):

    # Bert Inputs
    input_ids = np.random.uniform(1, 10, size=(batch_size, seq_length)).astype('int32')
    input_mask = np.ones((batch_size, seq_length), dtype="int32")
    segment_ids = np.zeros((batch_size, seq_length), dtype="int32")


    # External Feature
    n_external_feature = 250
    external_features = np.random.uniform(1, 10, size=(batch_size, seq_length*n_external_feature)).astype('float32') # resized back to 3 dimensions in the model

    return input_ids, input_mask, segment_ids, external_features

tf_model_path = "./tf_model"

#os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
#os.environ['PATH'] = os.environ['PATH'] + ":" + "/usr/loca/cuda-10.2/include/:/usr/loca/cuda-10.2/lib64/"

tf.compat.v1.reset_default_graph()
tf_session = tf.compat.v1.Session()
tf.compat.v1.saved_model.loader.load(tf_session, tags=['serve'], export_dir=tf_model_path);

output_placeholder = []
from tensorflow.python.tools import saved_model_utils
meta_output = saved_model_utils.get_meta_graph_def(tf_model_path, 'serve').signature_def['serve'].outputs
[output_placeholder.append(tf_session.graph.get_tensor_by_name(meta_output[k].name)) for k in meta_output.keys()];

total_runs = 300

tf.config.threading.set_inter_op_parallelism_threads(psutil.cpu_count(logical=True))
tf.config.threading.set_intra_op_parallelism_threads(psutil.cpu_count(logical=True))

input_ids, input_mask, segment_ids, external_features = get_input(1, 32)
result = tf_session.run(output_placeholder,feed_dict={"serve_X_1:0":input_ids, "serve_X_2:0": input_mask, "serve_X_3:0": segment_ids, "serve_X:0": external_features})

# timing
for batch_size in [1, 32]:
    for seq_length in [32, 64, 128, 256]:
        input_ids, input_mask, segment_ids, external_features = get_input(batch_size, seq_length)

        start = time.time()
        for _ in range(total_runs):
            result = tf_session.run(output_placeholder,feed_dict={"serve_X_1:0":input_ids, "serve_X_2:0": input_mask, "serve_X_3:0": segment_ids, "serve_X:0": external_features})
        end = time.time()

        print("Tensorflow Inference time for sequence length={} and batch size={} is {} ms".format(seq_length, batch_size, format((end - start) * 1000 / total_runs, '.2f')))
    print("*"*100)

tf_session.close()