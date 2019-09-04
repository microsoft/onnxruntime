from tensorflow.python.tools import saved_model_utils
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants

input_saved_model_dir = '.'
input_graph_def = saved_model_utils.get_meta_graph_def(
        input_saved_model_dir, tag_constants.SERVING).graph_def
print(input_graph_def)