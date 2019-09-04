import onnx
import sys
from PIL import Image
import numpy as np 
from onnx import numpy_helper

def write_tensor(f,tensor,input_name=None):
    if input_name:
	   tensor.name = input_name
    body = tensor.SerializeToString()
    f.write(body)

def save_image_as_pb_file(input_filename, input_name, color_space, output_file_path):
    im = Image.open(input_filename)
    if color_space == 'BGR':
      r,g,b = im.split()
      im = Image.merge("RGB", (b, g, r))
    elif color_space != 'RGB':
        raise RuntimeError('unknown color space')
    im_np = np.array(im)
    #convert to CHW format
    im_np = np.transpose(im_np, (2, 0, 1))
	#add the N dim
    im_np = np.expand_dims(im_np, axis=0)
    #now im_np is in NCHW format
	with  open(output_file_path, "wb") as f:
      t = numpy_helper.from_array(im_np.astype(np.float32))
      write_tensor(f,t,input_name)

image_file_name = sys.argv[1]
save_image_as_pb_file(image_file_name, 'RGB')
