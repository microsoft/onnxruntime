import tml_pb2
from pathlib import Path
import onnx
import shutil
from onnx import numpy_helper
from google.protobuf import text_format
from PIL import Image
import numpy as np 
import json
import sys
import csv
import os
import six
import glob
import subprocess
from onnx import ModelProto

#copied from https://github.com/google/protobuf/blob/master/python/google/protobuf/internal/encoder.py
def EncodeVarint(f, value):
    bits = value & 0x7f
    value >>= 7
    while value:
        f.write(six.int2byte(0x80|bits))
        bits = value & 0x7f
        value >>= 7
    return f.write(six.int2byte(bits))

def write_tensor(f,t,input_name=None,debug_info=None):
    data = tml_pb2.TraditionalMLData()
    if input_name:
      t.name = input_name
      data.name = input_name
    if debug_info:
      data.debug_info = debug_info
    data.tensor.CopyFrom(t);    
    body = data.SerializeToString()
    EncodeVarint(f,len(body))
    f.write(body)	 

def get_model_input(model):
    initializer_names = {init.name for init in model.graph.initializer};
    return [x for x in model.graph.input if x.name not in initializer_names]

def convert_csv_input(test_name, input_data, input_data_path, model):
    if 'shape' in input_data:
        shape = input_data['shape']
    else:
        shape = [1]
    binding_Names = None
    if 'binding_Names' in input_data:
        binding_Names = input_data['binding_Names']
    test_name = tafe_json['test_name']
    input_is_map = False    
    model_inputs = get_model_input(model)
    input_is_all_tensor = is_all_tensor(model_inputs)
    if len(model_inputs) == 1 and model_inputs[0].type.HasField("map_type"):
        input_is_map = True
        if model_inputs[0].type.map_type.key_type != onnx.TensorProto.STRING:
          print("not supported")
          exit(-1)
      
    with  open(input_data_path, "r") as f:
        reader = csv.reader(f, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        data_set_id = 0
        for row in reader:
            data_set_dir = os.path.join(test_name,"dataset_%d" % data_set_id)
            #print('creating %s' % data_set_dir)
            os.makedirs(data_set_dir)
            if binding_Names is None:
                cols = len(row) - 1
                binding_Names = [str(x) for x in range(1,cols + 1)]
            else:
                cols = len(binding_Names)
                if(len(row) < cols):
                    print('length mismatch, expect %d, got %d' % (len(binding_Names), cols))
                    exit(-1)
            if input_is_all_tensor:
                output_tensor_list(test_name,os.path.join(data_set_dir, "inputs.pb"),model_inputs,row,True)                            
            elif input_is_map:
                   name = model_inputs[0].name
                   data = tml_pb2.TraditionalMLData()                   
                   for i in range(cols):
                       string_data = row[i + 1]
                       if len(string_data) == 0:
                           x = np.float32(np.nan) #should read it from the replaced_value_float of Impute OP
                       else:
                           x = np.float32(string_data)
                       data.map_string_to_float.v[str(i+1)] = x
                   data.name = name
                   body = data.SerializeToString()
                   with open(os.path.join(data_set_dir, "inputs.pb"), "wb") as f:
                       EncodeVarint(f,len(body))
                       f.write(body) 
            else:
               with open(os.path.join(data_set_dir, "inputs.pb"), "wb") as f:
                    for i in range(cols):
                        #skip the first column
                        string_data = row[i + 1]
                        if test_name in ['coreml_OneHotEncoder_OpenML_3_chess']:
                            x = np.int64(string_data)
                        else:
                            if len(string_data) == 0:
                                x = np.float32(np.nan) #should read it from the replaced_value_float of Impute OP
                            else:
                                x = np.float32(string_data)
                        x = np.reshape(x, shape)                
                        t = numpy_helper.from_array(x)
                        write_tensor(f,t,binding_Names[i])                    
            data_set_id+=1
            if data_set_id >= 500:
               break

def save_image_as_pb_file(filename, data_set_dir, input_name, color_space, is_input):
    im = Image.open(filename)
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
    if is_input:
        fname = "inputs.pb"
    else:
        fname = "outputs.pb"
    with  open(os.path.join(data_set_dir,fname), "wb") as f:
      t = numpy_helper.from_array(im_np.astype(np.float32))
      write_tensor(f,t,input_name,os.path.basename(filename))
      

def GetCats(proto):   
    for input in proto.graph.node:
      #and input.output == 'classLabelProbs'
      if(input.op_type == 'ZipMap' and (input.output == ['classLabelProbs'] or input.output == ['prob'])):
        for l in input.attribute:
            if(l.name == 'classlabels_strings'):
                #dedup
                t = {}
                for c in l.strings:
                    if c in t:
                        continue
                    t[c] = 1;
                ret = list(t.keys())
                ret.sort()
                return ret

def gen_input_data(root_data_path, tafe_json, model):
    input_data = tafe_json['data_files']['model_input_data']   
    input_data_path = os.path.join(root_data_path, input_data['path'])
    if(not os.path.exists(input_data_path)):
       raise RuntimeError("input path %s not found" % input_data_path)
    if(os.path.isdir(input_data_path)):
       model_inputs = get_model_input(model)
       name = model_inputs[0].name
       return convert_image_input(input_data,input_data_path, name)
    else:
       convert_csv_input(tafe_json['test_name'], input_data,input_data_path, model)
       return None

def convert_image_input(input_data,input_data_path, input_name):
    data_set_id = 0
    test_name = tafe_json['test_name']
    filename_to_dataset_id = {}    
    for filename in glob.glob(os.path.join(input_data_path,'*.png')):
      data_set_dir = os.path.join(test_name,"dataset_%d" % data_set_id)
      #print('creating %s' % data_set_dir)
      os.makedirs(data_set_dir)
      color_space = 'RGB'
      if 'color_space' in input_data:
          color_space = input_data['color_space']
      save_image_as_pb_file(filename,data_set_dir,input_name, color_space, True)
      filename_to_dataset_id[os.path.basename(filename)] = data_set_id
      data_set_id+=1
    return filename_to_dataset_id

   
def convert_image_output(test_name, filename_to_dataset_id, output_data_path, tensor_name):
    color_space = 'RGB'
    if test_name == 'coreml_FNS-Candy_ImageNet':
        color_space = 'BGR'
    for filename, dataset_id in filename_to_dataset_id.items():
        data_set_dir = os.path.join(test_name,"dataset_%d" % dataset_id)
        save_image_as_pb_file(os.path.join(output_data_path,filename), data_set_dir, tensor_name, color_space, False)

#output_info is TypeProto.Tensor
def convert_python_list_to_tensor(test_name,input, output_info, f, name=None):
    shape = []
    for d in output_info.shape.dim:
        shape += [d.dim_value]
    if shape[0] == 0:
        shape[0] = 1
    try:
        if output_info.elem_type == onnx.TensorProto.FLOAT:
            input = [x if x !='' else 'NaN' for x in input]
            x = np.float32(input)
        elif output_info.elem_type == onnx.TensorProto.INT32:
            x = np.int32(input)
        elif output_info.elem_type == onnx.TensorProto.INT64:
            #if test_name in ['XGBoost_XGClassifier_sklearn_load_wine', 'XGBoost_XGClassifier_OpenML_312_scene', 'XGBoost_XGClassifier_sklearn_load_breast_cancer', 'XGBoost_XGClassifier_OpenML_1464_blood_transfusion']:
            #they often save int64 as float!Come on, it will lose precision
            x = np.int64([float(i) for i in input])
            #else:
            #    x = np.int64(input)            
        elif output_info.elem_type != onnx.TensorProto.STRING:
            print('unknown type %s' % output_info.elem_type)
            exit(-1)                    
    except ValueError as err:
        print('convert fail, input=%s, err=%s' % (input, err))
        raise err
    if output_info.elem_type == onnx.TensorProto.STRING:
        t = onnx.TensorProto()
        t.dims.extend(shape)
        t.data_type = onnx.TensorProto.STRING
        t.string_data.extend([x.encode('utf-8') for x in input])
    else:
        x = np.reshape(x, shape)
        t = numpy_helper.from_array(x)
    write_tensor(f,t,name)

def is_all_tensor(value_info_list):
    for x in value_info_list:
        if not x.type.HasField("tensor_type"):
            return False
    return True

def get_shape_size(shape):
    dims = len(shape.dim)
    ret = 1;
    for i in range(dims):
        d = shape.dim[i].dim_value
        if d <= 0:
            if i == 0:
                continue;
            else:
                printf("strange shape!!")
                exit(-1)
        ret *= d
    return ret


def output_tensor_list(test_name, filename, output_info,row, skip_first_col):
    with open(filename, "wb") as f:
        slice_lens = []
        if skip_first_col:
            total_len = 1;
        else:
            total_len = 0;
        for out in output_info:
            slen = get_shape_size(out.type.tensor_type.shape)
            slice_lens.append(slen)
            total_len += slen
        if total_len != len(row):
            print('length mismatch, expect %d, got %d' % (total_len, len(row)))
            print(output_info)
            exit(-1)
        if skip_first_col:
            start = 1;
        else:
            start = 0;
        for i in range(len(slice_lens)):
            end = start + slice_lens[i]
            convert_python_list_to_tensor(test_name, row[start:end],output_info[i].type.tensor_type,f,output_info[i].name)                   
            start = end

def gen_output_data(root_data_path, tafe_json, model, filename_to_dataset_id):
    output_data = tafe_json['data_files']['model_output_data']
    test_name = tafe_json['test_name']
    if 'shape' in output_data:
        shape = output_data['shape']
    else:
        a = model.graph.output[0].type   
        if a.HasField("tensor_type"):
            shape = []
            for d in a.tensor_type.shape.dim:
                shape += [d.dim_value]
            if shape[0] == 0:
              shape[0] = 1
    binding_Names = None
    if 'binding_Names' in output_data:
        binding_Names = output_data['binding_Names']    
    output_data_path = os.path.join(root_data_path, output_data['path'])
    if(os.path.isdir(output_data_path)):
        return convert_image_output(test_name,filename_to_dataset_id, output_data_path,'')
    output_is_all_tensor = is_all_tensor(model.graph.output)
    if not output_is_all_tensor:
        cats = GetCats(model)    
    with  open(output_data_path, "r") as f:
        reader = csv.reader(f, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        if filename_to_dataset_id: #image input, csv output                        
            for row in reader:
                data_set_id = filename_to_dataset_id[row[0]]
                data_set_dir = os.path.join(test_name,"dataset_%d" % data_set_id)
                if output_is_all_tensor:
                   output_tensor_list(test_name,os.path.join(data_set_dir, "outputs.pb"),model.graph.output,row,True)
                else:
                    #print("%s:%s" % (row[0],data_set_dir))
                    with  open(os.path.join(data_set_dir, "outputs.pb"), "wb") as f:
                        #first column is file name
                        tensor = onnx.TensorProto()
                        tensor.dims.extend([1,1])
                        tensor.data_type = onnx.TensorProto.STRING
                        tensor.string_data.extend([row[1].encode('utf-8')])
                        write_tensor(f,tensor,debug_info=row[0])

                        y = row[2:]
                        data = tml_pb2.TraditionalMLData()                    
                        a = data.vector_map_string_to_float.v.add()
                        for i in range(len(y)):
                            a.v[cats[i]] = float(y[i])
                        body = data.SerializeToString()
                        EncodeVarint(f,len(body))
                        f.write(body)                    
        else: #csv input,csv output
            data_set_id = 0
            for row in reader:           
               data_set_dir = os.path.join(test_name,"dataset_%d" % data_set_id)
               if output_is_all_tensor:
                   output_tensor_list(test_name,os.path.join(data_set_dir, "outputs.pb"),model.graph.output,row,False)
                   #'coreml_DictVectorizer-GradientBoostingRegressor_sklearn_load_boston', 'coreml_Imputer_OpenML_1492_plants_Missing','coreml_OneHotEncoder_OpenML_3_chess','coreml_OneHotEncoder_BikeSharing','coreml_Imputer_sklearn_load_Iris_missing','coreml_Imputer_sklearn_load_diabetes_missing','coreml_Imputer_OpenML_1464_blood_transfusion_missing','coreml_Normalizer-RandomForestRegressor_sklearn_load_diabetes','coreml_Normalizer-LinearRegression_sklearn_load_boston','coreml_Normalizer-GradientBoostingRegressor_sklearn_load_boston','coreml_Normalizer_sklearn_load_wine','coreml_Normalizer_sklearn_load_diabetes','coreml_Normalizer_sklearn_load_breast_cancer','coreml_Normalizer_OpenML_312_scene','coreml_Normalizer_OpenML_1464_blood_transfusion'                   
               elif len(model.graph.output) == 1 and model.graph.output[0].type.HasField("sequence_type"):
                   with  open(os.path.join(data_set_dir, "outputs.pb"), "wb") as f:
                       #libsvm_Nu_SVC_OpenML_312_scene
                        y = row
                        data = tml_pb2.TraditionalMLData()
                        a = data.vector_map_int64_to_float.v.add()
                        for i in range(len(y)):
                          a.v[i] = float(y[i])
                        body = data.SerializeToString()
                        EncodeVarint(f,len(body))
                        f.write(body)
               elif len(model.graph.output) == 2 and model.graph.output[0].type.HasField("tensor_type") and model.graph.output[1].type.HasField("sequence_type"):
                   #'coreml_SVC_sklearn_load_wine','coreml_SVC_sklearn_load_breast_cancer', 'coreml_SVC_OpenML_1464_blood_transfusion','coreml_SVC_OpenML_312_scene', 'coreml_RandomForestClassifier_sklearn_load_wine','coreml_RandomForestClassifier_sklearn_load_breast_cancer','coreml_RandomForestClassifier_OpenML_1464_blood_transfusion','coreml_RandomForestClassifier_OpenML_312_scene','coreml_Imputer-GradientBoostingClassifier_OpenML_1464_blood_transfusion','coreml_Imputer-GradientBoostingClassifier_sklearn_load_breast_cancer','coreml_Imputer-LogisticRegression_sklearn_load_breast_cancer','coreml_Imputer-LogisticRegression_OpenML_1464_blood_transfusion_missing','coreml_Normalizer-RandomForestClassifier_sklearn_load_wine','coreml_Normalizer-LinearSVC_sklearn_load_wine','coreml_LogisticRegression_sklearn_load_breast_cancer','coreml_LogisticRegression_OpenML_312_scene','coreml_LinearSVC_sklearn_load_wine','coreml_LinearSVC_sklearn_load_breast_cancer','coreml_LinearSVC_OpenML_1464_blood_transfusion','coreml_LinearSVC_OpenML_312_scene','coreml_GradientBoostingClassifier_sklearn_load_wine','coreml_GradientBoostingClassifier_sklearn_load_breast_cancer','coreml_GradientBoostingClassifier_OpenML_1464_blood_transfusion', 'coreml_GradientBoostingClassifier_OpenML_312_scene', 'coreml_DecisionTreeClassifier_sklearn_load_wine','coreml_DecisionTreeClassifier_sklearn_load_breast_cancer_infused', 'coreml_DecisionTreeClassifier_sklearn_load_breast_cancer', 'coreml_DecisionTreeClassifier_OpenML_1464_blood_transfusion', 'coreml_GradientBoostingClassifier_Criteo','coreml_GradientBoostingClassifier_BingClick', 'coreml_DecisionTreeClassifier_OpenML_312_scene']
                   with  open(os.path.join(data_set_dir, "outputs.pb"), "wb") as f:
                        convert_python_list_to_tensor(test_name,row[0:1],model.graph.output[0].type.tensor_type,f)
                        y = row[1:]
                        data = tml_pb2.TraditionalMLData()
                        a = data.vector_map_int64_to_float.v.add()
                        for i in range(len(y)):
                          a.v[i] = float(y[i])
                        body = data.SerializeToString()
                        EncodeVarint(f,len(body))
                        f.write(body)              
               else:
                   if binding_Names is None:
                        cols = len(row)                        
                   else:
                        cols = len(binding_Names)
                   with  open(os.path.join(data_set_dir, "outputs.pb"), "wb") as f:
                       for i in range(cols):
                            x = np.float32(row[i])
                            x = np.reshape(x, shape)
                            t = numpy_helper.from_array(x)
                            write_tensor(f,t)                                
               data_set_id+=1
               if data_set_id >= 500:
                 break

def get_input_vocabulary(model):
    for input in model.graph.node:
      #and input.output == 'classLabelProbs'
      if(input.op_type == 'DictVectorizer' and input.input == ['input']):
        for l in input.attribute:
            if(l.name == 'string_vocabulary'):
                return l.strings
    exit(-1)


tafe_file_name = sys.argv[1]
tafe_json = None
with  open(tafe_file_name, "r") as f:
    tafe_json = json.load(f)
model_file_name = os.path.basename(tafe_json['data_files']['model'])
#model_path = os.path.join(r'\\redmond\1windows\TestContent\CORE\SiGMa\GRFX\WinML\RS5_convertedmodels\08_14_2018',model_file_name)
model_path = os.path.join(r'\\redmond\1windows\TestContent\CORE\SiGMa\GRFX\WinML\RS5\models\onnx-1.2',model_file_name)
#root_data_path = r'\\redmond\1windows\TestContent\CORE\SiGMa\GRFX\WinML'
#model_path = os.path.join(r'\\redmond\1windows\TestContent\CORE\SiGMa\GRFX\WinML\RS5',tafe_json['data_files']['model'])
root_data_path = r'\\redmond\1windows\TestContent\CORE\SiGMa\GRFX\WinML\RS5'


if(not os.path.exists(model_path)):
    print('cannot find model file %s' % model_path)
    exit(-1)
model = onnx.load_model(model_path)
dest_dir = tafe_json['test_name'];
if os.path.exists(dest_dir):
    print('delete %s' % dest_dir)
    shutil.rmtree(dest_dir)
filename_to_dataset_id = gen_input_data(root_data_path, tafe_json,model)
gen_output_data(root_data_path, tafe_json, model, filename_to_dataset_id)
config = tml_pb2.TestCaseConfig()
config.per_sample_tolerance = 1e-3
config.relative_per_sample_tolerance = 1e-5
if 'tolerances' in tafe_json:
    tolerances = tafe_json['tolerances']
    if 'per_sample_tolerance' in tolerances:
        config.per_sample_tolerance = float(tolerances['per_sample_tolerance'])
    if 'relative_per_sample_tolerance' in tolerances:
        config.relative_per_sample_tolerance = float(tolerances['relative_per_sample_tolerance'])
with open(os.path.join(dest_dir,'config.txt'), 'w') as config_file:
    config_file.write(text_format.MessageToString(config))

dest_file_name = Path(model_file_name).stem + ".onnx"
shutil.copyfile(model_path,os.path.join(dest_dir,dest_file_name))
#subprocess.check_call([r'C:\src\build2\RelWithDebInfo\onnx_test_runner.exe',dest_dir])