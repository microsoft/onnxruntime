# ORT ep names 
cpu_ep = "CPUExecutionProvider"
cuda_ep = "CUDAExecutionProvider"
trt_ep = "TensorrtExecutionProvider"
acl_ep = "ACLExecutionProvider"

# provider names 
cpu = "ORT-CPUFp32"
cuda = "ORT-CUDAFp32"
cuda_fp16 = "ORT-CUDAFp16"
trt = "ORT-TRTFp32"
trt_fp16 = "ORT-TRTFp16"
standalone_trt = "TRTFp32"
standalone_trt_fp16 = "TRTFp16"
acl = "ORT-ACLFp32"

# column names 
model_title = 'Model'
group_title = 'Group'

# endings 
avg_ending =  ' \nmean (ms)'
percentile_ending = ' \n90th percentile (ms)'
memory_ending = ' \npeak memory usage (MiB)'

provider_list = [cpu, cuda, trt, standalone_trt, cuda_fp16, trt_fp16, standalone_trt_fp16]
table_headers = [model_title] + provider_list
