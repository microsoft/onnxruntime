import sys

def process_regline(reg_line, out_file, ops):
    #print("reg_line: " + reg_line)
    reg_starts = False
    #print("processing: " + reg_line)
    # check if this line contains the ops we're interested in
    # write it out to the output file
    # BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, float, ReduceSumSquare)>,
    end = reg_line.rfind(")>")
    start = reg_line[:-1].rfind(",")
    #print("start: " ,start ," end: ",end)
    op_name = reg_line[start+1:end]
    op_name = op_name.strip()
    #print("found op_name: [" + op_name + "]")
    #print(ops)
    if op_name in ops:
        #print("writing: " + reg_line)
        out_file.write(reg_line + "\n")
    
def main():
    OPS_FILE = sys.argv[1]
    ORT_FILE = sys.argv[2]
    OUT_FILE = sys.argv[3]
    
    # read ops file
    ops = set()
    with open(OPS_FILE, 'r') as f:
        for line in f:
            #print("reading line: " + line)
            ops.add(line.strip())

    # read cpu_execution_provider.cc file
    out_file = open(OUT_FILE, 'w')
    reg_line = ""
    reg_starts = False
    with open(ORT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            #print("line: " + line)
            if line.startswith('BuildKernelCreateInfo') and (line.find(")>,") != -1):
                process_regline(line, out_file, ops)
            elif line.startswith("BuildKernelCreateInfo"):
                reg_starts = True
                reg_line = line
            elif reg_starts == True and line.find(")>,") != -1:
                reg_line = reg_line + line
                reg_starts = False
                process_regline(reg_line, out_file, ops)
            else:
                out_file.write(line + "\n")
    out_file.close()

if __name__ == '__main__':
    main()
