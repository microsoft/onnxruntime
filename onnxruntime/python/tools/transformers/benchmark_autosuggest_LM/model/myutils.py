import torch
import numpy as np

def outFileHandler(fh):
    global common_fh
    global counterset
    global total_infer_time
    global mask_any_counter 
    mask_any_counter = 0
    total_infer_time = 0
    counterset = False
    common_fh = fh
    common_fh.write("TotalInferenceTime\tCounter\tTotalModelTime\tTotalSearchTime\tResult\tTotalQueryTime\n")

