import os
import cv2
import sys

import numpy as np
import pandas as pd


global start_time
def start_timer():
    global start_time
    start_time = pd.Timestamp.now()
    
def tick(msg=''):
    global start_time
    print(msg + ', Time Taken: %s'%str(pd.Timestamp.now() - start_time))
    
def check_memory_usage():
    a = []
    for var, obj in locals().items():
        a += [[var, sys.getsizeof(obj)]]
    print(pd.DataFrame(a).sort_values(1))
    
def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            #if param.dim() > 1:
                #print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            #else:
                #print(name, ':', num_param)
            total_param += num_param
    return total_param


def adjust_range(img):
    return (255*((img - img.min())/(img.max() - img.min()))).astype('uint8')

