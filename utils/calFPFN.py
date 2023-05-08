import numpy as np

def cal_FP(target, pred,idx=1):
    interest = target * pred
    fp = pred ^ interest
    fp_return = np.array(np.where(fp == True, idx, 0))
    return fp_return


def cal_FN(target, pred, idx=1):
    interest = target * pred
    fn = target ^ interest
    fn_return = np.array(np.where(fn == True, idx, 0))
    return fn_return