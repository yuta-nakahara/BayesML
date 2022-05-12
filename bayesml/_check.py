# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Yuji Iikubo <yuji-iikubo.8@fuji.waseda.jp>
import numpy as np
from ._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning

FLOATS = list({'float128','float64','float32','float16'} & set(dir(np)) | {float})
INTS = list({'int64','int32','int16','int8'} & set(dir(np)) | {int})

def float_in_closed01(val,val_name,exception_class):
    if type(val) in FLOATS:
        if val >= 0.0 and val <= 1.0:
            return val
    if type(val) in INTS:
        if val >= 0.0 and val <= 1.0:
            return float(val)
    raise(exception_class(val_name + " must be in [0,1]."))

def pos_float(val,val_name,exception_class):
    if type(val) in FLOATS:
        if val > 0.0:
            return val
    if type(val) in INTS:
        if val > 0.0:
            return float(val)
    raise(exception_class(val_name + " must be positive (not including 0.0)."))

def pos_int(val,val_name,exception_class):
    if type(val) in INTS:
        if val > 0:
            return val
    raise(exception_class(val_name + " must be int. Its value must be positive (not including 0)."))

def nonneg_int(val,val_name,exception_class):
    if type(val) in INTS:
        if val >= 0:
            return val
    raise(exception_class(val_name + " must be int. Its value must be non-negative (including 0)."))

def nonneg_ints(val,val_name,exception_class):
    try:
        return nonneg_int(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if val.dtype in INTS and np.all(val>=0):
            return val
    raise(exception_class(val_name + " must be int or a numpy.ndarray whose dtype is int. Its values must be non-negative (including 0)."))

def nonneg_int_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.dtype in INTS and val.ndim == 1 and np.all(val>=0):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray whose dtype is int. Its values must be non-negative (including 0)."))

def int_of_01(val,val_name,exception_class):
    if type(val) in INTS:
        if val == 0 or val ==1:
            return val
    raise(exception_class(val_name + " must be int. Its value must be 0 or 1."))

def ints_of_01(val,val_name,exception_class):
    try:
        return int_of_01(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if val.dtype in INTS and np.all(val >= 0) and np.all(val <= 1):
            return val
    raise(exception_class(val_name + " must be int or a numpy.ndarray whose dtype is int. Its values must be 0 or 1."))

def int_vec_of_01(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.dtype in INTS and val.ndim == 1 and np.all(val >= 0) and np.all(val <= 1):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray whose dtype is int. Its values must be 0 or 1."))

def scalar(val,val_name,exception_class):
    if type(val) in INTS or type(val) in FLOATS:
        return val
    raise(exception_class(val_name + " must be a scalar."))

def pos_scalar(val,val_name,exception_class):
    if type(val) in INTS or type(val) in FLOATS:
        if val > 0.0:
            return val
    raise(exception_class(val_name + " must be a positive scalar."))

def sym_mat(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.ndim == 2 and val.shape[0] == val.shape[1]:
            if np.allclose(val, val.T):
                return val
    raise(exception_class(val_name + " must be a symmetric 2-dimensional numpy.ndarray."))

def pos_def_sym_mat(val,val_name,exception_class):
    sym_mat(val,val_name,exception_class)
    try:
        np.linalg.cholesky(val)
        return val
    except np.linalg.LinAlgError:
        pass
    raise(exception_class(val_name + " must be a positive definite symmetric 2-dimensional numpy.ndarray."))

def float_(val,val_name,exception_class):
    if type(val) in FLOATS:
        return val
    if type(val) in INTS:
        return float(val)
    raise(exception_class(val_name + " must be a scalar."))

def floats(val,val_name,exception_class):
    try:
        return float_(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if val.dtype in INTS:
            return val.astype(float)
        if val.dtype in FLOATS:
            return val
    raise(exception_class(val_name + " must be float or a numpy.ndarray."))

def pos_floats(val,val_name,exception_class):
    try:
        return pos_float(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if val.dtype in INTS and np.all(val>0):
            return val.astype(float)
        if val.dtype in FLOATS and np.all(val>0.0):
            return val
    raise(exception_class(val_name + " must be float or a numpy.ndarray. Its values must be positive (not including 0)"))

def float_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.dtype in INTS and val.ndim == 1:
            return val.astype(float)
        if val.dtype in FLOATS and val.ndim == 1:
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray."))

def pos_float_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.dtype in INTS and val.ndim == 1 and np.all(val>0):
            return val.astype(float)
        if val.dtype in FLOATS and val.ndim == 1 and np.all(val>0.0):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray. Its values must be positive (not including 0)"))
    
def float_vecs(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.dtype in INTS and val.ndim >= 1:
            return val.astype(float)
        if val.dtype in FLOATS and val.ndim >= 1:
            return val
    raise(exception_class(val_name + " must be a numpy.ndarray whose ndim >= 1."))

if __name__ == '__main__':
    a = np.ones([1,1,2])
    print(pos_floats(np.ones([3,4])*(1.0e-8),'tmp',DataFormatError))
