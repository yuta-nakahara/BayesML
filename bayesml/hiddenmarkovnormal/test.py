from bayesml import hiddenmarkovnormal
import numpy as np

model = hiddenmarkovnormal.GenModel(
    3,2,h_w_mats=np.tile(np.identity(2)*2,[4,1,1]))

print(model.get_h_params())