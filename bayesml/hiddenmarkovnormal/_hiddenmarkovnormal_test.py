from bayesml import hiddenmarkovnormal
import numpy as np

model = hiddenmarkovnormal.GenModel(3,1)

print(model.get_params())

model.set_params(mu_vecs=np.ones([3,1]))

print(model.get_params())
