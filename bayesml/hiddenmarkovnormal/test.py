from bayesml import hiddenmarkovnormal
import numpy as np

model = hiddenmarkovnormal.LearnModel(
    3,2,h0_eta_vec=2)

print(model.get_hn_params())