from bayesml import gaussianmixture
import numpy as np
from matplotlib import pyplot as plt
from time import time

gen_model = gaussianmixture.GenModel(
    c_num_classes=2,
    c_degree=2,
    pi_vec=np.ones(3) / 3,
    )
print(gen_model.get_params())