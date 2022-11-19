from bayesml import gaussianmixture
import numpy as np
from matplotlib import pyplot as plt
from time import time

gen_model = gaussianmixture.GenModel(
    c_num_classes=3,
    c_degree=1,
    pi_vec=np.array([0.444,0.444,0.112]),
    mu_vecs=np.array([[-2.8],[-0.8],[2]]),
    lambda_mats=np.array([[[6.25]],[[6.25]],[[100]]])
    )

x,z = gen_model.gen_sample(300)

learn_model = gaussianmixture.LearnModel(3,1)
learn_model.update_posterior(x)#,init_type='random_responsibility')
learn_model.visualize_posterior()