from bayesml import gaussianmixture
import numpy as np
from matplotlib import pyplot as plt
from time import time

gen_model = gaussianmixture.GenModel(
    num_classes=2,
    degree=2,
    mu_vecs=np.array([[-2,-2],[2,2]]),
    )
x,z = gen_model.gen_sample(sample_size=100)
print(x.shape)

learn_model = gaussianmixture.LearnModel(num_classes=10, degree=2, h0_alpha_vec=10)
for i in range(100):
    learn_model.pred_and_update(x[i])
    plt.scatter(x[:i+1,0],x[:i+1,1])
    plt.show()
    learn_model.visualize_posterior()
