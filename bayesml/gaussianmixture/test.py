from bayesml import gaussianmixture
import numpy as np
from matplotlib import pyplot as plt
from time import time

# gen_model = gaussianmixture.GenModel(
#     num_classes=2,
#     degree=1,
#     mu_vecs=np.array([[-2],[2]]),
#     )
# gen_model.save_sample('GMM_sample',sample_size=1000)

x = np.load('GMM_sample.npz')['x']
learn_model = gaussianmixture.LearnModel(num_classes=4, degree=1,seed=123)

start = time()
learn_model.update_posterior(x)
end = time()
print(end-start)
print(learn_model.hn_m_vecs)

# learn_model.visualize_posterior()