from bayesml import gaussianmixture
import numpy as np

model = gaussianmixture.LearnModel(num_classes=5, degree=3)

x = np.random.rand(10,3)

model.update_posterior(x,num_init=3)
