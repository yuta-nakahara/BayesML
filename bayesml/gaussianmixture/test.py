from bayesml import gaussianmixture
import numpy as np

# model = gaussianmixture.GenModel(pi_vec=np.array([0.444,0.444,0.112]),
#                                  mu_vecs=np.array([[-2.8],[-0.8],[2]]),
#                                  lambda_mats=np.array([[[6.25]],[[6.25]],[[100]]]))
model = gaussianmixture.GenModel(mu_vecs=np.array([[2,2],[-2,-2]]))
# model.gen_params()
model.visualize_model()