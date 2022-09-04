from bayesml import gaussianmixture
import numpy as np

model = gaussianmixture.GenModel(num_classes=2,degree=3)
print(model.get_params())
params = model.get_params()
params['pi_vec'] = np.ones(6)/6
params['mu_vecs'] = np.ones([2,3,3])
params['lambda_mats'] = np.tile(np.identity(4),[2,3,1,1])
model.set_params(*params.values())
print(model.get_params())