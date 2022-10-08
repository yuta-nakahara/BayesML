from bayesml import gaussianmixture
import numpy as np

model = gaussianmixture.LearnModel(num_classes=2, degree=2, h0_w_mats=np.identity(2)*2)

print(model.get_hn_params())
model.set_hn_params(hn_m_vecs=np.ones(2))
print(model.get_hn_params())
