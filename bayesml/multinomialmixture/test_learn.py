import numpy as np

from bayesml.multinomialmixture import LearnModel

model = LearnModel(c_num_classes=5,
                   c_degree=4,
                   h0_alpha_vec=np.array([1.0] * 5),
                   h0_beta_vecs=np.array([[1.0, 1.0, 1.0, 1.0]] * 5))

print(model.get_h0_params())

print(model.get_hn_params())
