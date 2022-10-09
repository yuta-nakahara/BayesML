import numpy as np

from bayesml.multinomialmixture import GenModel

model = GenModel(c_num_classes=5,
                 c_degree=4,
                 pi_vec=np.array([0.2, 0.2, 0.2, 0.0, 0.4]),
                 theta_vecs=np.array([[0.25, 0.25, 0.0, 0.5], [0.5, 0.25, 0.25, 0.0], [0.0, 0.5, 0.25, 0.25],
                                      [0.25, 0.0, 0.5, 0.25], [0.25, 0.25, 0.0, 0.5]]),
                 h_alpha_vec=np.array([1.0] * 5),
                 h_beta_vec=np.array([1.0] * 4))

print(model.get_params())

print(model.get_h_params())

model.set_params(pi_vec=np.array([0.2, 0.2, 0.2, 0.1, 0.3]),
                 theta_vecs=np.array([[0.25, 0.25, 0.1, 0.4], [0.4, 0.25, 0.25, 0.1], [0.1, 0.4, 0.25, 0.25],
                                      [0.25, 0.1, 0.4, 0.25], [0.25, 0.25, 0.1, 0.4]]))

print(model.get_params())

model.set_h_params(h_alpha_vec=np.array([0.1] * 5), h_beta_vec=np.array([0.1] * 4))

print(model.get_h_params())

print('\n')
print('warning: ')

model = GenModel(c_num_classes=5,
                 c_degree=4,
                 pi_vec=np.array([0.2, 0.2, 0.2, 0.0, 0.4]),
                 theta_vecs=np.array([[0.25, 0.25, 0.0, 0.5], [0.5, 0.25, 0.25, 0.0], [0.0, 0.5, 0.25, 0.25],
                                      [0.25, 0.0, 0.5, 0.25], [0.25, 0.25, 0.0, 0.5]]),
                 h_alpha_vec=np.array([1.0] * 5),
                 h_beta_vec=np.array([1.0] * 4))

#model.set_params(pi_vec=np.array([0.2, 0.2, 0.2, 0.4]),
#                 theta_vecs=np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]))

print(model.get_params())
print(model.get_h_params())

model = GenModel(c_num_classes=5,
                 c_degree=4,
                 pi_vec=np.array([0.2, 0.2, 0.2, 0.0, 0.4]),
                 theta_vecs=np.array([[0.25, 0.25, 0.0, 0.5], [0.5, 0.25, 0.25, 0.0], [0.0, 0.5, 0.25, 0.25],
                                      [0.25, 0.0, 0.5, 0.25], [0.25, 0.25, 0.0, 0.5]]),
                 h_alpha_vec=np.array([1.0] * 5),
                 h_beta_vec=np.array([1.0] * 4))

model.set_h_params(h_alpha_vec=np.array([0.1] * 5), h_beta_vec=np.array([0.1] * 5))

print(model.get_params())
print(model.get_h_params())
