import numpy as np
from bayesml.categorical import GenModel as Model
from bayesml.categoricalmixture import GenModel as MixModel



model = MixModel(m_degree=5, degree=4, pi_vec=np.array([0.2, 0.2, 0.2, 0.0, 0.4]), 
            theta_mat=np.array([[0.25, 0.25, 0.0, 0.5], [0.5, 0.25, 0.25, 0.0], [0.0, 0.5, 0.25, 0.25], [0.25, 0.0, 0.5, 0.25], [0.25, 0.25, 0.0, 0.5]]), 
            h_alpha_vec=np.array([1.0]*5), h_beta_vec=np.array([1.0]*4))

print(model.get_params())

print(model.get_h_params())

model.set_params(pi_vec=np.array([0.2, 0.2, 0.2, 0.1, 0.3]), 
            theta_mat=np.array([[0.25, 0.25, 0.1, 0.4], [0.4, 0.25, 0.25, 0.1], [0.1, 0.4, 0.25, 0.25], [0.25, 0.1, 0.4, 0.25], [0.25, 0.25, 0.1, 0.4]]))

print(model.get_params())

model.set_h_params(h_alpha_vec=np.array([0.1]*5), h_beta_vec=np.array([0.1]*4))

print(model.get_h_params())

print('\n')
print('warning: ')

model = MixModel(m_degree=5, degree=4, pi_vec=np.array([0.2, 0.2, 0.2, 0.0, 0.4]), 
            theta_mat=np.array([[0.25, 0.25, 0.0, 0.5], [0.5, 0.25, 0.25, 0.0], [0.0, 0.5, 0.25, 0.25], [0.25, 0.0, 0.5, 0.25], [0.25, 0.25, 0.0, 0.5]]), 
            h_alpha_vec=np.array([1.0]*5), h_beta_vec=np.array([1.0]*4))

model.set_params(pi_vec=np.array([0.2, 0.2, 0.2, 0.4]), 
            theta_mat=np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]))

print(model.get_params())
print(model.get_h_params())

model = MixModel(m_degree=5, degree=4, pi_vec=np.array([0.2, 0.2, 0.2, 0.0, 0.4]), 
            theta_mat=np.array([[0.25, 0.25, 0.0, 0.5], [0.5, 0.25, 0.25, 0.0], [0.0, 0.5, 0.25, 0.25], [0.25, 0.0, 0.5, 0.25], [0.25, 0.25, 0.0, 0.5]]), 
            h_alpha_vec=np.array([1.0]*5), h_beta_vec=np.array([1.0]*4))

model.set_h_params(h_alpha_vec=np.array([0.1]*4), h_beta_vec=np.array([0.1]*5))

print(model.get_params())
print(model.get_h_params())
