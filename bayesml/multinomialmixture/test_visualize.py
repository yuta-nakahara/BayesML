import numpy as np

from bayesml.multinomialmixture import GenModel

model = GenModel(pi_vec=np.array([0.444, 0.444, 0.112]), theta_vecs=np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]))

x, z = model.gen_sample(10)
print(x)
print(z)
model.save_sample('bayesml/multinomialmixture/sample_2', 10)
model.visualize_model('bayesml/multinomialmixture/figure_2.png', 1000)

model.gen_params()
print(model.pi_vec)
print(model.theta_vecs)

model = GenModel(pi_vec=np.array([0.444, 0.444, 0.112]),
                 theta_vecs=np.array([[0.1, 0.2, 0.7], [0.5, 0.2, 0.3], [0.1, 0.8, 0.1]]))

x, z = model.gen_sample(10)
print(x)
print(z)
model.save_sample('bayesml/multinomialmixture/sample_3', 10)
model.visualize_model('bayesml/multinomialmixture/figure_3.png', 1000, num_trials=80)

model.gen_params()
print(model.pi_vec)
print(model.theta_vecs)