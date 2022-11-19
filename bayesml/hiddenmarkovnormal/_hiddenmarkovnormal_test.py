from bayesml import hiddenmarkovnormal
import numpy as np
from matplotlib import pyplot as plt

gen_model = hiddenmarkovnormal.GenModel(
    c_num_classes=2,
    c_degree=2,
    mu_vecs=np.array([[5,5],[-5,-5]]),
    a_mat=np.array([[0.95,0.05],[0.1,0.9]])
)
# model.visualize_model()
x,z = gen_model.gen_sample(sample_length=200)

learn_model = hiddenmarkovnormal.LearnModel(
    c_num_classes=3,
    c_degree=2,
)
learn_model.update_posterior(x[:150])#,init_type='random_responsibility')

pred_values = np.empty([50,2])
for i in range(50):
    pred_values[i] = learn_model.pred_and_update(x[150+i],loss="0-1")

plt.plot(np.arange(50),pred_values[:,0])
plt.plot(np.arange(50),x[150:,0])
plt.show()
