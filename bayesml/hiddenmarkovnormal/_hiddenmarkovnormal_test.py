from bayesml import hiddenmarkovnormal
import numpy as np

gen_model = hiddenmarkovnormal.GenModel(
    c_num_classes=3,
    c_degree=1,
    mu_vecs=np.array([[5],[0],[-5]]),
    a_mat=np.array([[0.95,0.05,0.0],[0.0,0.9,0.1],[0.1,0.0,0.9]])
)
# gen_model.visualize_model()
x,z = gen_model.gen_sample(sample_length=200)

learn_model = hiddenmarkovnormal.LearnModel(
    c_num_classes=3,
    c_degree=1,
)
learn_model.update_posterior(x)#,init_type='random_responsibility')
# print(learn_model.get_hn_params())