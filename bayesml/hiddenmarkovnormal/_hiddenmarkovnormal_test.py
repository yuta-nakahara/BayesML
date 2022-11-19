from bayesml import hiddenmarkovnormal
import numpy as np

gen_model = hiddenmarkovnormal.GenModel(
    c_num_classes=2,
    c_degree=2,
    mu_vecs=np.array([[2,2],[-2,-2]]),
    a_mat=np.array([[0.95,0.05],[0.1,0.9]])
)
# model.visualize_model()
x,z = gen_model.gen_sample(sample_length=200)

learn_model = hiddenmarkovnormal.LearnModel(
    c_num_classes=2,
    c_degree=2,
)
learn_model.update_posterior(x)#,init_type='random_responsibility')
learn_model.visualize_posterior()
# print(learn_model.estimate_params(loss='KL'))