from bayesml import hiddenmarkovnormal
import numpy as np
model = hiddenmarkovnormal.GenModel(
        c_num_classes=2,
        c_degree=1,
        mu_vecs=np.array([[5],[-5]]),
        a_mat=np.array([[0.95,0.05],[0.1,0.9]]))
model.visualize_model()