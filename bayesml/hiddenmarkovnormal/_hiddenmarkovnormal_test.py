from bayesml import hiddenmarkovnormal
import numpy as np

model = hiddenmarkovnormal.LearnModel(
                c_num_classes=3,
                c_degree=1)
# model.visualize_model()

print(model._ln_c_h0_eta_vec)
print(model._ln_c_h0_zeta_vecs)
print(model._ln_b_h0_w_nus)