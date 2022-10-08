from bayesml import gaussianmixture
import numpy as np

model = gaussianmixture.LearnModel(num_classes=3, degree=2, h0_w_mats=np.identity(2)*2)
print(model.calc_vl())