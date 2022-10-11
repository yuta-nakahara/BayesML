import numpy as np
from bayesml import logisticregression

model = logisticregression.LearnModel(2,h0_lambda_mat=np.identity(2)*2)

print(model.get_h0_params())