from bayesml import linearregressionmixture
import numpy as np

model = linearregressionmixture.LearnModel(
    4,
    3,
    )

print(model.get_h0_params())