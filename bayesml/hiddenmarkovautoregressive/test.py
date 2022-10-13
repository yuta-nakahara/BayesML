from bayesml import hiddenmarkovautoregressive
import numpy as np

learn_model = hiddenmarkovautoregressive.LearnModel(
    3,1)

print(learn_model.get_hn_params())