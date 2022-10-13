from bayesml import hiddenmarkovautoregressive
import numpy as np

gen_model = hiddenmarkovautoregressive.GenModel(
    3,1,h_betas=2)

print(gen_model.get_h_params())