from bayesml import metatree
from bayesml import normal
import numpy as np

gen_model = metatree.GenModel(3,3,h_g=0.7)
print(gen_model.get_h_params())
