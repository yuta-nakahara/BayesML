from bayesml import metatree
from bayesml import normal
import numpy as np

gen_model = metatree.GenModel(3,3,h_g=0.7)
gen_model.gen_params()
gen_model.visualize_model()
