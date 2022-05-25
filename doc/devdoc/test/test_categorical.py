import numpy as np
from bayesml.categorical import GenModel, LearnModel

gen_model = GenModel()
h_params = gen_model.get_h_params()
params = gen_model.get_params()
learn_model = LearnModel()
h0_params = learn_model.get_h0_params()
hn_params = learn_model.get_hn_params()

learn_model.set_h0_params(**h_params)
print('ok1')
learn_model.set_h0_params(**h0_params)
print('ok2')
learn_model.set_h0_params(**hn_params)
print('ok3')
try:
    learn_model.set_h0_params(**params)
    print('!!!!!!!!!!!NG!!!!!!!!!!!!')
except:
    print('ok4')

learn_model.set_hn_params(**h_params)
print('ok5')
learn_model.set_hn_params(**h0_params)
print('ok6')
learn_model.set_hn_params(**hn_params)
print('ok7')
try:
    learn_model.set_hn_params(**params)
    print('!!!!!!!!!!!NG!!!!!!!!!!!!')
except:
    print('ok8')

gen_model.set_h_params(**h_params)
print('ok9')
gen_model.set_h_params(**h0_params)
print('ok10')
gen_model.set_h_params(**hn_params)
print('ok11')
try:
    gen_model.set_h_params(**params)
    print('!!!!!!!!!!!NG!!!!!!!!!!!!')
except:
    print('ok12')

gen_model.set_params(**params)
print('ok13')
try:
    gen_model.set_params(**h_params)
    print('!!!!!!!!!!!NG!!!!!!!!!!!!')
except:
    print('ok14')

import copy
gen_model = GenModel()
x = gen_model.gen_sample(100)
learn_model = LearnModel()
h0_params = copy.deepcopy(learn_model.get_h0_params())
hn_params = copy.deepcopy(learn_model.get_hn_params())
learn_model.update_posterior(x)
if str(hn_params) != str(learn_model.get_hn_params()):
    print('ok15')
else:
    print('!!!!!!!!!!!NG!!!!!!!!!!!!')
learn_model.reset_hn_params()
if str(hn_params) == str(learn_model.get_hn_params()):
    print('ok16')
else:
    print('!!!!!!!!!!!NG!!!!!!!!!!!!')
learn_model.update_posterior(x)
learn_model.overwrite_h0_params()
if str(h0_params) != str(learn_model.get_h0_params()):
    print('ok17')
else:
    print('!!!!!!!!!!!NG!!!!!!!!!!!!')
