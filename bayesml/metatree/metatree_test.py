from bayesml import metatree
from bayesml import normal
from bayesml import poisson
from bayesml import bernoulli
from bayesml import exponential
from bayesml import categorical
from bayesml import linearregression
import numpy as np
import time

dim_continuous = 2
dim_categorical = 1

gen_model = metatree.GenModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_max_depth=2,
    h_g=0.75,
    SubModel=bernoulli,
    # sub_constants={'c_degree':2},
    # sub_h_params={'h_lambda_mat':np.eye(2)*0.01},
    # sub_h_params={'h_kappa':0.1})
    # sub_h_params={'h_alpha':0.3,'h_beta':0.3}
)
gen_model.gen_params(threshold_type='even')
gen_model.visualize_model(filename='tree.pdf')

x_continuous,x_categorical,y = gen_model.gen_sample(200)

learn_model = metatree.LearnModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_num_children_vec=2,
    c_max_depth=2,
    h0_g=0.75,
    SubModel=bernoulli,
    # sub_constants={'c_degree':2},
    # sub_h0_params={'h0_lambda_mat':np.eye(2)*0.01},
    # sub_h0_params={'h0_kappa':0.1})
    # sub_h0_params={'h0_alpha':0.3,'h0_beta':0.3})
)

learn_model.visualize_posterior(filename='tree2.pdf')

# hn_params = learn_model.get_hn_params()
# hn_params['sub_hn_params']['hn_kappa'] = 0.2
# print(learn_model.get_h0_params())
# learn_model.set_hn_params(sub_hn_params=hn_params['sub_hn_params'])
# print(learn_model.get_hn_params())

start = time.time()
learn_model.update_posterior(x_continuous,x_categorical,y)
end = time.time()

learn_model.calc_pred_dist(np.ones(dim_continuous),np.zeros(dim_categorical,dtype=int))
print(learn_model.make_prediction())
# learn_model.overwrite_h0_params()
# print(learn_model.get_h0_params())
# print(learn_model.get_hn_params())

learn_model.visualize_posterior(filename='tree3.pdf')
print(end-start)

learn_model.estimate_params(filename='tree4.pdf')