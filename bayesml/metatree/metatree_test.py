from bayesml import metatree
import numpy as np
import time

dim_continuous = 0
dim_categorical = 5

gen_model = metatree.GenModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_max_depth=5,
    h_g=1.0,
    sub_h_params={'h_alpha':1.0,'h_beta':1.0},
    seed=0,
)
gen_model.gen_params()
true_root = gen_model.get_params()
x_continuous,x_categorical,y = gen_model.gen_sample(400)

learn_model = metatree.LearnModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_max_depth=5,
)
learn_model.set_h0_params(h0_metatree_list=[true_root['root']])
learn_model.set_h0_params(h0_g=0.5)

time_vec = np.empty(100)
for i in range(100):
    learn_model.reset_hn_params()
    start = time.perf_counter()
    learn_model.update_posterior(
        x_continuous,
        x_categorical,
        y,
        alg_type='given_MT',
    )
    end = time.perf_counter()
    time_vec[i] = end-start
print(time_vec.sum()/100*1000)
print(np.mean(time_vec)*1000)
print(np.std(time_vec)*1000)
