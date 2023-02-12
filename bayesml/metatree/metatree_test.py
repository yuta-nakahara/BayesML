from bayesml import metatree
import time

dim_continuous = 1
dim_categorical = 1

gen_model = metatree.GenModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_max_depth=3,
    h_g=0.9,
    seed=0,
)
gen_model.gen_params()
gen_model.visualize_model(filename='model.gv')
x_continuous,x_categorical,y = gen_model.gen_sample(500)

learn_model = metatree.LearnModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_max_depth=3,
)
start = time.time()
learn_model.update_posterior(
    x_continuous,
    x_categorical,
    y,
    random_state=0
)
end = time.time()
print(end-start)
learn_model.visualize_posterior(filename='posterior2.gv')