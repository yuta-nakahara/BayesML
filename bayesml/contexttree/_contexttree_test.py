import numpy as np
import bayesml.contexttree as model

gen_model = model.GenModel(c_k=2,c_d_max=3,h_g=0.75)
# print(gen_model.get_params())
gen_model.gen_params()
gen_model.visualize_model(filename='tmp1.gv')
# params = gen_model.get_params()

x = gen_model.gen_sample(30)
print(x)

learn_model_1 = model.LearnModel(2,3,h0_g=0.75)
learn_model_2 = model.LearnModel(2,3,h0_g=0.75)

learn_model_1.update_posterior(x)
learn_model_1.visualize_posterior(filename='tmp2.gv')
# learn_model.estimate_params(filename='tmp3.gv')

for i in range(30):
    learn_model_2.pred_and_update(x[:i+1])
learn_model_2.visualize_posterior(filename='tmp3.gv')