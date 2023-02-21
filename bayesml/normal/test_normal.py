from bayesml import normal
import numpy as np

gen_model = normal.GenModel()
x = gen_model.gen_sample(100)

learn_model = normal.LearnModel()

tmp_sum = 0.0
for i in range(100):
    tmp_sum += np.log(learn_model.pred_and_update(x[i],loss='KL').pdf(x[i]))

print(tmp_sum)

learn_model.reset_hn_params()
print(learn_model.update_posterior(x).calc_log_marginal_likelihood())