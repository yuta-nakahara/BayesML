from bayesml import gaussianmixture
import time

gen_model = gaussianmixture.GenModel(
    c_degree=2,
    c_num_classes=3,
    h_alpha_vec=100,
    h_kappas=0.01,
    h_nus=5,
    seed=0,
)
gen_model.gen_params()
gen_model.visualize_model()
x,z = gen_model.gen_sample(1000)


learn_model = gaussianmixture.LearnModel(
    c_degree=2,
    c_num_classes=5,
    h0_alpha_vec=0.0001,
    seed=0,
)
tmp_sum = 0.0
for i in range(100):
    learn_model.reset_hn_params()
    start = time.perf_counter()
    learn_model.update_posterior(x)
    end = time.perf_counter()
    tmp_sum += (end - start)

print(tmp_sum/100)
# learn_model.visualize_posterior()
