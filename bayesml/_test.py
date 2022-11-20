from bayesml import contexttree

gen_model = contexttree.GenModel(c_k=2,c_d_max=3,h_g=0.75)
gen_model.gen_params()
gen_model.visualize_model()
x = gen_model.gen_sample(500)
learn_model = contexttree.LearnModel(c_k=2,c_d_max=3)
learn_model.update_posterior(x)
# learn_model.visualize_posterior()
