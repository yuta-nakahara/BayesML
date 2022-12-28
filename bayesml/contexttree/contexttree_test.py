from bayesml import contexttree

gen_model = contexttree.GenModel(2,2,h_g=0.75)

gen_model.gen_params()
gen_model.visualize_model(filename='tree.pdf')

x = gen_model.gen_sample(1000)

learn_model = contexttree.LearnModel(2,2,h0_g=0.75,h0_beta_vec=1.01)
learn_model.update_posterior(x)
learn_model.visualize_posterior(filename='tree2.pdf')
learn_model.estimate_params(filename='map_tree.pdf')
h_params = learn_model.get_hn_params()

gen_model2 = contexttree.GenModel(2,3)
gen_model2.set_h_params(h_root=h_params['hn_root'])
gen_model2.gen_params()
gen_model2.visualize_model(filename='tree3.pdf')
gen_model2.set_h_params(h_beta_vec=0.1)
gen_model2.gen_params()
gen_model2.visualize_model(filename='tree4.pdf')
gen_model2.gen_params()
gen_model2.visualize_model(filename='tree5.pdf')
