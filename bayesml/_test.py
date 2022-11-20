from bayesml import bernoulli

gen_model = bernoulli.GenModel(theta=0.7)

gen_model.visualize_model()

x = gen_model.gen_sample(sample_size=20)

learn_model = bernoulli.LearnModel()

learn_model.visualize_posterior()

learn_model.update_posterior(x)
learn_model.visualize_posterior()

print(learn_model.estimate_params(loss='squared'))
print(learn_model.estimate_params(loss='abs'))
print(learn_model.estimate_params(loss='0-1'))