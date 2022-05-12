import os
import sys
sys.path.insert(0, os.path.abspath('..'))


from bayesml import bernoulli

gen_model = bernoulli.GenModel()
# model.visualize_model()

X = gen_model.gen_sample(20)

print(X)

learn_model = bernoulli.LearnModel()

learn_model.update_posterior(X)

learn_model.visualize_posterior()

