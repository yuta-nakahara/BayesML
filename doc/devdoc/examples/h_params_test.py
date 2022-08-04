from bayesml import bernoulli

model = bernoulli.GenModel(h_alpha=0.5,h_beta=0.3)
params = model.get_params()
model.save_params('tmp.pkl')
print(params)

model_2 = bernoulli.GenModel(theta=0.9,h_alpha=1.0,h_beta=1.0)
print(model_2.get_params())
# model_2.set_h_params(*h_params.values())
# print(model_2.get_h_params())

model_2.load_params('tmp.pkl')
print(model_2.get_params())
