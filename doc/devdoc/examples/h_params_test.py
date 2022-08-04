from bayesml import normal as bayesml_model
import numpy as np

h0_params = {'h0_m':2,'h0_kappa':1,'h0_alpha':3,'h0_beta':4}

print('Gen to Learn 1')
model = bayesml_model.GenModel()
print(model.get_h_params())
model.save_h_params('tmp.pkl')

model_2 = bayesml_model.LearnModel(**h0_params)
print(model_2.get_h0_params())
model_2.load_h0_params('tmp.pkl')
print(model_2.get_h0_params())

print('Gen to Learn 2')
model = bayesml_model.GenModel()
print(model.get_h_params())
model.save_h_params('tmp.pkl')

model_2 = bayesml_model.LearnModel(**h0_params)
print(model_2.get_hn_params())
model_2.load_hn_params('tmp.pkl')
print(model_2.get_hn_params())

print('Learn to Gen 1')
model_2 = bayesml_model.LearnModel(**h0_params)
print(model_2.get_h0_params())
model_2.save_h0_params('tmp.pkl')

model = bayesml_model.GenModel()
print(model.get_h_params())
model.load_h_params('tmp.pkl')
print(model.get_h_params())

print('Learn to Gen 2')
model_2 = bayesml_model.LearnModel(**h0_params)
print(model_2.get_hn_params())
model_2.save_hn_params('tmp.pkl')

model = bayesml_model.GenModel()
print(model.get_h_params())
model.load_h_params('tmp.pkl')
print(model.get_h_params())
