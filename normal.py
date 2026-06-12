import pickle
model = pickle.load(open('car_price_model.pkl', 'rb'))
print(type(model))
