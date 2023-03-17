# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pickle

# Loading the saved model 

loaded_model = pickle.load(open('model.pkl','rb'))
input_data = (50,10,30,2.5,210,120,35.5,19.5,100,156,220,95,110,4,10,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person has not churned')
else:
  print('The person has churned')