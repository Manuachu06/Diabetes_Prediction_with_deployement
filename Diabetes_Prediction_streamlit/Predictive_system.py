import numpy as np
import pickle

loaded_model = pickle.load(open('D:/Diabetes_Prediction_streamlit/trained_model_sav' , 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)

#changing input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#Standardized input_data

#std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = loaded_model.predict(input_data_reshaped)
#print(prediction)

if prediction[0]==0:
  print('The person is not having diabetic')
else:
  print('The person is diabetic')

