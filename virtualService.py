from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

savedModel = load('pkmnClassifierModel.pkl')

def classify(id):
    inputData = np.array(id)
    inputDataReshaped = inputData.reshape(1, -1)
    prediction = savedModel.predict(inputDataReshaped).tolist()
    return prediction
