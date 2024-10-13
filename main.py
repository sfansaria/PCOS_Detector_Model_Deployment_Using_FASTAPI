
#import libraries
import uvicorn
from fastapi import FastAPI
from model import PCOS_Detector
import numpy as np
import pickle
import pandas as pd

#Create the app object

app = FastAPI()
pickle_in = open("/home/saba/Documents/pcos_prediction_project_supervised_learning_project/pcos_classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

#Index Route opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

#Route with a single parameter, returns the parameter within a message
#Located at: http://127.0.0.1:8000/AnyNameHere

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to discover the PCOS Machine Learning Model Deployment by':  f'{name}'}


@app.post('/predict')
def predict_pcos(data:PCOS_Detector):
    data = data.dict()
    Age = data['Age']
    Weight = data['Weight']
    Height = data['Height']
    #BMI = data['BMI']
    Blood_Group = data['Blood_Group']
    Pulse_rate = data['Pulse_rate']
    RR = data['RR']
    Hb = data['Hb']
    Cycle = data['Cycle']
    Cycle_length = data['Cycle_length']
    Marraige_Status = data['Marraige_Status']
    Pregnant = data['Pregnant']
    aborptions = data['aborptions']
    I_beta_HCG = data['I_beta_HCG'] 
    II_beta_HCG = data['II_beta_HCG']
    FSH = data['FSH']
    LH = data['LH']
    #FSH_BY_LH = data['FSH_BY_LH'] 
    Hip = data['Hip']
    Waist = data['Waist']
    #Waist_Hip_Ratio = data['Waist_Hip_Ratio']
    TSH = data['TSH']
    AMH = data['AMH']
    PRL = data['PRL']
    Vit_D3 = data['Vit_D3']
    PRG = data['PRG']
    RBS = data['RBS']
    Weight_gain = data['Weight']
    hair_growth = data['hair_growth']
    Skin_darkening = data['Skin_darkening']
    Hair_loss = data['Hair_loss']
    Pimples = data['Pimples']
    Fast_food = data['Fast_food']
    Reg_Exercise = data['Reg_Exercise']
    BP_Systolic = data['BP_Systolic']
    BP_Diastolic = data['BP_Diastolic']
    Follicle_num_l = data['Follicle_num_l']
    Follicle_num_r = data['Follicle_num_r']
    Avg_F_size_L_mm = data['Avg_F_size_L_mm']
    Avg_F_size_R_mm = data['Avg_F_size_R_mm']
    Endometrium_mm = data['Endometrium_mm'] 

    #print(classifier.predict([[variance, skewness,curtosis,entropy]]))
    prediction = classifier.predict([[Age, Weight, Height, Blood_Group,
                                       Pulse_rate, RR, Hb, Cycle, Cycle_length,
                                         Marraige_Status, Pregnant, aborptions, 
                                         I_beta_HCG, II_beta_HCG, FSH, LH,
                                           Hip, Waist, TSH, AMH, PRL,
                                             Vit_D3, PRG, RBS, Weight_gain, hair_growth,
                                               Skin_darkening, Hair_loss, Pimples, Fast_food, 
                                               Reg_Exercise, BP_Systolic, BP_Diastolic, Follicle_num_l, 
                                               Follicle_num_r, Avg_F_size_L_mm, Avg_F_size_R_mm, Endometrium_mm]])
    #print(prediction)
    if prediction[0] == 1.0:
        prediction = "PCOS Detected"
    else:
        prediction = "PCOS Not Detected"
    return {'Prediction': prediction}
        





#run the api with uvicorn
#Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn main:app --reload