
# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
   
    # Convert the json string to a python dictionary object
    data_dic = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    data_dic_df = pd.DataFrame.from_dict([data_dic])


    

    data_dic_df = data_dic_df[(data_dic_df['Commodities'] == 'APPLE GOLDEN DELICIOUS')]
    dataPrediction = data_dic_df[['Weight_Kg','Sales_Total','Low_Price','High_Price','Total_Qty_Sold','Total_Kg_Sold','Stock_On_Hand','avg_price_per_kg']]
                                
    # ------------------------------------------------------------------------

    return dataPrediction

def readModel(path_to_model:str):

    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):

    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
