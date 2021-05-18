"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""







import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import pickle
# Fetch training data and preprocess for modeling

df_train = pd.read_csv('C:\\Users\\bonol\\OneDrive\\Desktop\\Explore Course Work\\NoteBooks\\project git\\team_ts_4\\Advanced_regression_predict-Starter-Data-2629\\df-train_set.csv')
# Process the data
df_train = df_train[df_train.Commodities == 'APPLE GOLDEN DELICIOUS']  # filter for APPLE GOLDEN DELICIOUS
y_train = df_train['avg_price_per_kg']
df_train[['Date']] = pd.to_datetime(df_train['Date'])
df_train['day'] = df_train['Date'].dt.day
df_train['month'] = df_train['Date'].dt.month
df_train['year'] = df_train['Date'].dt.year


df_train = df_train.drop(['Commodities', 'Date', 'year'], axis=1)





df_train = pd.get_dummies(df_train)
df_train.columns = [col.replace(" ", "_") for col in df_train.columns]
df_train.columns = [col.replace(".", "_") for col in df_train.columns]
df_train.columns = [col.replace("-", "_") for col in df_train.columns]

df_train = df_train[['Container_M4183', 'Province_W_CAPE_BERGRIVER_ETC', 'Size_Grade_1X',
   'Container_EC120', 'Size_Grade_1M', 'Container_EF120', 'day',
   'Size_Grade_2L', 'Container_JG110', 'Size_Grade_2M',
   'Province_EASTERN_CAPE', 'Container_JE090', 'Weight_Kg',
   'Size_Grade_2S', 'Container_IA400', 'Province_NATAL']]



#Fit the model
lm_var = LinearRegression() ##??(normalize=True)
print("Training Model...")
lm_var.fit(df_train, y_train)



# Pickle model for use within our API
save_path = '../assets/trained-models/mlr_model.pkl'
print(f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_var, open(save_path, 'wb'))
