"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
import numpy as np
import pandas as pd
from sklearn import linear_model  # Scikit learn library that implements generalized linear models

df_train = pd.read_csv('Advanced_regression_predict-Starter-Data-2629/df-train_set.csv') # import the train data for exploration
df_train['Date'] = pd.to_datetime(df_train['Date'])  # change the date column from string datatype
new_df_train = df_train[df_train.Commodities == 'APPLE GOLDEN DELICIOUS']  # filter for APPLE GOLDEN DELICIOUS
## Dummy variable encoding
df_dummies = pd.get_dummies(new_df_train)

#lets make sure that all the column names have underscores instead of whitespaces
df_dummies.columns = [col.replace(" ", "_") for col in df_dummies.columns]
df_dummies.columns = [col.replace(".", "_") for col in df_dummies.columns]
df_dummies.columns = [col.replace("-", "_") for col in df_dummies.columns]

# Correlations and Model Structure
# We will now move our target variable to be the last column of our table for ease of reference
column_titles = [col for col in df_dummies.columns if col!= 'avg_price_per_kg'] + ['avg_price_per_kg']
df_dummies = df_dummies.reindex(columns=column_titles)

# Fitting the model
#Lets try fitting the model as it currently is using statsmodels.ols to see what results the OLS model summary gives us.

#lets import the packages and generate the regression string
from statsmodels.formula.api import ols

# copy the DataFrame with all of the columns:
dfm = df_dummies.copy()

# create a ariable for the dependent variable:
y_name = 'avg_price_per_kg'
# create a variable for the independent variable :we will use all the columns in the model DataFrame
X_names = [col for col in dfm.columns if col != y_name]

# gererate the regression string
formula_str = y_name+" ~ "+" + ".join(X_names)

# Lets fit the model using the dfm dataframe that we just created
model=ols(formula=formula_str, data=dfm)
fitted = model.fit()

#we can now print the summary
#Let us now create a new df_dummies table and drop the previous one in order not to assume any relationships beween our features.
df_dummies = pd.get_dummies(new_df_train, drop_first=True)

# Lets make sure that all the column names have underscores instead of whitespaces, fullstops and dashes
df_dummies.columns = [col.replace(" ", "_") for col in df_dummies.columns]
df_dummies.columns = [col.replace(".", "_") for col in df_dummies.columns]
df_dummies.columns = [col.replace("-", "_") for col in df_dummies.columns]

# Reorder columns with the dependent avg_price_per_kg variable as the last column
column_titles = [col for col in df_dummies.columns if col !=
                 'avg_price_per_kg'] + ['avg_price_per_kg']
df_dummies = df_dummies.reindex(columns=column_titles)

#We now have 35 columns instead of 39 that can possibly help us create a model to predict the average price per kg.
# We'll keep the model DataFrame, but only specify the columns we want to fit this time
X_names = [col for col in df_dummies.columns if col != y_name]

#generate the regression string
formula_str = y_name+' ~ '+'+'.join(X_names)

# Lets fit the model using the dfm dataframe
model = ols(formula=formula_str, data=dfm)
fitted = model.fit()

# Variable Selection by Correlation and Significance
#Lets start with computing the correlations between dependant variable and the independant variables
corrs = df_dummies.corr()['avg_price_per_kg'].sort_values(ascending=False)

# lets build a list of p-values and correlations coefficients for each variable
from scipy.stats import pearsonr

# Build a dictionary of correlation coefficients and p-values
dict_cp = {}

column_titles = [col for col in corrs.index if col != 'avg_price_per_kg']
for col in column_titles:
    p_val = round(pearsonr(df_dummies[col], df_dummies['avg_price_per_kg'])[1], 6)
    dict_cp[col] = {'Correlation_Coefficient': corrs[col],
                    'P_Value': p_val}

df_cp = pd.DataFrame(dict_cp).T
df_cp_sorted = df_cp.sort_values('P_Value')

# The dependent variable remains the same:
y_data = df_dummies[y_name]  #we prviously set y_name as avg_price_per_kg'

# Model building - Independent Variable (IV) DataFrame
X_names = list(df_cp[df_cp['P_Value'] < 0.05].index)
X_data = df_dummies[X_names]

# As before, we create the correlation matrix
# and find rows and columns where correlation coefficients > 0.9 or <-0.9
corr = X_data.corr()
r, c = np.where(np.abs(corr) > 0.9)

# We are only interested in the off diagonal entries:
off_diagonal = np.where(r != c)

# Lets do a ols summary with a new subset that doesnt contain the Low price and high price variable
X_remove = ['Low_Price', 'High_Price']
X_corr_names = [col for col in X_names if col not in X_remove]

# Create our new OLS formula based-upon our smaller subset
formula_str = y_name+' ~ '+' + '.join(X_corr_names)

# Fit the OLS model using the model dataframe
model = ols(formula=formula_str, data=dfm)
fitted = model.fit()

#Due to the changes made to our copy above, let us now make those changes to our actual data x_data and x names.
X_data = X_data.drop(['Low_Price', 'High_Price'], axis = 1 )
X_remove = ['Low_Price', 'High_Price']
X_names = [col for col in X_names if col not in X_remove]

# Variable Selection by Variance Thresholds
# Prepare the data and import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_data)
X_normalize = pd.DataFrame(X_scaled, columns=X_data.columns)

# Create VarianceThreshold object
selector = VarianceThreshold(threshold=0.03)

# Use the object to apply the threshold on data
selector.fit(X_normalize)
# Select new columns
X_new = X_normalize[X_normalize.columns[selector.get_support(indices=True)]]

# Save variable names for later
X_var_names = X_new.columns

# View first few entries

#  Model prediction of Average price per kg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Pre-processing
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_data,
                                                    test_size=0.10,
                                                    shuffle=False)
# Get training and testing data for variance threshold model
X_var_train = X_train[X_var_names]
X_var_test = X_test[X_var_names]
# Get training and testing data for correlation threshold model

#Fit the model

lm_var = LinearRegression() ##??(normalize=True)
print("Training Model...")
lm_var.fit(X_var_train, y_train)



# Pickle model for use within our API
save_path = '../assets/trained-models/mlr_model.pkl'
print(f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_var, open(save_path, 'wb'))
