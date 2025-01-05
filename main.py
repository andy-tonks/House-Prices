import pandas as pd
import numpy as np


#ADDING THIS NEW COMMENT _ TO CHECK COMMIT IN GITHUB


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

fp = 'melb_data.csv'
data1 = pd.read_csv(fp)


data1 = data1.dropna(axis=0)

# Create a model to predict the price!!!!-------------------------------

y = data1.Price #series data set from dot notation, use colum for prediction target.

data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] #features list of colums.
x = data1[data_features]

data1 = DecisionTreeRegressor(random_state=1)
data1.fit(x,y)
# Predict the price from model -----------------------------------------
print("Making predictions for following 5 houses")
print(x.head())
print("Predictions Are: ")
print(data1.predict(x.head()))
print()

#Validate data ---------------------------------------------------------
print("MAE Prediction accuracy:- ")
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)
# Define model
data1 = DecisionTreeRegressor()
# Fit model
data1.fit(train_X, train_y)
# get predicted prices on validation data
val_predictions = data1.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#-----------------------------------------------------------------

#underfitting and overfitting the models check------------------

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#------------------------------------------------------------------
# Random forests 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
print("Random Forest prediction:- ")
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

#------------------------------------------------------------------











print()


