import numpy as np
import pandas as pd

# --------------
# Reading train.csv file 
train_fpath_1 = r'/Users/mac/Desktop/training_set_processed.csv'
train_set = pd.read_csv(train_fpath_1)
train_fpath_2 = r'/Users/mac/Desktop/prediction_set.csv'
predict_set = pd.read_csv(train_fpath_2)
# --------------

# List of columns that are to dropped from the X_train and X_test
cols_drop = ['Id', 'ProductId', 'UserId', 'Time', 'Summary', 'Text']

# Dropping columns that will not be used for modelling
data = train_set.drop(cols_drop, axis=1)
predict_data = predict_set.drop(cols_drop, axis=1)

# Creating the X and y variables for training set and droping score 
X_t = data.drop('Score', axis=1)
y_t = data['Score']

# Creating X variable for prediciton set and droping score 
X_p = predict_data.drop('Score', axis=1)

# --------------
# Scaling the columns of both 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_t = scaler.fit_transform(data)
X_p = scaler.transform(predict_data)

# Printing the shape of X_train and X_test
print(data.shape)
print(predict_data.shape)

# --------------
# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.20, random_state=42)

# ----------------

# Importing library for linear regression
from sklearn.linear_model import LinearRegression
# Importing library for classification report 
from sklearn.metrics import classification_report
# Importing library for mean_squared error 
from sklearn.metrics import mean_squared_error 
# Importing sk learn for cross validation
from sklearn.model_selection import cross_val_score

# fitting the linear regression model with X_train and y_train
model = LinearRegression().fit(X=X_train,y=y_train)
# Predicting the testing set 
y_pred_test = model.predict(X_test)
# Rounding the test to the nearest integar
y_pred_test = pd.DataFrame(y_pred_test).apply(lambda x: round(x))
# Converting test to numpy array to take care of outliers
y_pred_test= np.array(y_pred_test)
# Changes values greater than 5 to 5 and less than 1 to 1
for i in range(len(y_pred_test)):
    if y_pred_test[i]<1:
        y_pred_test[i]=1
    elif y_pred_test[i]>5:
        y_pred_test[i]=5

# -----------------
# Tools to assess performance of the model
# Printing the classification report 
print(classification_report(y_test, y_pred_test))
# Printing the mean squared error 
print(mean_squared_error(y_test, y_pred_test))
# Finding the score of each fold of the cross validation
scores = cross_val_score(model, X_t, y_t, cv = 10)
# Printing the overall accuracy
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

# --------------------

# Predicting the score for the prediction set 
y_p= model.predict(X_p)
# Rounding the prediction to the nearest integar
y_p = pd.DataFrame(y_p).apply(lambda x: round(x))
# Converting prediction to numpy array to take care of outliers
y_p= np.array(y_p)

# Changes values greater than 5 to 5 and less than 1 to 1 
for i in range(len(y_test)):
    if y_test[i]<1:
        y_test[i]=1
    elif y_test[i]>5:
        y_test[i]=5

# Creating a test_DF dataframe from y_test to store as csv file
pd.DataFrame(y_test)
print(y_test)
# Appending it to prediction csv given to us 
df = pd.read_csv("predication.csv")
df['Score']= y_test
# Saving results to a csv file
df.to_csv("attempt.csv")