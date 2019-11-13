import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
import pickle

#path_copy="C:/Users/Antons Prokopenko/Desktop/Assignment/data/"

def MultipleLinearRegression():
    
    featurePath="new_feature_list(without-outliers).csv" #Using train dataset without outliers, since it showed higher accuracy.
    featureDF=pd.read_csv(featurePath, sep=",").fillna(0)
    train, test = train_test_split(featureDF, test_size=0.33,random_state=7)

    X=train[["issue_type", "priority","reporter_mean_FRT","from_create_to_update","assignee_mean_FCTU"]].values
    Y=train["full_resolution_time"].values
    
    Xt=test[["issue_type", "priority","reporter_mean_FRT","from_create_to_update","assignee_mean_FCTU"]].values
    Yt=test["full_resolution_time"].values
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(X, Y)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    predicted=regr.predict(Xt)
    print("Predicted Resolution Time (100 datapoints sample)")
    for i in range(len(Yt)):
        print("Error",i, abs(predicted[i]-Yt[i])/86400.0)
    print()

    #Evaluation of the model
    print("Mean_absolute_error",metrics.mean_absolute_error(Yt, predicted)/86400.0,"days")
    print("Regression score",regr.score(Xt, Yt))

    predictionDF = pd.DataFrame({'Actual': abs(Yt.flatten()), 'Predicted': abs(predicted.flatten())})
    
    #Plotting the prediction result of first 40 test samples
    df1 = predictionDF.head(40)
    df1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    return regr

#save the model to disk
#filename = 'finalized_model.sav'
model=MultipleLinearRegression()
#pickle.dump(model, open(filename, 'wb'))