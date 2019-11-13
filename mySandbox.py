#%%
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
#from keras.models import Sequential
#from keras.layers import Dense, Flatten, MaxPool1D, Conv1D, LSTM
import pandas as pd
import seaborn as sns; sns.set(style='ticks', color_codes=True)
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import math
import pickle

def deleteKeys():
    df=pd.read_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/data/avro-transitions.csv", sep=',')
    df_keys=pd.read_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/Reopened_keys.csv", sep=',')
    key_list=df_keys['key'].values
    new_list=[]
    for i in key_list:
        new_list.append(df.loc[df['key']==i].index[0])
    df=df.drop(new_list)
    df.to_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/data/avro-transitions(no_reopened).csv", sep=",", index=False)
    return

def PredictMLR(clf,key):
    df=pd.read_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/data/full_JIRA_list.csv", sep=',')
    selected_task= df.loc[df["key"] == key]
    features=selected_task[["issue_type", "priority","reporter_mean_FRT","from_create_to_update","assignee_mean_FCTU"]].values
    features=features
    prediction=abs(clf.predict(features))
    
    resolution_time= pd.to_timedelta(prediction,unit='s')
    creation_date=pd.to_datetime(selected_task["created"])
    resolution_date = creation_date + resolution_time
    return resolution_date

def MultipleLinearRegression():
    featurePath="C:/Users/Antons Prokopenko/Desktop/Assignment/data/new_feature_list.csv"
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
    print("Predicted Resolution Time")
    for i in range(len(Yt)):
        print("Error",i, abs(predicted[i]-Yt[i])/86400.0)
    print()
    print(metrics.mean_absolute_error(Yt, predicted))
    print(regr.score(Xt, Yt))

    predictionDF = pd.DataFrame({'Actual': Yt.flatten(), 'Predicted': predicted.flatten()})
    df1 = predictionDF.head(40)
    df1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    return regr

def predictClassifier(index, clf, clf1, clf2):
    dataFile="C:/Users/Antons Prokopenko/Desktop/Assignment/test_dataFrame_list.csv"
    df=pd.read_csv(dataFile, sep=",").fillna(0)
    data_features=df.loc[index][['priority','description_length','watch_count','comment_count','reporter_mean_FRT','reporter_mean_FLU','num_of_trans']].values
    data_features=data_features.reshape((1, 7))
    print()
    print("DT Prediction",clf.predict(data_features))
    print("SVM Prediction",clf1.predict(data_features))
    print("Naive-Bayes",clf2.predict(data_features))

def TrainClassifier(path):
    df=pd.read_csv(path, sep=",").fillna(0)
    
    train, test = train_test_split(df, test_size=0.3,random_state=5)
    X_train=train[['priority','description_length','watch_count','comment_count','reporter_mean_FRT','reporter_mean_FLU','num_of_trans']].values#,'summary_length'
    y_train=train[['classes_by_frt','classes_by_flu']].values
    y_train_frt=train[['classes_by_frt']].values
    y_train_flu=train[['classes_by_flu']].values
    
    X_test=test[['priority','description_length','watch_count','comment_count','reporter_mean_FRT','reporter_mean_FLU','num_of_trans']].values
    y_test=test[['classes_by_frt','classes_by_flu']].values
    y_test_frt=test[['classes_by_frt']].values
    y_test_flu=test[['classes_by_flu']].values

    #DecisonTreeClassifier()    
    clf1 = DecisionTreeClassifier()
    clf1 = clf1.fit(X_train,y_train)
    y_pred1 = clf1.predict(X_test)
    accuracy = np.sum(np.equal(y_test, y_pred1))/float(y_test.size) 
    print("Accuracy Decision Tree:", accuracy)

    print()
    clf2_1 = svm.SVC(gamma='scale')
    clf2_2 = svm.SVC(gamma='scale')
    clf2_1.fit(X_train, y_train_frt.ravel())
    clf2_2.fit(X_train, y_train_flu.ravel())
    y_pred_frt=clf2_1.predict(X_test)
    y_pred_flu=clf2_2.predict(X_test)
    accuracy_frt=metrics.accuracy_score(y_test_frt, y_pred_frt)
    accuracy_flu=metrics.accuracy_score(y_test_flu, y_pred_flu)
    print("Accuracy FRT Support Vector Machine:", accuracy_frt)
    print("Accuracy FLU Support Vector Machine:", accuracy_flu)
    print("~AVERAGE SVM = ", (accuracy_frt+accuracy_flu)/2)

    print()
    clf3_1= GaussianNB()
    clf3_2 = GaussianNB()
    clf3_1.fit(X_train, y_train_frt.ravel())
    clf3_2.fit(X_train, y_train_flu.ravel())
    y_pred_frt=clf3_1.predict(X_test)
    y_pred_flu=clf3_2.predict(X_test)
    accuracy_frt=metrics.accuracy_score(y_test_frt, y_pred_frt)
    accuracy_flu=metrics.accuracy_score(y_test_flu, y_pred_flu)
    print("Accuracy FRT Naive-Bayes:", accuracy_frt)
    print("Accuracy FLU Naive-Bayes:", accuracy_flu)
    print("~AVERAGE NB = ", (accuracy_frt+accuracy_flu)/2)
        
    return clf1,clf2_1,clf3_1

def plot_relations():
    df=pd.read_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/data/avro-issues.csv", sep=",")
    fig, axs = plt.subplots(5, 5)
    MyArray=[[df["description_length"],df["summary_length"],df["comment_count"],df["watch_count"],df["vote_count"]],
            [df["description_length"],df["summary_length"],df["comment_count"],df["watch_count"],df["vote_count"]]]
    print(MyArray[1][3].name)
    for i in range(2):
        for y in range(5):
            axs[i][y].scatter(MyArray[i][y],MyArray[i][y])
            axs[i][y].set_xlabel(MyArray[i][y].name)
            axs[i][y].set_ylabel(MyArray[i][y].name)
            axs[i][y].grid(True)
    fig.tight_layout()
    plt.show()

def count_issues():
    df=pd.read_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/data/avro-daycounts.csv", sep=",")
    open_issue=[]
    reopened_issue=[]
    resovled_issue=[]
    inprogress_issue=[]
    closed_issue=[]
    patch_issue=[]
    print(df.columns)
    for index, rows in df.iterrows():
        #print(rows['status'],rows['count'])
        if rows['status']=="Open":
            open_issue.append(rows['count'])
        elif rows['status']=="Closed":
            closed_issue.append(rows['count'])
        elif rows['status']=="Reopened":
            reopened_issue.append(rows['count'])
        elif rows['status']=="Resolved":
            resovled_issue.append(rows['count'])
        elif rows['status']=="In Progress":
            inprogress_issue.append(rows['count'])
        elif rows['status']=="Patch Available":
            patch_issue.append(rows['count'])
    fig=plt.figure(figsize=(10,5))
    plt.plot(open_issue, c="blue",label="open_issue")
    plt.plot(closed_issue, c="red", label="closed_issue")
    plt.plot(reopened_issue, c="green", label="reopened_issue")
    plt.plot(resovled_issue, c="black", label="resovled_issue")
    plt.plot(inprogress_issue, c="orange", label="inprogress_issue")
    plt.plot(patch_issue, c="cyan", label="patch_issue")
    plt.ylabel("Issue count")
    plt.xlabel("Time")
    plt.legend()
    plt.show()


def extract_by_status(issue_status):
    df_i=pd.read_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/code/avro-issues(without-outliers).csv", sep=",")
    df_t=pd.read_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/code/avro-transitions(without-outliers).csv", sep=",")
    #df_i_grouped= df_i.loc[df_i['status'] == issue_status] #FIXME Comment to extract features for all issues
    #df_i_grouped=df_i_grouped.reset_index(drop=True)
    df_i_grouped=df_i #FIXME Uncomment to extract features for all issues
    # * Creating features to use later in Multi Linear Regression
    
    frt=pd.to_datetime(df_i_grouped['resolutiondate']).sub(pd.to_datetime(df_i_grouped['created'])) #From created to resolved 
    flu=pd.to_datetime(df_i_grouped['updated']).sub(pd.to_datetime(df_i_grouped['resolutiondate'])) #From updated to resolved
    fctu=pd.to_datetime(df_i_grouped['updated']).sub(pd.to_datetime(df_i_grouped['created'])) #from created to updated

    priority = df_i_grouped['priority'].replace(['Trivial','Low','Minor','Normal','Major','High','Critical','Urgent','Blocker'],[0,1,2,3,4,5,6,7,8])
    issue_type = df_i_grouped['issue_type'].replace(['Wish','Test','Improvement','New Feature','Sub-task','Task','Bug'],[0,1,2,3,4,5,6])
    #df_t_grouped=df_t.loc[df_t['status'] == issue_status] #FIXME Comment to extract features for all issues
    df_t_grouped=df_t #FIXME Uncomment to extract features for all issues
    dups_keys = df_t_grouped.pivot_table(index=['key'], aggfunc='size')
    key_trans_list=[]
    for keys in df_i_grouped['key'].values: 
        key_trans_list.append(dups_keys[keys])
    key_trans_list=pd.Series(key_trans_list)
        
    # * Creating mean_FRT and mean_FLU variables per user
    reporter_list=df_i_grouped["reporter"].unique()
    reporter_dic={}
    
    for i in reporter_list:
        reporter_DF=df_i_grouped.loc[df_i_grouped["reporter"]==i]
        mean_FRT=(pd.to_datetime(reporter_DF['resolutiondate']).sub(pd.to_datetime(reporter_DF['created'])).dt.total_seconds()).mean()
        mean_FLU=(pd.to_datetime(reporter_DF['updated']).sub(pd.to_datetime(reporter_DF['resolutiondate'])).dt.total_seconds()).mean()
        mean_FCTU=(pd.to_datetime(reporter_DF['updated']).sub(pd.to_datetime(reporter_DF['created'])).dt.total_seconds()).mean()
        reporter_dic[i]=[mean_FRT, mean_FLU,mean_FCTU]

    mean_FRT_list=[]
    mean_FLU_list=[]
    mean_FCTU_list=[]
    for items in df_i_grouped['reporter'].values: 
        mean_FRT_list.append(reporter_dic[items][0])
        mean_FLU_list.append(reporter_dic[items][1])
        mean_FCTU_list.append(reporter_dic[items][2])
    mean_FLU_list=(pd.Series(mean_FLU_list)).fillna(0.0)
    mean_FRT_list=(pd.Series(mean_FRT_list)).fillna(0.0)
    mean_FCTU_list=(pd.Series(mean_FCTU_list)).fillna(0.0)

    assignee_list_noedit=df_i_grouped["assignee"].unique()
    assignee_list=[]
    for i in assignee_list_noedit:
        if isinstance(i, float)==False:
            assignee_list.append(i)
    assignee_dic={}
    for i in assignee_list:
        assignee_DF=df_i_grouped.loc[df_i_grouped["assignee"]==i]
        mean_FRT=((pd.to_datetime(assignee_DF['resolutiondate']).sub(pd.to_datetime(assignee_DF['created']))).dt.total_seconds()).mean()
        mean_FLU=((pd.to_datetime(assignee_DF['updated']).sub(pd.to_datetime(assignee_DF['resolutiondate']))).dt.total_seconds()).mean()
        mean_FCTU=((pd.to_datetime(assignee_DF['updated']).sub(pd.to_datetime(assignee_DF['created']))).dt.total_seconds()).mean()
        assignee_dic[i]=[mean_FRT, mean_FLU,mean_FCTU]
    assignee_mean_FRT_list=[]
    assignee_mean_FLU_list=[]
    assignee_mean_FCTU_list=[]
    

    for item in df_i_grouped['assignee'].values:
        if isinstance(item, float):
            assignee_mean_FRT_list.append(0.0)
            assignee_mean_FLU_list.append(0.0)
            assignee_mean_FCTU_list.append(0.0)
        else:
            assignee_mean_FRT_list.append(assignee_dic[item][0])
            assignee_mean_FLU_list.append(assignee_dic[item][1])
            assignee_mean_FCTU_list.append(assignee_dic[item][2])
    assignee_mean_FLU_list=(pd.Series(assignee_mean_FLU_list)).fillna(0.0)
    assignee_mean_FRT_list=(pd.Series(assignee_mean_FRT_list)).fillna(0.0)
    assignee_mean_FCTU_list=(pd.Series(assignee_mean_FCTU_list)).fillna(0.0)

    frt_list=(frt.dt.total_seconds()).fillna(0.0)
    flu_list=(flu.dt.total_seconds()).fillna(0.0)
    fctu_list=(fctu.dt.total_seconds()).fillna(0.0)
    
    classes_frt=[]
    for i in frt_list:
        if i/86400<=1.0:
            classes_frt.append(0)
        if i/86400>1.0 and i/86400<=5.0:
            classes_frt.append(1)
        if i/86400>5.0 and i/86400<=10.0:
            classes_frt.append(2)
        if i/86400>10.0 and i/86400<=50.0:
            classes_frt.append(3)
        if i/86400>50.0:
            classes_frt.append(4)

    classes_flu=[]
    for i in flu_list:
        if i/86400<=1.0:
            classes_flu.append(0)
        if i/86400>1.0 and i/86400<=5.0:
            classes_flu.append(1)
        if i/86400>5.0 and i/86400<=10.0:
            classes_flu.append(2)
        if i/86400>10.0 and i/86400<=50.0:
            classes_flu.append(3)
        if i/86400>50.0:
            classes_flu.append(4)

    classes_fctu=[]
    for i in fctu_list:
        if i/86400<=1.0:
            classes_fctu.append(0)
        if i/86400>1.0 and i/86400<=5.0:
            classes_fctu.append(1)
        if i/86400>5.0 and i/86400<=10.0:
            classes_fctu.append(2)
        if i/86400>10.0 and i/86400<=50.0:
            classes_fctu.append(3)
        if i/86400>50.0:
            classes_fctu.append(4)
    
    # * Adding features to dictionary to write it to .csv
    #
    d = {'created': df_i_grouped['created'],
        'full_resolution_time':frt_list,
        'from_last_update':flu_list,
        'from_create_to_update':fctu_list,
        'issue_type':issue_type,
        'priority':priority,
        'description_length': df_i_grouped['description_length'],
        'summary_length':df_i_grouped['summary_length'],
        'watch_count': df_i_grouped['watch_count'],
        'comment_count': df_i_grouped['comment_count'],
        'reporter_mean_FRT': mean_FRT_list,
        'reporter_mean_FLU': mean_FLU_list,
        'reporter_mean_FCTU': mean_FCTU_list,
        'reporter': df_i_grouped['reporter'],
        'key': df_i_grouped['key'],
        'num_of_trans': key_trans_list,
        'classes_by_frt': classes_frt,
        'classes_by_flu': classes_flu,
        'classes_by_fctu': classes_fctu,
        'assignee_mean_FRT': assignee_mean_FRT_list,
        'assignee_mean_FLU': assignee_mean_FLU_list,
        'assignee_mean_FCTU': assignee_mean_FCTU_list}
    
    new_df=pd.DataFrame(data=d)   
    return new_df

#%% Extracting the Train Data
new_dataFrame=extract_by_status('Resolved')
new_dataFrame.to_csv("C:/Users/Antons Prokopenko/Desktop/Assignment/code/full_JIRA_list(without-outliers).csv", sep=",", index=False)

#%% Plotting the relations
#sns_plot=sns.pairplot(new_dataFrame,x_vars=["full_resolution_time"],y_vars=["issue_type", "priority","reporter_mean_FRT","from_create_to_update"])
#sns_plot.savefig("C:/Users/Antons Prokopenko/Desktop/Assignment/graphs/Wish.png")
#plt.show()

#%% Test Classification algorithms
#clf1,clf2,clf3=TrainClassifier("C:/Users/10268717/Documents/Python Scripts/data/new_feature_list.csv")
#test_dataFrame=extract_by_status('Open')
#predictClassifier(12,clf1,clf2,clf3)
#test_dataFrame.to_csv("C:/Users/10268717/Documents/Python Scripts/data/test_dataFrame_list.csv", sep=",", index=False)

#%% Serialize and Deserialize MLP model
# save the model to disk
#filename = 'C:/Users/Antons Prokopenko/Desktop/Assignment/code/finalized_model.sav'
#model=MultipleLinearRegression()
#pickle.dump(model, open(filename, 'wb'))

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#date=PredictMLR(loaded_model,'AVRO-2032')
#print(date)

#%% Deleting Outliers
#deleteKeys()