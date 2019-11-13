import pandas as pd
import pickle

#path_copy="C:/Users/Antons Prokopenko/Desktop/Assignment/data/"

def PredictMLR(clf,issue_key="empty"):
    df=pd.read_csv("full_JIRA_list(without-outliers).csv", sep=',')
    if issue_key!="empty":
        selected_issue= df.loc[df["key"] == issue_key]
    else:
        selected_issue=df

    features=selected_issue[["issue_type", "priority","reporter_mean_FRT","from_create_to_update","assignee_mean_FCTU"]].values
    prediction=abs(clf.predict(features))
    
    resolution_time= pd.to_timedelta(prediction,unit='s')
    creation_date=pd.to_datetime(selected_issue["created"])
    resolution_date = creation_date + resolution_time
    if issue_key!="empty":
        issue_df=pd.read_csv("full-prediction-list(without-outliers).csv", sep=",")
        if(issue_key in issue_df["key"].values):
            return str(resolution_date.values[0])
        else:
            single_issue_df=pd.DataFrame(data={'key':[issue_key],'prediction':[str(resolution_date.values[0])]})
            #single_issue_df=pd.DataFrame([[issue_key, str(resolution_date.values[0])]], columns=list(['key','prediction']))
            print(len(issue_df))
            issue_df=issue_df.append(single_issue_df)
            print(len(issue_df))
            issue_df.to_csv("full-prediction-list(without-outliers).csv", sep="," , index=False)
            return str(resolution_date.values[0])
    else:
        d={'key':df['key'].values,'prediction':resolution_date}
        full_prediction_list=pd.DataFrame(data=d)
        full_prediction_list.to_csv("full-prediction-list(without-outliers).csv", sep="," , index=False)
        return (resolution_date.values)


# load the model from disk
filename = 'finalized_model(without-outliers).sav'
loaded_model = pickle.load(open(filename, 'rb'))
date=PredictMLR(loaded_model)
print(date)