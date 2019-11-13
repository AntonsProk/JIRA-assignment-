
import pandas as pd
def resolve_planning_assistance(date):
    df=pd.read_csv("full-prediction-list.csv",sep=",")
    return_list=[]  

    prediction_list=df["prediction"]
    key_list=df["key"]  

    for  i in range (len(df["prediction"])):
        if pd.to_datetime(prediction_list[i])>pd.to_datetime(date):
            return_list.append({'issue':key_list[i],'predicted_resolution_date': prediction_list[i]})
            
    return return_list

#Test
'''
result=resolve_planning_assistance('2018-04-16 03:21:58.193056552+00:00')
print(result)
'''