# JIRA-assignment-
JIRA assignment for Xccelerated interview 

# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Installation 
```bash
pip install -U scikit-learn
pip install -U Flask
pip install flask-restful
pip install pickle-mixin 
pip install seaborn
pip install matplotlib
```
# Feature Extraction
For building ML model I have generated of list of features, which could explain data spread.
#### These features were genereted:

* full_resolution_time - Time required to resolve the issues (applicable to "Resolved issues")
* from_last_update - The time from last status update to issue resolution (applicable to "Resolved issues")
* from_create_to_update - The time from moment of creation until last update
* reporter_mean_FRT - Average full resolution time per reporter
* reporter_mean_FLU - Average time from last update to issue resolution per reporter
* reporter_mean_FCTU - Average time from creation to last update per reporter
* assignee_mean_FRT -	Average full resolution time per assignee
* assignee_mean_FLU -	Average time from last update to issue resolution per assignee
* assignee_mean_FCTU - Average time from creation to last update per assignee
* num_of_trans -  Number of transitions required to get to the last updated status
* classes_by_frt - Class under which issue falls based on full resolution time (used only for classifiers)
* classes_by_flu - Class under which issue falls based on time from last update to issue resolution (used only for classifiers)
* classes_by_fctu - Class under which issue falls based on time from creation to last update (used only for classifiers)
	 
#### These features were transfered from avro-issue.csv and avro-transitions.csv:
	 
* created - The date when issues is created, used to calculate final resolution date
* issue_type - The type of the issue (Categorized)
* priority - The priority of the issue (Categorized)
* description_length - the length of the issue description
* summary_length - the length of the issue summary
* watch_count - the number of the issue watchers
* comment_count - the number of the issue comments
* reporter - the name of the reporter
* key - the ID of the issue key

# Running the tests

## Building the model
Training of a Multi Linear Regression model for prediction the resolution date for the issue. 

```python
python buildMLR.py path
```
After model is trained the Pickle operation is used to serialize the model. 

## Testing

To predict the resolution date, it is required to deserialize the creted model and run prediction algorithm.

```python
python predictMLR.py {issue-key}
```

## Results
The main assignment required me to integrate the prediction to a REST API with, which returns JSON response

```python
python JIRA_prediction.py
```
For calling the command both Command promp and Postman can be used.

Call Example:

	GET /issue/{issue-key}/resolve-prediction
* Command prompt ``` curl http://127.0.0.1:5000/issue/AVRO-1333/resolve-prediction```
* Postman ``` GET 127.0.0.1:5000/issue/AVRO-1333/resolve-prediction```

Response Example:
```bash
{
	'issue' : 'AVRO-1333',
	'predicted_resolution_date' : '2013-09-07T09:24:31.761+0000'
}
```
Additionally, I was asked to build "Release planning Assistance" , which takes as the input valid date and respond with all issues that  are not resolved at the time of calling. 

Call Example:

	GET /release/{date}/resolved-since-now
* Command prompt 
``` curl http://127.0.0.1:5000/release/2013-05-27T09:33:23.123+0200/resolved-since-now```
* Postman 
``` GET 127.0.0.1:5000/release/2013-05-27T09:33:23.123+0200/resolved-since-now```

Response Example:
```bash
{
     'now' : '2013-05-27T09:33:23.123+0200',
     'issues' : [
         {
              'issue' : 'AVRO-1333',
              'predicted_resolution_date' : '2013-09-07T09:24:31.761+0000'
         },
         {
              'issue' : 'AVRO-1335',
              'predicted_resolution_date' : '2013-09-12T09:24:31.761+0000'
         }
      ]
}
```

# Future Improvements
Additional recommendation to improve the accuracy of the predicion model

## Points of Improvement

1. Clean data from outliers 
2. Build separate models based on classes (requires more data)
3. Use different approach of predicting resoluiton time (Use classification models)
4. Check the amount of open task at the (More open, more time takes to resolve)
5. Check the day of week for specific date (Less likely to be resolved on weekends)
6. Check reporters reputatuion (Calculate from average counts and comments)
7. Check the amount of tasks (Assignee has on the moment)
8. Use NLP for Tone/Sentiment Analysis (To determine importance or urgency)
