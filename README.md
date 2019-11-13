# JIRA-assignment-
JIRA assignment for Xccelerated interview.
The assignment of this interview required to development a solution for "Issue Resolution Time" problem. 
For building a prediction model was decided to use Multi Linear Regression.
The reason for this choice was based on the amount of possible feature that explain the data variation and the model output type which was suppose to be be exact date.

Another possible approach for the following problem could be using Classification algorithms.
Instead of predicting the exact date we could assign a "Resolution Time" class to an issue.
This solution could be a back-up plan.

# Feature Extraction
First of all, before applying any ML algorithm, it is required to extract the sufficient features from a collected data.
The data was provided: ``` avro-issues.csv    avro-transitions.csv   avro-daycounts,csv ```
For building ML model I have generated of list of features, which could explain data variation.
#### These features were genereted:
``` new_feature_list.csv ```
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
```
	Where classes are:
	0 - less than 1 day
	1 - 1-5 days
	2 - 5-10 days
	3 - 10-50 days
	4 - more than 50 days 
```

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
```
	Where priority is:			Where issue-type is:	 
	0 - Trival				0 - Wish
	1 - Low					1 - Task
	2 - Minor				2 - Subtask
	3 - Normal				3 - Test
	4 - Major				4 - Improvement
	5 - High				5 - Bug
	6 - Critical
	7 - Urgent
	8 - Blocker
```
There could be more feature generated, however due to time restrictions I could not generate more. 
# Getting Started
The assignment required to integrate my prediction model to REST API. 
Install following packages(if they are missing) before running the code...

## Installation 
```bash
pip install -U scikit-learn
pip install -U Flask
pip install flask-restful
pip install pickle-mixin 
pip install seaborn
pip install matplotlib
```
Download the Project zip file and use Command prompt to get to project directory.

# Running the tests
For demonstation puproses, the model was already trained and saved in the same folder with the name
``` finalized_model.sav ``` and ``` finalized_model(without-outliers).sav  ```
Therefore, training and testing the model is not required. Although, if you want to do that, following commands explaining you how. 

## Model Training
Training of a Multi Linear Regression model for prediction the resolution date for the issue. 
After model is trained the Pickle operation is used to serialize the model using following command:

```python
python build_MLR_model.py 
```


## Model Testing

To predict the resolution date, it is required to deserialize the creted model and run prediction algorithm.
It is done using following command:

```python
python prediction_MLR.py
```

## Assignment Results
The main assignment required me to integrate the prediction to a REST API with, which returns JSON response.
To start the program type following command in the Command prompt

```python
python JIRA_restful_api.py
```
For sending requests both Postman or Command prompt can be used. (Open additional cmd window for the requests)


Request Example:

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

Request Example:

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

1. Clean data from more outliers 
2. Build separate models based on classes (requires more data)
3. Use different approach of predicting resoluiton time (Use classification models)
4. Check the amount of open task at the (More open, more time takes to resolve)
5. Check the day of week for specific date (Less likely to be resolved on weekends)
6. Check reporters reputatuion (Calculate from average counts and comments)
7. Check the amount of tasks (Assignee has on the moment)
8. Use NLP for Tone/Sentiment Analysis (To determine importance or urgency)
