from flask import Flask,jsonify,request
from flask_restful import Resource, Api
import prediction_MLR as pred
import ResolveAssistance as assist
import pickle

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'about':'Hello world!'}
    def post(self):
        some_json = request.get_json()
        return {'you sent': some_json}, 201

class Issue(Resource):
    def get(self, value):

        filename = 'finalized_model(without-outliers).sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        date=pred.PredictMLR(loaded_model, value)
        message={'issue': value,
                'predicted_resolution_date' : date}
        return message

class Assistance(Resource):
    def get(self, date):
        output_list=assist.resolve_planning_assistance(date)
        message={'now': date,'issues': output_list}
        return message

api.add_resource(HelloWorld, '/')
api.add_resource(Issue,'/issue/<string:value>/resolve-prediction')
api.add_resource(Assistance,'/release/<string:date>/resolved-since-now')

if __name__ == '__main__':
    app.run(debug=True)