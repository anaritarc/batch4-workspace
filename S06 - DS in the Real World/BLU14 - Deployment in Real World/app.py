import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict



########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open(os.path.join('columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################
########################################
# Input validation functions


def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation_id" not in request:
        response={}
        response['observation_id'] = None
        response['error']= "observation_id"
    
    elif "data" not in request:
        response={}
        response['observation_id'] = request['observation_id']
        response['error']= "data"
    
       
    # doesn't have age
    
    elif "age" not in request['data']:
        response={}
        response['observation_id'] = request['observation_id']
        response['error']= "age"  
        
        

    # bloodpressure assert 4
    elif "bloodpressure" in request['data']:
        response={}
        response['observation_id'] = request['observation_id']
        response['error']= "bloodpressure" 
    # sex assert 5 
    elif int(request['data']["sex"]) > 2:

        response = {"error": ('sex', '3')}

        
    # Hello world
    
    elif not isinstance(request['data']['ca'] , int):
            response = {"error": ('ca', 'Hello world')}

        
    #age

    elif request['data']['age'] < 0:
            response = {"error": ('age', str(request['data']['age']))}
           
    elif request['data']['age'] >120:
            response = {"error": ('age', str(request['data']['age']))}

            
    # trstbps
    elif request['data']['trestbps'] < 40:
            response = {"error": ('trestbps', str(request['data']['trestbps']))}
            
    elif request['data']['trestbps'] >200:
            response = {"error": ('trestbps', str(request['data']['trestbps']))}

    # oldpeak
    
    elif request['data']['oldpeak'] >10:
            response = {"error": ('oldpeak', str(request['data']['oldpeak']))}


    
    return response


# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    
    response = check_request(obs_dict)
    if "error" not in check:
        return jsonify(response)

    _id = obs_dict['observation_id']
    observation = obs_dict['data']
    
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'prediction': bool(prediction), 'probability': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)



@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


    
if __name__ == "__main__":
    app.run()

