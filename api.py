#            ------------------------------
#                      Instructions
#            ------------------------------

# Set to current working directory
    # cd C:\Users\Ariel\Google Drive\Personal\Programming\MLMoney\Midsem\Basic Version\
# Run app on python
    # python api.py
# Go to the url it specifies and follow the prompts
    # http://127.0.0.1:5000/
# Or, to test API on postman, send a POST request to
    # http://127.0.0.1:5000/api_call
# To test on the browser, enter the below payment details on stripe
    #ariel.serravalle@gmail.com
    #4242 4242 4242 4242


# This is a flask web app which allows a user to fill out a form or call an API
# This specific startup is a human resources one which predicts job satisfaction and performance
# Request will have JSON file to test on ML Model
# Return prediction and probability
# Future versions will have Firebase and Stripe compatability

#            ------------------------------
#                Import Classes, Models
#            ------------------------------

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import sys

# File management
import json
from flask import request, jsonify
from sklearn.externals import joblib
from starlette.responses import HTMLResponse, JSONResponse

# Machine Learning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Payment
import stripe
public_key = 'pk_test_EhTx8BiFa6qxrItPF2UipwJi00Oqk3UZQl'
stripe.api_key = "sk_test_f0LSsYoe1S5pWAsfLsugc6BJ00sKAIsv4X"

# App
app = Flask(__name__)
app.config["DEBUG"] = True

#            ------------------------------
#                      Define Routes
#            ------------------------------

# Home
@app.route('/')
def index():
    # Line 24 of index.html
        # Redirects user to /payment when you succeed with payment
    return render_template('index.html', public_key=public_key)

# Payment
@app.route('/payment', methods = ['POST'])
def payment():
    #customer info
    customer = stripe.Customer.create(email=request.form['stripeEmail'],
                                        source=request.form['stripeToken'])
    #payment info
    charge = stripe.Charge.create(
        customer = customer.id,
        amount=500,
        currency = 'usd',
        description = 'Predict candidate performance and satisfaction'
    )

    # store charge ID in our database
    #new_order.charge_id = charge.id

    # Redirect to Thank You page after payment
    return redirect(url_for('thankyou'))

@app.route('/thankyou')
def thankyou():
    # thankyou.html should allow the user to fill out a form
        # When we have FireBase, we should verify the charge_id before allowing API call
    return render_template('thankyou.html')

# Get the prediction from an upload
@app.route('/results', methods=['POST'])
def receive_form():
    Age = int(request.form['Age'])
    BusinessTravel = request.form['BusinessTravel']
    Department = request.form['Department']
    DistanceFromHome = int(request.form['DistanceFromHome'])
    Education = request.form['Education']
    EducationField = request.form['EducationField']
    EnvironmentSatisfaction = int(request.form['EnvironmentSatisfaction'])
    Gender = request.form['Gender']
    JobRole = request.form['JobRole']
    MaritalStatus = request.form['MaritalStatus']
    MonthlyIncome = int(request.form['MonthlyIncome'])
    NumCompaniesWorked = int(request.form['NumCompaniesWorked'])
    OverTime = request.form['OverTime']
    PercentSalaryHike = int(request.form['PercentSalaryHike'])
    StandardHours = int(request.form['StandardHours'])
    StockOptionLevel_0to3 = int(request.form['StockOptionLevel_0to3'])
    TotalWorkingYears = int(request.form['TotalWorkingYears'])
    WorkLifeBalance_1to5 = int(request.form['WorkLifeBalance_1to5'])
    YearsAtCompany = int(request.form['YearsAtCompany'])
    YearsInCurrentRole = int(request.form['YearsInCurrentRole'])
    YearsSinceLastPromotion = int(request.form['YearsSinceLastPromotion'])
    YearsWithCurrManager = int(request.form['YearsWithCurrManager'])

    # Store values into dataframe
    xobs_array = [Age ,BusinessTravel ,Department ,DistanceFromHome ,Education ,EducationField ,EnvironmentSatisfaction ,Gender ,JobRole ,MaritalStatus ,MonthlyIncome ,NumCompaniesWorked ,OverTime ,PercentSalaryHike ,StandardHours ,StockOptionLevel_0to3 ,TotalWorkingYears ,WorkLifeBalance_1to5 ,YearsAtCompany ,YearsInCurrentRole ,YearsSinceLastPromotion ,YearsWithCurrManager]
    cols = ['Age','BusinessTravel','Department','DistanceFromHome','Education','EducationField','EnvironmentSatisfaction','Gender','JobRole','MaritalStatus','MonthlyIncome','NumCompaniesWorked','OverTime','PercentSalaryHike','StandardHours','StockOptionLevel_0to3','TotalWorkingYears','WorkLifeBalance_1to5','YearsAtCompany', 'YearsInCurrentRole','YearsSinceLastPromotion', 'YearsWithCurrManager']
    xobs = pd.DataFrame([xobs_array], columns = cols)

    # Make prediction
    predJS, predPR = predict(xobs)
    return render_template('results.html', predJS=predJS, predPR=predPR )

# Or an API
@app.route("/api_call", methods=['POST'])
def api_call():
    # Run prediction on JSON file from API call
    xobs = pd.DataFrame([request.json])

    # Make prediction
    predJS, predPR = predict(xobs)

    return jsonify(JobSatisfaction = predJS,
                    PerformanceRating = predPR)

#            ------------------------------
#                      Predictive Algorithm
#            ------------------------------

def predict(xobs):
    # Load, pipeline, and make predictions on Xobs
    model1 = joblib.load('model_jobsat.pkl')
    model2 = joblib.load('model_perform.pkl')
    sc = joblib.load('std_scaler.bin')
    ohe = joblib.load('ohe.bin')

    # Preprocess the example observation
    cat_features = [f for f in xobs.columns if (np.dtype(xobs[f]) == 'object')]
    num_features = [f for f in xobs.columns if (np.dtype(xobs[f]) != 'object')]

    # OHE categorical
    df_cat2 = xobs.loc[:,cat_features]
    df_cat2 = pd.DataFrame(ohe.transform(df_cat2).toarray())

    # Standardise numerical columns
    df_num2 = xobs.loc[:,num_features]
    test = pd.concat([df_num2, df_cat2], axis = 1)
    print('train shape = ',test.shape)
    test = pd.DataFrame(sc.transform(test), columns = test.columns)

    # Prediction
    predJS = model1.predict(test)[0][0]
    predJS = round(float(predJS), 2)
    predPR = model2.predict(test)[0][0]
    predPR = round(float(predPR), 2)

    return predJS, predPR

#            ------------------------------
#                      Firebase
#            ------------------------------

# Incomplete Firebase functionality to store user data
'''
from flask import Flask, request
from flask_firebase import FirebaseAuth
from flask_login import LoginManager, UserMixin, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy

app.config.from_object(...)

db = SQLAlchemy(app)
auth = FirebaseAuth(app)
login_manager = LoginManager(app)

app.register_blueprint(auth.blueprint, url_prefix='/auth')

'''

#            ------------------------------
#                      Stripe
#            ------------------------------



app.run()
