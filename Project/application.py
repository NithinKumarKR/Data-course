#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:54:25 2023

@author: nithinkumar
"""

from sklearn.preprocessing import StandardScaler


import pickle

from flask import Flask,render_template,request,jsonify

app=Flask(__name__)

@app.route('/' )
def home():
    return render_template('index.html')

stand_scaler=pickle.load(open('/Users/nithinkumar/Desktop/Personal Studies/Data-course/Standard_scaler.pkl','rb'))

model=pickle.load(open('/Users/nithinkumar/Desktop/Personal Studies/Data-course/support_vector_classifier.pkl','rb'))


@app.route('/predict',methods=['GET','POST'])
def Working():
    if request.method=='POST':
        Full_name = request.form.get('Full_name')
        Phone_number = request.form.get('Phone_number')
        house_number = request.form.get('house_number')
        address_line_one = request.form.get('address_line_one')
        address_line_two = request.form.get('address_line_two')
        landmark = request.form.get('landmark')
        postal_code = request.form.get('postal_code')
        
        Pregnancies = request.form.get('Pregnancies')
        Glucose = request.form.get('Glucose')
        BloodPressure = request.form.get('BloodPressure')
        SkinThickness = request.form.get('SkinThickness')
        Insulin = request.form.get('Insulin')
        BMI = request.form.get('BMI')
        DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
        Age = request.form.get('Age')
        
        Scaled_data=stand_scaler.transform([[Pregnancies,
                                                           Glucose,
                                                           BloodPressure,
                                                           SkinThickness,
                                                           Insulin,
                                                           BMI,
                                                           DiabetesPedigreeFunction,
                                                           Age]])
        result=model.predict(Scaled_data)[0]
        
        
      
        return  render_template('result.html' , result = result)
    else:
        return render_template('home.html')
    #if(request.method == 'POST'):
    #    return render_template('index.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
