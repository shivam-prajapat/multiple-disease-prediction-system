# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 21:25:04 2025

@author: shiva
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# =======================
# Load the saved models
# =======================

diabetes_model = pickle.load(
    open("C:/Users/shiva/OneDrive/Desktop/deploy ML model/diabetes_model.sav", "rb")
)

heart_model = pickle.load(
    open("C:/Users/shiva/OneDrive/Desktop/deploy ML model/trained_model.sav", "rb")
)

parkinsons_model = pickle.load(
    open("C:/Users/shiva/OneDrive/Desktop/deploy ML model/parkinsons_model.sav", "rb")
)


# =======================
# Sidebar Navigation
# =======================

with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )


# =======================
# Diabetes Prediction
# =======================

if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[
            float(Pregnancies), float(Glucose), float(BloodPressure),
            float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), float(Age)
        ]])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The Person is Diabetic'
        else:
            diab_diagnosis = 'The Person is not Diabetic'

    st.success(diab_diagnosis)


# =======================
# Heart Disease Prediction
# =======================

if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')

    with col3:
        cp = st.text_input('Chest Pain Type')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Cholesterol')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar')

    with col1:
        restecg = st.text_input('Rest ECG')

    with col2:
        thalach = st.text_input('Maximum Heart Rate')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('Oldpeak')

    with col2:
        slope = st.text_input('Slope')

    with col3:
        ca = st.text_input('Number of Major Vessels')

    with col1:
        thal = st.text_input('Thal')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[
            float(age), float(sex), float(cp), float(trestbps),
            float(chol), float(fbs), float(restecg),
            float(thalach), float(exang), float(oldpeak),
            float(slope), float(ca), float(thal)
        ]])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The Person has Heart Disease'
        else:
            heart_diagnosis = 'The Person does not have Heart Disease'

    st.success(heart_diagnosis)


# =======================
# Parkinsons Prediction
# =======================

if selected == 'Parkinsons Prediction':

    st.title('Parkinsons Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col1:
        jitter = st.text_input('MDVP:Jitter(%)')

    with col2:
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')

    with col3:
        rap = st.text_input('MDVP:RAP')

    with col1:
        ppq = st.text_input('MDVP:PPQ')

    with col2:
        ddp = st.text_input('Jitter:DDP')

    with col3:
        shimmer = st.text_input('MDVP:Shimmer')

    with col1:
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')

    with col2:
        apq3 = st.text_input('Shimmer:APQ3')

    with col3:
        apq5 = st.text_input('Shimmer:APQ5')

    with col1:
        apq = st.text_input('MDVP:APQ')

    with col2:
        dda = st.text_input('Shimmer:DDA')

    with col3:
        nhr = st.text_input('NHR')

    with col1:
        hnr = st.text_input('HNR')

    with col2:
        rpde = st.text_input('RPDE')

    with col3:
        dfa = st.text_input('DFA')

    with col1:
        spread1 = st.text_input('Spread1')

    with col2:
        spread2 = st.text_input('Spread2')

    with col3:
        d2 = st.text_input('D2')

    with col1:
        ppe = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button('Parkinsons Test Result'):
        parkinsons_prediction = parkinsons_model.predict([[
            float(fo), float(fhi), float(flo), float(jitter),
            float(jitter_abs), float(rap), float(ppq), float(ddp),
            float(shimmer), float(shimmer_db), float(apq3),
            float(apq5), float(apq), float(dda), float(nhr),
            float(hnr), float(rpde), float(dfa), float(spread1),
            float(spread2), float(d2), float(ppe)
        ]])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = 'The Person has Parkinson’s Disease'
        else:
            parkinsons_diagnosis = 'The Person does not have Parkinson’s Disease'

    st.success(parkinsons_diagnosis)
