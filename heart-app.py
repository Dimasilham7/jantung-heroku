import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# heart disease Prediction App
This app predicts the **heart** disease!
Data obtained from the [heart disease library](https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv) by roni.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[CSV input file clean](https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        fbs = st.sidebar.slider('Fbs',0,1) #dari encode 
        exang = st.sidebar.slider ('exang(type)', 0,1)#dari encode
        cp = st.sidebar.slider ('cp(type)', 0,3)
        age = st.sidebar.slider ('age(person)', 29,77)
        trestbps = st.sidebar.slider ('trestbps(mm HG)', 94,200 )
        chol = st.sidebar.slider ('chol(mg/dl)', 126,564)
        restecg = st.sidebar.slider ('restecg(Value ecg)', 0,2)
        thalach = st.sidebar.slider ('thalach(heart rate)', 71,202)
        oldpeak = st.sidebar.slider ('oldpeak(ST depression induced by exercise relative to rest)', 0.0,3.5,6.2)
        slope = st.sidebar.slider ('slope(value)', 0,2)
        ca = st.sidebar.slider ('ca(number of major vessels)', 0,3)
        thal = st.sidebar.slider ('cp(level)', 3, 6, 7)
        target = st.sidebar.slider('target',0,1)
        data = {'fbs': fbs,
                'exang': exang,
                'cp':cp,
                'age':age,
                'trestbps':trestbps,
                'chol':chol,
                'restecg':restecg,
                'thalach':thalach,
                'oldpeak':oldpeak,
                'slope':slope,
                'ca':ca,
                'thal':thal,
                'target':target}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
heart_raw = pd.read_csv('heart.csv')
heart = heart_raw.drop(columns=['sex'])
df = pd.concat([input_df,heart],axis=0)

encode = ['exang','fbs']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Fitur yang diinput user')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('File CSV yang terupload. dibawah adalah contoh parameter yang ditampilkan.')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('heart.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
sex_person = np.array(['male', 'female'])
st.write(sex_person[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)