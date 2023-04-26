import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("E:/diabetes.csv")


x_train,x_test,y_train,y_test=train_test_split(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']],df.Outcome,test_size=0.3)
model=LogisticRegression(max_iter=500)
model.fit(x_train,y_train)
print(model.predict([[6,148,72,35,0,33.6,0.627,50]]))
import streamlit as st
st.title("Diabetic Prediction")
st.subheader("USING MACHINE LEARNING")
v1=st.number_input("Enter Pregnency details ")
v2=st.number_input("Glucose Details")
v3=st.number_input("Blood Pressure")
v4=st.number_input("SkinThickness")
v5=st.number_input("Insulin")
v6=st.number_input("BMI")
v7=st.number_input("DiabetesPedigreeFunction")
v8=st.number_input("Age")
if st.button("SUBMIT"):
    x=model.predict([[v1,v2,v3,v4,v5,v6,v7,v8]])
    if x==1:
        st.error("Diabetics Detected")
    else:
        st.success("Diabetics is not detected")
